#!/usr/bin/env python3
"""Run scope / typical / hybrid retrieval ablations without LLM calls."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.journal_store import JournalStore
from src.journals.typical_abstract_store import TypicalAbstractStore
from src.journals.vector_searcher import FaissIndex, VectorSearcher
from src.journals.accepted_paper_store import AcceptedPaperStore
from src.papers.paper_model import PaperProfile
from src.ranker.feature_builder import FEATURE_NAMES
from src.ranker.rule_scorer import RuleScorer
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.embedding_retriever import EmbeddingRetriever
from src.retriever.typical_abstract_retriever import (
    TypicalAbstractBM25Retriever,
    TypicalAbstractEmbeddingRetriever,
    TypicalAbstractTextRetriever,
)
from src.utils.embedding import OllamaEmbedding


VARIANTS = (
    "accepted",
    "scope_accepted",
    "full_hybrid",
)


class CachedEmbeddingClient:
    """Small run-local cache so variant ablations reuse identical query embeddings."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.cache = {}
        self.query_cache = {}

    def embed(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.wrapped.embed(text)
        return self.cache[text]

    def embed_batch(self, texts, concurrency: int = 1, timeout: float = 60.0):
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str):
        if text not in self.query_cache:
            query_method = getattr(self.wrapped, "embed_query", None)
            self.query_cache[text] = (
                query_method(text)
                if callable(query_method)
                else self.wrapped.embed(text)
            )
        return self.query_cache[text]


def route_config_for_mode(mode: str) -> dict[str, int]:
    config = {
        "title": {"bm25": 22, "vector": 22, "text": 16},
        "abstract": {"bm25": 28, "vector": 28, "text": 14},
        "full": {"bm25": 32, "vector": 32, "text": 16},
    }
    return config.get(mode, config["abstract"]).copy()


def build_route_results_for_variant(
    generator: CandidateGenerator,
    variant: str,
    rich_query: str,
    paper_profile: PaperProfile,
    cfg: dict[str, int],
    weights: dict[str, float],
):
    """Build route results for one ablation variant.

    `typical` intentionally excludes scope routes and identity_anchor so it
    measures generated semantic anchors alone.
    """
    if variant == "scope":
        return generator._scope_route_results(rich_query, paper_profile, cfg, weights)
    if variant in ("hybrid", "full_hybrid"):
        # full_hybrid 与 hybrid 在 CandidateGenerator 里语义相同 —— scope + typical
        # + accepted + identity_anchor 全部 routes,前提是对应 retriever 都注入。
        return generator._hybrid_route_results(rich_query, paper_profile, cfg, weights)

    typical_routes = _typical_routes(generator, rich_query, cfg, weights)
    accepted_routes = _accepted_routes(generator, rich_query, cfg, weights)
    scope_routes_factory = lambda: generator._scope_route_results(
        rich_query, paper_profile, cfg, weights
    )

    if variant == "typical":
        return typical_routes
    if variant == "accepted":
        return accepted_routes
    if variant == "scope_typical":
        return {**scope_routes_factory(), **typical_routes}
    if variant == "scope_accepted":
        return {**scope_routes_factory(), **accepted_routes}
    if variant == "typical_accepted":
        return {**typical_routes, **accepted_routes}

    raise ValueError(f"Unknown ablation variant: {variant}")


def _typical_routes(generator, rich_query, cfg, weights):
    route_results = {}
    if generator.typical_bm25_retriever:
        route_results["typical_bm25"] = (
            generator.typical_bm25_retriever.retrieve(rich_query, top_k=cfg["bm25"]),
            weights["bm25"],
        )
    if generator.typical_embedding_retriever:
        route_results["typical_vector"] = (
            generator.typical_embedding_retriever.retrieve(rich_query, top_k=cfg["vector"]),
            weights["vector"],
        )
    if generator.typical_text_retriever:
        route_results["typical_text"] = (
            generator.typical_text_retriever.retrieve(rich_query, top_k=cfg["text"]),
            weights["text"],
        )
    return route_results


def _accepted_routes(generator, rich_query, cfg, weights):
    route_results = {}
    if generator.accepted_bm25_retriever:
        route_results["accepted_bm25"] = (
            generator.accepted_bm25_retriever.retrieve(
                rich_query, top_k=cfg.get("accepted_bm25", cfg["bm25"]),
            ),
            weights["bm25"],
        )
    if generator.accepted_embedding_retriever:
        route_results["accepted_vector"] = (
            generator.accepted_embedding_retriever.retrieve(
                rich_query, top_k=cfg.get("accepted_vector", cfg["vector"]),
            ),
            weights["vector"],
        )
    return route_results


def generate_candidates_for_variant(
    generator: CandidateGenerator,
    query_text: str,
    paper_profile: PaperProfile,
    variant: str,
    mode: str,
    candidate_top_k: int,
    route_config: dict[str, int] | None = None,
):
    rich_query = generator._build_rich_query(query_text, paper_profile)
    cfg = (route_config or generator._route_config_for_mode(mode)).copy()
    weights = generator._weights_for_profile(paper_profile)
    route_results = build_route_results_for_variant(
        generator,
        variant,
        rich_query,
        paper_profile,
        cfg,
        weights,
    )
    return generator._merge_route_results(route_results, top_k=candidate_top_k)


def evaluate_variant(
    papers: Sequence[dict],
    generator: CandidateGenerator,
    scorer: RuleScorer,
    journal_name_to_id: dict[str, str],
    variant: str,
    mode: str = "abstract",
    candidate_top_k: int = 50,
    show_progress: bool = False,
    route_config: dict[str, int] | None = None,
    accepted_paper_store: AcceptedPaperStore | None = None,
) -> dict:
    retrieval = _new_metric_accumulator()
    rule = _new_metric_accumulator()
    paper_results = []
    miss_stage_counts: dict[str, int] = {}
    route_attribution: dict[str, dict[str, int]] = {}
    baseline_final_hit_at_5 = 0
    missing_target_count = 0
    evaluated_count = 0

    progress_bar = tqdm(
        total=len(papers),
        desc=f"{variant} ablation",
        unit="paper",
        disable=not show_progress,
    )
    try:
        for paper in papers:
            target_id = _target_journal_id(paper, journal_name_to_id)
            if not target_id:
                missing_target_count += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    _progress_snapshot(
                        evaluated_count=evaluated_count,
                        missing_target_count=missing_target_count,
                        retrieval=retrieval,
                        rule=rule,
                        baseline_final_hit_at_5=baseline_final_hit_at_5,
                        miss_stage_counts=miss_stage_counts,
                    )
                )
                continue

            evaluated_count += 1
            profile = paper_profile_from_metadata(paper)
            query_text = " ".join(part for part in [profile.title, profile.abstract] if part)
            candidates, retrieval_trace = generate_candidates_for_variant(
                generator,
                query_text,
                profile,
                variant=variant,
                mode=mode,
                candidate_top_k=candidate_top_k,
                route_config=route_config,
            )

            candidate_ids = [journal.journal_id for journal in candidates]
            retrieval_rank = _rank_of(target_id, candidate_ids)
            _accumulate_metrics(retrieval, retrieval_rank)

            rule_ranked = scorer.rank(
                candidates,
                profile,
                top_k=len(candidates),
                retrieval_trace=retrieval_trace,
            )
            rule_ids = [journal.journal_id for journal, _, _ in rule_ranked]
            rule_rank = _rank_of(target_id, rule_ids)
            _accumulate_metrics(rule, rule_rank)
            final_hit_5 = bool(paper.get("final_hit_5", False))
            if final_hit_5:
                baseline_final_hit_at_5 += 1

            miss_stage = _miss_stage(retrieval_rank, rule_rank, final_hit_5)
            miss_stage_counts[miss_stage] = miss_stage_counts.get(miss_stage, 0) + 1
            target_attribution = _target_route_attribution(retrieval_trace, target_id)
            _accumulate_route_attribution(
                route_attribution,
                target_attribution,
                retrieval_rank=retrieval_rank,
                rule_rank=rule_rank,
            )

            # 4.1.e:把 LTR 训练特征注入 trace,再把每条候选的 features
            # 落到 paper_result["candidate_features"]。caller 在 4.1.f
            # 会拿这个 JSON 转成 LTR 训练数据。
            rule_ranks_map = {
                jid: idx + 1 for idx, (jid, *_) in enumerate(
                    ((j.journal_id, j) for j, _, _ in rule_ranked)
                )
            }
            rule_scores_map = {
                jid: float(score) for jid, (_, score, _) in (
                    (j.journal_id, (j, s, _)) for j, s, _ in rule_ranked
                )
            }
            try:
                generator.attach_features(
                    trace=retrieval_trace,
                    paper_profile=profile,
                    rule_ranks=rule_ranks_map,
                    rule_scores=rule_scores_map,
                    accepted_paper_store=accepted_paper_store,
                )
            except Exception as exc:  # noqa: BLE001
                # features 注入失败不能让 ablation 整体失败
                print(f"[warn] attach_features failed for paper {paper.get('title', '')}: {exc}", file=sys.stderr)
                candidate_features = {}
            else:
                candidate_features = {
                    jid: entry.get("features", [])
                    for jid, entry in retrieval_trace.items()
                }

            paper_results.append({
                "title": paper.get("title", ""),
                "venue": paper.get("venue", ""),
                "target_journal_id": target_id,
                "retrieval_rank": retrieval_rank,
                "rule_rank": rule_rank,
                "baseline_final_hit_5": final_hit_5,
                "miss_stage": miss_stage,
                "target_route_attribution": target_attribution,
                "retrieval_top5": candidate_ids[:5],
                "rule_top5": rule_ids[:5],
                "rule_top20": rule_ids[:20],  # per plan 4.2:hard negative 用 Rule Top20,不是 top5
                "candidate_features": candidate_features,
            })
            progress_bar.update(1)
            progress_bar.set_postfix(
                _progress_snapshot(
                    evaluated_count=evaluated_count,
                    missing_target_count=missing_target_count,
                    retrieval=retrieval,
                    rule=rule,
                    baseline_final_hit_at_5=baseline_final_hit_at_5,
                    miss_stage_counts=miss_stage_counts,
                )
            )
    finally:
        progress_bar.close()

    finalized_retrieval = _finalize_metrics(retrieval, evaluated_count)
    finalized_rule = _finalize_metrics(rule, evaluated_count)
    return {
        "variant": variant,
        "evaluated_count": evaluated_count,
        "missing_target_count": missing_target_count,
        "candidate_top_k": candidate_top_k,
        "coarse_hit_at_50": int(finalized_retrieval.get("Hit@50", 0)),
        "rule_hit_at_20": int(finalized_rule.get("Hit@20", 0)),
        "baseline_final_hit_at_5": baseline_final_hit_at_5,
        "miss_stage_counts": miss_stage_counts,
        "route_attribution": route_attribution,
        "retrieval": finalized_retrieval,
        "rule": finalized_rule,
        "feature_names": list(FEATURE_NAMES),
        "paper_results": paper_results,
    }


def paper_profile_from_metadata(paper: dict) -> PaperProfile:
    snapshot = paper.get("paper_profile_snapshot")
    if isinstance(snapshot, dict):
        return paper_profile_from_snapshot(snapshot, paper)

    return PaperProfile(
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        research_area=_as_list(paper.get("research_area", [])),
        keywords=_as_list(paper.get("keywords", [])),
        techniques=_as_list(paper.get("techniques", [])),
        datasets=_as_list(paper.get("datasets", [])),
        evaluation_metrics=_as_list(paper.get("evaluation_metrics", [])),
        novelty_type=paper.get("novelty_type", "") or "",
    )


def paper_profile_from_snapshot(snapshot: dict, paper: dict) -> PaperProfile:
    return PaperProfile(
        title=snapshot.get("title") or paper.get("title", ""),
        abstract=snapshot.get("abstract") or paper.get("abstract", ""),
        research_area=_as_list(snapshot.get("research_area", [])),
        method_type=snapshot.get("method_type") or "method",
        paper_type=snapshot.get("paper_type") or "application",
        keywords=_as_list(snapshot.get("keywords", [])),
        novelty=snapshot.get("novelty", "") or "",
        application_domain=_as_list(snapshot.get("application_domain", [])),
        difficulty_level=snapshot.get("difficulty_level") or "medium",
        style=snapshot.get("style") or "journal_like",
        techniques=_as_list(snapshot.get("techniques", [])),
        datasets=_as_list(snapshot.get("datasets", [])),
        evaluation_metrics=_as_list(snapshot.get("evaluation_metrics", [])),
        novelty_type=snapshot.get("novelty_type", "") or "",
        quality_level=snapshot.get("quality_level"),
        paper_strength=snapshot.get("paper_strength"),
        readiness=snapshot.get("readiness"),
        ccf_research_area=_as_list(snapshot.get("ccf_research_area", [])),
    )


def load_papers(path: str, limit: int | None = None) -> list[dict]:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
            if limit and len(papers) >= limit:
                break
    return papers


def load_comparable_eval_papers(
    papers_path: str,
    baseline_eval_path: str,
    limit: int | None = None,
) -> list[dict]:
    """Use a completed full evaluation as the denominator for ablations.

    Two paper-id field formats are accepted (top-level takes precedence):
      - top-level "arxiv"  (newer 540 corpus)
      - "external_ids.arXiv"  (older format, e.g. light30/full-v2-90)

    Two JSON layouts are accepted:
      - top-level "paper_results"  (legacy, pre-variants)
      - "variants.<name>.paper_results"  (current, full_hybrid etc.)

    Both degrade gracefully via title+venue fallback if arxiv lookup fails.
    """
    metadata_papers = load_papers(papers_path)
    metadata_by_arxiv: dict[str, dict] = {}
    metadata_by_title_venue: dict[tuple[str, str], dict] = {}
    for paper in metadata_papers:
        arxiv_id = (
            str(paper.get("external_ids", {}).get("arXiv", "") or "")
            or str(paper.get("arxiv", "") or "")
        )
        if arxiv_id:
            metadata_by_arxiv[arxiv_id] = paper
        metadata_by_title_venue[(
            _normalize_title(paper.get("title", "")),
            _normalize_name(paper.get("venue", "")),
        )] = paper

    with open(baseline_eval_path, "r", encoding="utf-8") as f:
        baseline_eval = json.load(f)

    # Accept both layouts: top-level "paper_results" or "variants.<name>.paper_results"
    paper_results = baseline_eval.get("paper_results")
    if paper_results is None:
        for variant_payload in baseline_eval.get("variants", {}).values():
            if isinstance(variant_payload, dict) and "paper_results" in variant_payload:
                paper_results = variant_payload["paper_results"]
                break
    if paper_results is None:
        paper_results = []

    comparable = []
    for result in paper_results:
        arxiv_id = str(result.get("arxiv", "") or "")
        metadata = metadata_by_arxiv.get(arxiv_id) if arxiv_id else None
        if metadata is None:
            metadata = metadata_by_title_venue.get((
                _normalize_title(result.get("title", "")),
                _normalize_name(result.get("venue", "")),
            ))
        if metadata is None:
            metadata = {}

        merged = dict(metadata)
        merged["title"] = metadata.get("title") or result.get("title", "")
        merged["venue"] = metadata.get("venue") or result.get("venue", "")
        merged["paper_profile_snapshot"] = result.get("paper_profile_snapshot", {})
        merged["final_hit_5"] = _baseline_final_hit_at_5(result)
        merged["baseline_result"] = {
            "hit_5": merged["final_hit_5"],
            "coarse_hit": bool(result.get("coarse_hit", False)),
            "coarse_hit_in_rule_top20": bool(result.get("coarse_hit_in_rule_top20", False)),
            "miss_stage": result.get("venue_diagnostic", {}).get("miss_stage"),
        }
        comparable.append(merged)
        if limit and len(comparable) >= limit:
            break

    return comparable


def build_candidate_generator(app_config: dict, include_vector: bool = False) -> CandidateGenerator:
    data_config = app_config.get("data", {})
    retrieval_config = app_config.get("candidate_generator", {})

    store = JournalStore(data_config.get("journal_store_path", "data/processed/journals.jsonl"))
    store.load()

    bm25 = BM25Retriever(store)
    bm25.build_index()

    embedding_client = None
    embedding_retriever = None
    if include_vector:
        faiss_idx = FaissIndex(
            data_config.get("faiss_index_path", "data/processed/journals_index.faiss"),
            data_config.get("metadata_path", "data/processed/journals_metadata.parquet"),
        )
        faiss_idx.load()
        if faiss_idx.is_loaded:
            store.set_vector_searcher(VectorSearcher(faiss_idx))
            embedding_client = CachedEmbeddingClient(
                OllamaEmbedding(
                    base_url=app_config["ollama"]["base_url"],
                    model=app_config["ollama"]["embedding_model"],
                    timeout=app_config.get("ollama", {}).get(
                        "timeout_seconds", 60
                    ),
                    query_instruction=app_config.get("ollama", {}).get(
                        "embedding_query_instruction"
                    ),
                )
            )
            embedding_retriever = EmbeddingRetriever(store, embedding_client)

    typical_store = TypicalAbstractStore(
        data_config.get("typical_abstracts_dir", "data/typical_abstracts")
    )
    typical_store.load()
    typical_bm25 = TypicalAbstractBM25Retriever(typical_store, store)
    typical_bm25.build_index()
    typical_text = TypicalAbstractTextRetriever(typical_store, store)
    typical_embedding = None
    if include_vector and embedding_client:
        typical_embedding = TypicalAbstractEmbeddingRetriever(
            abstract_store=typical_store,
            journal_store=store,
            embedding_client=embedding_client,
            faiss_path=data_config.get(
                "typical_abstracts_faiss_path",
                "data/processed/typical_abstracts_index.faiss",
            ),
            metadata_path=data_config.get(
                "typical_abstracts_metadata_path",
                "data/processed/typical_abstracts_metadata.parquet",
            ),
        )

    # accepted-paper 路由 (任务 3.3 修复:这里之前漏了接线,
    # 导致 accepted / scope_accepted / full_hybrid variant 全部静默退化)
    from src.journals.accepted_paper_store import AcceptedPaperStore
    from src.retriever.accepted_paper_retriever import (
        AcceptedPaperBM25Retriever,
        AcceptedPaperEmbeddingRetriever,
    )

    accepted_bm25 = None
    accepted_embedding = None
    accepted_store = AcceptedPaperStore(
        accepted_dir=data_config.get("accepted_papers_dir", "data/accepted_papers")
    )
    accepted_store.load()
    if accepted_store.count > 0:
        accepted_bm25 = AcceptedPaperBM25Retriever(accepted_store, store)
        accepted_bm25.build_index()
        if include_vector and embedding_client:
            accepted_embedding = AcceptedPaperEmbeddingRetriever(
                accepted_store=accepted_store,
                journal_store=store,
                embedding_client=embedding_client,
                faiss_path=data_config.get(
                    "accepted_papers_faiss_path",
                    "data/processed/accepted_papers_index.faiss",
                ),
                metadata_path=data_config.get(
                    "accepted_papers_metadata_path",
                    "data/processed/accepted_papers_metadata.parquet",
                ),
            )
            if not accepted_embedding.is_available:
                accepted_embedding = None

    return CandidateGenerator(
        store,
        bm25,
        embedding_retriever,
        merge_weights=retrieval_config.get("merge_weights", {"bm25": 0.45, "vector": 0.35, "text": 0.20}),
        retrieval_target=retrieval_config.get("retrieval_target", "typical_abstracts"),
        typical_bm25_retriever=typical_bm25,
        typical_embedding_retriever=typical_embedding,
        typical_text_retriever=typical_text,
        accepted_bm25_retriever=accepted_bm25,
        accepted_embedding_retriever=accepted_embedding,
        hybrid_scope_weight=retrieval_config.get("hybrid_scope_weight", 0.75),
        hybrid_typical_weight=retrieval_config.get("hybrid_typical_weight", 0.25),
        identity_anchor_weight=retrieval_config.get("identity_anchor_weight", 0.03),
        accepted_paper_weight=retrieval_config.get("accepted_paper_weight", 0.20),
        fusion_strategy=retrieval_config.get("fusion_strategy", "weighted_minmax"),
        rrf_k=retrieval_config.get("rrf_k", 60),
        route_top_k=retrieval_config.get("route_top_k"),
    )


def journal_name_to_id_map(store: JournalStore) -> dict[str, str]:
    return {_normalize_name(journal.journal_name): journal.journal_id for journal in store.journals}


def run_ablation(
    papers_path: str,
    config_path: str,
    output_path: str,
    variants: Iterable[str] = VARIANTS,
    mode: str = "abstract",
    candidate_top_k: int = 50,
    limit: int | None = None,
    include_vector: bool = False,
    baseline_eval_path: str = "",
    show_progress: bool = True,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    generator = build_candidate_generator(app_config, include_vector=include_vector)
    scorer = RuleScorer(
        journals=generator.store.journals,
        weights=app_config.get("ranking", {}).get("rule_scorer", {}),
    )
    if baseline_eval_path:
        papers = load_comparable_eval_papers(papers_path, baseline_eval_path, limit=limit)
    else:
        papers = load_papers(papers_path, limit=limit)
    name_to_id = journal_name_to_id_map(generator.store)

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "papers_path": papers_path,
        "baseline_eval_path": baseline_eval_path,
        "config_path": config_path,
        "mode": mode,
        "candidate_top_k": candidate_top_k,
        "include_vector": include_vector,
        "denominator_source": "baseline_eval" if baseline_eval_path else "papers_metadata",
        "paper_count": len(papers),
        "variants": {},
    }
    for variant in variants:
        results["variants"][variant] = evaluate_variant(
            papers=papers,
            generator=generator,
            scorer=scorer,
            journal_name_to_id=name_to_id,
            variant=variant,
            mode=mode,
            candidate_top_k=candidate_top_k,
            show_progress=show_progress,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def print_summary(results: dict) -> None:
    print(f"Saved: {results.get('output_path', '')}")
    print("variant\teval\tcoarse@50\trule@5\trule@20\tbaseline_final@5\tret_mrr\trule_mrr\tret_ndcg5\trule_ndcg5")
    for variant, data in results["variants"].items():
        retrieval = data["retrieval"]
        rule = data["rule"]
        print(
            "\t".join([
                variant,
                str(data["evaluated_count"]),
                _fmt_metric(retrieval, "Hit@50"),
                _fmt_metric(rule, "Hit@5"),
                _fmt_metric(rule, "Hit@20"),
                str(data.get("baseline_final_hit_at_5", 0)),
                f"{retrieval['MRR']:.4f}",
                f"{rule['MRR']:.4f}",
                f"{retrieval['NDCG@5']:.4f}",
                f"{rule['NDCG@5']:.4f}",
            ])
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scope / typical / hybrid retrieval ablation.")
    parser.add_argument("--papers", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", default="abstract", choices=["title", "abstract", "full"])
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-vector", action="store_true", help="Use Ollama-backed vector routes.")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress bars.")
    parser.add_argument(
        "--baseline-eval",
        default="",
        help="Completed run_evaluation JSON to reuse denominator and paper_profile_snapshot.",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or f"data/evaluation/results/retrieval_ablation_{args.mode}_{timestamp}.json"
    results = run_ablation(
        papers_path=args.papers,
        config_path=args.config,
        output_path=output,
        variants=args.variants,
        mode=args.mode,
        candidate_top_k=args.candidate_top_k,
        limit=args.limit,
        include_vector=args.include_vector,
        baseline_eval_path=args.baseline_eval,
        show_progress=not args.no_progress,
    )
    results["output_path"] = output
    print_summary(results)


def _new_metric_accumulator() -> dict[str, float]:
    return {
        "Hit@1": 0,
        "Hit@3": 0,
        "Hit@5": 0,
        "Hit@10": 0,
        "Hit@20": 0,
        "Hit@50": 0,
        "MRR": 0.0,
        "NDCG@5": 0.0,
        "NDCG@10": 0.0,
        "NDCG@20": 0.0,
        "NDCG@50": 0.0,
    }


def _accumulate_metrics(metrics: dict[str, float], rank: int | None) -> None:
    for k in [1, 3, 5, 10, 20, 50]:
        if rank is not None and rank <= k:
            metrics[f"Hit@{k}"] += 1
    if rank is not None:
        metrics["MRR"] += 1.0 / rank
    for k in [5, 10, 20, 50]:
        metrics[f"NDCG@{k}"] += _single_relevant_ndcg(rank, k)


def _progress_snapshot(
    evaluated_count: int,
    missing_target_count: int,
    retrieval: dict[str, float],
    rule: dict[str, float],
    baseline_final_hit_at_5: int,
    miss_stage_counts: dict[str, int],
) -> dict[str, str]:
    denominator = max(evaluated_count, 1)
    return {
        "eval": str(evaluated_count),
        "missing": str(missing_target_count),
        "coarse@50": _fmt_count_rate(retrieval.get("Hit@50", 0), denominator),
        "rule@20": _fmt_count_rate(rule.get("Hit@20", 0), denominator),
        "base@5": _fmt_count_rate(baseline_final_hit_at_5, denominator),
        "ret_mrr": f"{retrieval.get('MRR', 0.0) / denominator:.3f}",
        "rule_mrr": f"{rule.get('MRR', 0.0) / denominator:.3f}",
        "ret_ndcg5": f"{retrieval.get('NDCG@5', 0.0) / denominator:.3f}",
        "miss_top50": str(miss_stage_counts.get("not_in_top50", 0)),
        "rule_supp": str(miss_stage_counts.get("rule_suppressed", 0)),
    }


def _finalize_metrics(metrics: dict[str, float], evaluated_count: int) -> dict[str, float]:
    if evaluated_count <= 0:
        return metrics
    finalized = metrics.copy()
    for key in ["MRR", "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50"]:
        finalized[key] = finalized[key] / evaluated_count
    return finalized


def _single_relevant_ndcg(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _rank_of(target_id: str, ranked_ids: Sequence[str]) -> int | None:
    for idx, journal_id in enumerate(ranked_ids, start=1):
        if journal_id == target_id:
            return idx
    return None


def _target_journal_id(paper: dict, journal_name_to_id: dict[str, str]) -> str:
    if paper.get("positive_journal_id"):
        return paper["positive_journal_id"]
    return journal_name_to_id.get(_normalize_name(paper.get("venue", "")), "")


def _baseline_final_hit_at_5(result: dict) -> bool:
    if "hit_5" in result:
        return bool(result.get("hit_5"))
    venue = _normalize_name(result.get("venue", ""))
    recommended = [
        _normalize_name(name)
        for name in result.get("recommended_journals", [])[:5]
    ]
    return bool(venue and venue in recommended)


def _miss_stage(
    retrieval_rank: int | None,
    rule_rank: int | None,
    final_hit_5: bool,
) -> str:
    if final_hit_5:
        return "baseline_final_hit"
    if retrieval_rank is None:
        return "not_in_top50"
    if rule_rank is None or rule_rank > 20:
        return "rule_suppressed"
    return "after_rule_top20_lost"


def _target_route_attribution(retrieval_trace: dict, target_id: str) -> dict:
    trace = retrieval_trace.get(target_id, {})
    routes = trace.get("routes", {})
    return {
        "retrieval_rank": trace.get("retrieval_rank"),
        "total_score": trace.get("total_score"),
        "primary_routes": trace.get("primary_routes", []),
        "routes": {
            route: {
                "rank": data.get("rank"),
                "raw_score": data.get("raw_score"),
                "normalized_score": data.get("normalized_score"),
                "weighted_score": data.get("weighted_score"),
            }
            for route, data in routes.items()
        },
    }


def _accumulate_route_attribution(
    accumulator: dict[str, dict[str, int]],
    target_attribution: dict,
    retrieval_rank: int | None,
    rule_rank: int | None,
) -> None:
    for route in target_attribution.get("routes", {}):
        item = accumulator.setdefault(
            route,
            {
                "seen": 0,
                "coarse_hit": 0,
                "rule_top20": 0,
            },
        )
        item["seen"] += 1
        if retrieval_rank is not None and retrieval_rank <= 50:
            item["coarse_hit"] += 1
        if rule_rank is not None and rule_rank <= 20:
            item["rule_top20"] += 1


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _normalize_title(title: str) -> str:
    return " ".join((title or "").strip().lower().split())


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _fmt_metric(metrics: dict[str, float], key: str) -> str:
    return str(int(metrics.get(key, 0)))


def _fmt_count_rate(count: float, denominator: int) -> str:
    return f"{int(count)}/{denominator} ({count * 100 / denominator:.1f}%)"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
期刊推荐系统评估脚本 - 交互式实验

支持模式：
- 标题模式 (title)
- 摘要模式 (abstract)
- 全文模式 (full) - 从PDF提取文本

评估指标：
- Hit@K (K=3/5/10 可选)
- Level Match Rate (论文质量与期刊等级匹配)
- 领域匹配准确率
- 分质量等级 Hit@5 (强/中/弱)
"""

import json
import sys
import os
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 导入推荐系统组件
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.papers.paper_model import PaperInput, PaperProfile
from src.recommender.pipeline import RecommenderPipeline
from src.papers.paper_parser import PaperParser
from src.papers.quality_assessor import PaperQualityAssessor
from src.journals.journal_store import JournalStore
from src.journals.vector_searcher import VectorSearcher, FaissIndex
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.embedding_retriever import EmbeddingRetriever
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.typical_abstract_retriever import (
    TypicalAbstractBM25Retriever,
    TypicalAbstractEmbeddingRetriever,
    TypicalAbstractTextRetriever,
)
from src.journals.typical_abstract_store import TypicalAbstractStore
from src.ranker.rule_scorer import RuleScorer
from src.ranker.llm_ranker import LLMRanker
from src.utils.llm_config import build_minimax_llm
from src.utils.embedding import OllamaEmbedding
from src.utils.text import clean_text
from src.utils.file_parser import extract_layout_blocks
from src.papers.section_splitter import build_paper_ast
from src.evaluation.benchmark_manifest import build_benchmark_manifest


BENCHMARK_PROFILE_INPUTS = {
    "light30": "data/evaluation/papers_metadata_light_30.jsonl",
    "full-v2": "data/evaluation/papers_metadata_v2.jsonl",
    "full-v2-90": "data/evaluation/papers_metadata_full_v2_90.jsonl",
    "holdout240": "data/evaluation/papers_metadata_holdout240.jsonl",
}


@dataclass
class EvaluationResult:
    """评估结果"""
    # 基本统计
    total_count: int
    mode: str  # title/abstract/full
    top_k: int

    # Hit@K
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    hit_at_10: int

    # 领域匹配
    area_match_count: int

    # 推荐期刊的专业领域命中（subject_tags 命中 research_area）
    area_subject_tag_match_count: int

    # Level Match Rate
    level_match_count: int

    # MRR 和 NDCG@5
    mrr: float = 0.0
    ndcg_at_5: float = 0.0

    # 新增指标
    ndcg_at_10: float = 0.0  # NDCG@10（真正跑满10个）
    area_hit_at_5: int = 0   # 同领域命中（推荐top5中有同领域期刊）
    area_hit_at_10: int = 0  # 同领域命中 top10
    level_hit_at_5: int = 0   # 同CCF档位命中（top5中有同CCF等级期刊）
    level_hit_at_10: int = 0  # 同CCF档位命中 top10
    acceptable_journal_hit_at_5: int = 0  # exact hit 或同领域且同CCF档位

    # 粗排命中统计
    coarse_hit_count: int = 0  # 粗排命中（实际期刊在 top 50 候选中）
    coarse_hit_in_rule_top10_count: int = 0  # 粗排候选在 RuleScorer top10 中
    coarse_hit_in_rule_top20_count: int = 0  # 粗排候选在 RuleScorer top20 中
    fallback_count: int = 0
    llm_success_count: int = 0
    empty_recommendation_count: int = 0

    # 分质量等级统计 (A/B/C/D)
    level_a_count: int = 0
    level_a_hit_at_5: int = 0
    level_b_count: int = 0
    level_b_hit_at_5: int = 0
    level_c_count: int = 0
    level_c_hit_at_5: int = 0
    level_d_count: int = 0
    level_d_hit_at_5: int = 0

    # 按领域统计
    by_area: dict = None
    by_level: dict = None

    # 每篇论文详情
    paper_results: list = None


def get_paper_quality_level(strength: float) -> str:
    """根据 paper_strength 划分质量等级"""
    if strength is None:
        return "unknown"
    if strength >= 0.65:
        return "strong"
    elif strength >= 0.50:
        return "medium"
    else:
        return "weak"


def calculate_mrr(relevant_rank: int) -> float:
    """计算倒数排名（rank 从 1 开始，未命中为 0）"""
    if relevant_rank <= 0:
        return 0.0
    return 1.0 / relevant_rank


def calculate_dcg_at_k(gains: list, k: int) -> float:
    """计算 DCG@k"""
    dcg = 0.0
    for i, gain in enumerate(gains[:k]):
        dcg += gain / (i + 1)  # 或使用 1/log2(i+2)，这里用位置折扣
    return dcg


def calculate_ndcg_at_k(recommended_gains: list, ideal_gains: list, k: int) -> float:
    """计算 NDCG@k（recommended_gains 和 ideal_gains 都是相关性分数列表）"""
    dcg = calculate_dcg_at_k(recommended_gains, k)
    idcg = calculate_dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_metrics(result: EvaluationResult) -> dict:
    """计算各项指标"""
    total = result.total_count if result.total_count > 0 else 1

    metrics = {
        "Hit@1": f"{result.hit_at_1}/{total} ({result.hit_at_1*100/total:.1f}%)",
        "Hit@3": f"{result.hit_at_3}/{total} ({result.hit_at_3*100/total:.1f}%)",
        "Hit@5": f"{result.hit_at_5}/{total} ({result.hit_at_5*100/total:.1f}%)",
        "Hit@10": f"{result.hit_at_10}/{total} ({result.hit_at_10*100/total:.1f}%)",
        "Area Match Rate": f"{result.area_match_count}/{total} ({result.area_match_count*100/total:.1f}%)",
        "Level Match Rate": f"{result.level_match_count}/{total} ({result.level_match_count*100/total:.1f}%)",
        "MRR": f"{result.mrr/total:.4f}" if total > 0 else "N/A",
        "NDCG@5": f"{result.ndcg_at_5/total:.4f}" if total > 0 else "N/A",
        "NDCG@10": f"{result.ndcg_at_10/total:.4f}" if total > 0 else "N/A",
        "同领域命中@5": f"{result.area_hit_at_5}/{total} ({result.area_hit_at_5*100/total:.1f}%)",
        "同领域命中@10": f"{result.area_hit_at_10}/{total} ({result.area_hit_at_10*100/total:.1f}%)",
        "同CCF档位@5": f"{result.level_hit_at_5}/{total} ({result.level_hit_at_5*100/total:.1f}%)",
        "同CCF档位@10": f"{result.level_hit_at_10}/{total} ({result.level_hit_at_10*100/total:.1f}%)",
        "可接受期刊命中@5": f"{result.acceptable_journal_hit_at_5}/{total} ({result.acceptable_journal_hit_at_5*100/total:.1f}%)",
    }

    # 分质量等级
    if result.level_a_count > 0:
        metrics["A级 Hit@5"] = f"{result.level_a_hit_at_5}/{result.level_a_count} ({result.level_a_hit_at_5*100/result.level_a_count:.1f}%)"
    if result.level_b_count > 0:
        metrics["B级 Hit@5"] = f"{result.level_b_hit_at_5}/{result.level_b_count} ({result.level_b_hit_at_5*100/result.level_b_count:.1f}%)"
    if result.level_c_count > 0:
        metrics["C级 Hit@5"] = f"{result.level_c_hit_at_5}/{result.level_c_count} ({result.level_c_hit_at_5*100/result.level_c_count:.1f}%)"
    if result.level_d_count > 0:
        metrics["D级 Hit@5"] = f"{result.level_d_hit_at_5}/{result.level_d_count} ({result.level_d_hit_at_5*100/result.level_d_count:.1f}%)"

    return metrics


def load_papers_metadata(path: str) -> list:
    """加载论文元数据"""
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def _snapshot_match_key(title: str, venue: str) -> tuple[str, str]:
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())

    return normalize(title), normalize(venue)


def attach_baseline_profile_snapshots(papers: list, baseline_eval_path: str) -> list:
    """Attach fixed profiles from a completed evaluation to the current benchmark."""
    with open(baseline_eval_path, "r", encoding="utf-8") as f:
        baseline_eval = json.load(f)

    snapshots = {}
    for result in baseline_eval.get("paper_results", []):
        snapshot = result.get("paper_profile_snapshot")
        if isinstance(snapshot, dict) and snapshot:
            snapshots[_snapshot_match_key(result.get("title", ""), result.get("venue", ""))] = snapshot

    attached = []
    missing = []
    for paper in papers:
        key = _snapshot_match_key(paper.get("title", ""), paper.get("venue", ""))
        snapshot = snapshots.get(key)
        if snapshot is None:
            missing.append(paper.get("title", ""))
            continue
        merged = dict(paper)
        merged["paper_profile_snapshot"] = dict(snapshot)
        attached.append(merged)

    if missing:
        preview = ", ".join(title[:60] for title in missing[:3])
        raise ValueError(
            f"baseline eval 缺少固定 paper_profile_snapshot: {len(missing)} 篇"
            f"（示例: {preview}）"
        )
    return attached


def paper_profile_from_snapshot(snapshot: dict, paper: dict) -> PaperProfile:
    """Build an isolated PaperProfile from a saved evaluation snapshot."""
    list_fields = {
        "research_area",
        "ccf_research_area",
        "keywords",
        "techniques",
        "datasets",
        "evaluation_metrics",
        "application_domain",
    }
    values = {
        key: list(value) if key in list_fields and isinstance(value, list) else value
        for key, value in snapshot.items()
        if key in PaperProfile.model_fields
    }
    values["title"] = snapshot.get("title") or paper.get("title", "")
    values["abstract"] = paper.get("abstract", "") or ""
    return PaperProfile(**values)


def resolve_benchmark_input(benchmark_profile: str, input_path: str | None) -> str:
    if benchmark_profile == "custom":
        if not input_path:
            raise ValueError("--input is required when --benchmark-profile custom")
        return input_path
    if input_path:
        return input_path
    return BENCHMARK_PROFILE_INPUTS[benchmark_profile]


def init_pipeline() -> RecommenderPipeline:
    """初始化推荐 pipeline"""
    import yaml
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # 加载配置
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 初始化存储
    store = JournalStore()
    store.load()

    # 向量搜索器
    faiss_path = app_config["data"]["faiss_index_path"]
    metadata_path = app_config["data"]["metadata_path"]
    faiss_idx = FaissIndex(faiss_path, metadata_path)
    faiss_idx.load()
    if faiss_idx.is_loaded:
        vector_searcher = VectorSearcher(faiss_idx)
        store.set_vector_searcher(vector_searcher)

    bm25 = BM25Retriever(store)
    bm25.build_index()

    # LLM
    llm = build_minimax_llm(app_config)

    embedding_client = OllamaEmbedding(
        base_url=app_config["ollama"]["base_url"],
        model=app_config["ollama"]["embedding_model"],
        timeout=app_config.get("ollama", {}).get("timeout_seconds", 60),
    )

    embedding_retriever = None
    if store.has_vector_search():
        embedding_retriever = EmbeddingRetriever(store, embedding_client)

    retrieval_config = app_config.get("candidate_generator", {})
    merge_weights = retrieval_config.get("merge_weights", {"bm25": 0.45, "vector": 0.35, "text": 0.20})
    retrieval_target = retrieval_config.get("retrieval_target", "scope_text")
    fusion_strategy = retrieval_config.get("fusion_strategy", "weighted_minmax")
    hybrid_scope_weight = retrieval_config.get("hybrid_scope_weight", 0.75)
    hybrid_typical_weight = retrieval_config.get("hybrid_typical_weight", 0.25)
    identity_anchor_weight = retrieval_config.get("identity_anchor_weight", 0.03)
    accepted_paper_weight = retrieval_config.get("accepted_paper_weight", 0.20)
    rrf_k = retrieval_config.get("rrf_k", 60)
    route_top_k = retrieval_config.get("route_top_k")

    typical_bm25_retriever = None
    typical_text_retriever = None
    typical_embedding_retriever = None
    accepted_bm25_retriever = None
    accepted_embedding_retriever = None
    if retrieval_target in {"typical_abstracts", "semantic_anchors"}:
        typical_store = TypicalAbstractStore(abstracts_dir=app_config["data"]["typical_abstracts_dir"])
        typical_store.load()

        typical_bm25_retriever = TypicalAbstractBM25Retriever(typical_store, store)
        typical_bm25_retriever.build_index()
        typical_text_retriever = TypicalAbstractTextRetriever(typical_store, store)

        if store.has_vector_search():
            typical_embedding_retriever = TypicalAbstractEmbeddingRetriever(
                typical_store, store, embedding_client,
                faiss_path=app_config["data"]["typical_abstracts_faiss_path"],
                metadata_path=app_config["data"]["typical_abstracts_metadata_path"],
            )

        # accepted-paper 路由:本地 corpus + FAISS 索引齐全才启用
        from src.journals.accepted_paper_store import AcceptedPaperStore
        from src.retriever.accepted_paper_retriever import (
            AcceptedPaperBM25Retriever,
            AcceptedPaperEmbeddingRetriever,
        )

        accepted_store = AcceptedPaperStore(
            accepted_dir=app_config["data"].get("accepted_papers_dir", "data/accepted_papers")
        )
        accepted_store.load()
        if accepted_store.count > 0:
            accepted_bm25_retriever = AcceptedPaperBM25Retriever(accepted_store, store)
            accepted_bm25_retriever.build_index()
            if store.has_vector_search():
                accepted_embedding_retriever = AcceptedPaperEmbeddingRetriever(
                    accepted_store=accepted_store,
                    journal_store=store,
                    embedding_client=embedding_client,
                    faiss_path=app_config["data"].get(
                        "accepted_papers_faiss_path",
                        "data/processed/accepted_papers_index.faiss",
                    ),
                    metadata_path=app_config["data"].get(
                        "accepted_papers_metadata_path",
                        "data/processed/accepted_papers_metadata.parquet",
                    ),
                )
                if not accepted_embedding_retriever.is_available:
                    accepted_embedding_retriever = None

    generator = CandidateGenerator(
        store, bm25, embedding_retriever,
        merge_weights=merge_weights,
        retrieval_target=retrieval_target,
        typical_bm25_retriever=typical_bm25_retriever,
        typical_embedding_retriever=typical_embedding_retriever,
        typical_text_retriever=typical_text_retriever,
        accepted_bm25_retriever=accepted_bm25_retriever,
        accepted_embedding_retriever=accepted_embedding_retriever,
        hybrid_scope_weight=hybrid_scope_weight,
        hybrid_typical_weight=hybrid_typical_weight,
        identity_anchor_weight=identity_anchor_weight,
        accepted_paper_weight=accepted_paper_weight,
        fusion_strategy=fusion_strategy,
        rrf_k=rrf_k,
        route_top_k=route_top_k,
    )
    # 预建 RuleScorer 的 BM25 索引（传入期刊列表）
    rule_weights = app_config.get("ranking", {}).get("rule_scorer", {})
    scorer = RuleScorer(journals=store.journals, weights=rule_weights)

    # 6.3: 6.3 evidence role ranker 接入 production pipeline.
    # evidence_role 是默认 OFF 的新路径。打开时用 evidence snapshot 喂
    # LLMEvidenceRoleRanker; 没 snapshot 或 loading 失败则降级到 LLMRanker。
    ranking_cfg = app_config.get("ranking", {})
    use_direct_default = ranking_cfg.get("llm_direct", {}).get("enabled", True)
    evidence_cfg = ranking_cfg.get("evidence_role", {})
    use_evidence = bool(evidence_cfg.get("enabled", False))
    snapshot_path = evidence_cfg.get("snapshot_path", "")
    evidence_snapshot = None
    # Track the extractor so the API layer can use it for the per-request
    # online path (when a paper isn't in the snapshot). The closure capture
    # below needs the variable to be bound in the outer scope of init_pipeline.
    evidence_extractor = None
    if use_evidence:
        from pathlib import Path as _Path
        from src.ranker.llm_evidence_role_ranker import (
            LLMEvidenceRoleRanker,
            load_evidence_snapshot,
        )
        sp = _Path(snapshot_path)
        if not sp.exists():
            print(
                f"[warn] evidence_role.enabled=true but snapshot not found: "
                f"{snapshot_path}; falling back to llm_direct"
            )
        else:
            try:
                evidence_snapshot = load_evidence_snapshot(str(sp))
                # Build a real extractor as required by the constructor even
                # though we always read from the snapshot (extract() never called).
                from src.ranker.llm_evidence_extractor import LLMEvidenceExtractor
                evidence_extractor = LLMEvidenceExtractor(
                    llm=llm,
                    system_prompt=prompts["llm_evidence_extractor_system"],
                    user_prompt_template=prompts["llm_evidence_extractor_user"],
                    timeout_seconds=ranking_cfg.get("llm_ranker_timeout_seconds", 200),
                )
                accepted_store = None
                if "accepted_store" in locals():
                    accepted_store = locals()["accepted_store"]
                llm_ranker = LLMEvidenceRoleRanker(
                    evidence_extractor=evidence_extractor,
                    journal_store=store,
                    accepted_paper_store=accepted_store,
                    prior_source=evidence_cfg.get("prior_source", "rule"),
                    evidence_weight=float(evidence_cfg.get("evidence_weight", 0.8)),
                    prior_weight=float(evidence_cfg.get("prior_weight", 0.2)),
                    ltr_score_weight=float(evidence_cfg.get("ltr_score_weight", 0.0)),
                    evidence_snapshot=evidence_snapshot,
                )
                print(
                    f"[ok] evidence_role enabled, snapshot={snapshot_path}, "
                    f"prior_source={evidence_cfg.get('prior_source', 'rule')}, "
                    f"ltr_score_weight={evidence_cfg.get('ltr_score_weight', 0.0)}, "
                    f"papers={len(evidence_snapshot)}"
                )
            except Exception as exc:
                print(f"[warn] failed to load evidence snapshot: {exc}; falling back to llm_direct")
                evidence_snapshot = None

    if "llm_ranker" not in locals():
        if not use_direct_default:
            raise RuntimeError(
                "ranking config has neither llm_direct.enabled=true nor a working "
                "evidence_role snapshot; cannot build a ranker"
            )
        llm_ranker = LLMRanker(
            llm,
            prompts["llm_ranker_system"],
            prompts["llm_ranker_user"],
            timeout_seconds=app_config.get("ranking", {}).get("llm_ranker_timeout_seconds", 200),
        )
    quality_assessor = PaperQualityAssessor(llm)
    parser = PaperParser(llm)

    # 5.3: LTR adapter(默认 OFF 时 LTRAdapter.enabled=False,pipeline 走原路径)
    from src.ranker.ltr_adapter import LTRAdapter
    ltr_config = (app_config.get("ranking", {}).get("learned_reranker", {}) or {})
    # accepted_paper_store 引用:仅当 retrieval_target 包含 typical / accepted 时才有
    accepted_paper_store_ref = accepted_store if (retrieval_target in {"typical_abstracts", "semantic_anchors"} and "accepted_store" in locals()) else None
    learned_reranker = LTRAdapter(
        config=ltr_config,
        journal_store=store,
        accepted_paper_store=accepted_paper_store_ref,
    )

    pipeline = RecommenderPipeline(
        candidate_generator=generator,
        rule_scorer=scorer,
        llm_ranker=llm_ranker,
        quality_assessor=quality_assessor,
        llm_anchor_guard=app_config.get("ranking", {}).get("llm_anchor_guard", {}),
        learned_reranker=learned_reranker,
        evidence_extractor=evidence_extractor,
        # 6.4: enable 26-dim feature schema when evidence_role is on AND
        # the snapshot is loaded. attach_features() will fall back to 20
        # dims per-paper if a paper is missing from the snapshot.
        evidence_lookup=evidence_snapshot or None,
        feature_schema=(
            "26_dim_with_llm_evidence" if evidence_snapshot else "20_dim_base"
        ),
    )
    pipeline.parser = parser

    return pipeline


def evaluate_single_paper(
    paper: dict,
    pipeline: RecommenderPipeline,
    prompts: dict,
    mode: str,
    top_k: int,
    reuse_profile_snapshot: bool = False,
) -> dict:
    """
    并行评估单篇论文，返回结果字典（非 EvaluationResult，避免锁竞争）
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    venue = paper.get("venue", "")
    ccf_level = paper.get("ccf_level", "C")
    research_area = paper.get("research_area", [""])[0] if paper.get("research_area") else ""
    arxiv_id = paper.get("external_ids", {}).get("arXiv", "")
    pdf_path = paper.get("pdf_path", "")

    def _normalize_venue(venue: str) -> str:
        """标准化期刊名用于比较（去除首尾空格，转小写）"""
        return venue.strip().lower() if venue else ""

    def _find_journal_by_venue(venue_name: str):
        store = getattr(getattr(pipeline, "candidate_generator", None), "store", None)
        venue_normalized = _normalize_venue(venue_name)
        for journal in getattr(store, "journals", []):
            if _normalize_venue(journal.journal_name) == venue_normalized:
                return journal
        return None

    target_journal = _find_journal_by_venue(venue)

    # 获取全文（如果需要）
    full_text = ""
    if mode == "full" and pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            blocks, _ = extract_layout_blocks(pdf_content, pdf_path)
            paper_ast = build_paper_ast(blocks, title=title)
            full_text = paper_ast.to_markdown()
        except Exception as e:
            print(f"\nPDF读取失败: {arxiv_id} - {e}")

    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text=full_text,
        mode=mode,
    )

    # 正式消融可复用固定快照，避免 PaperParser / QualityAssessor 随机性污染排序对比。
    if reuse_profile_snapshot:
        snapshot = paper.get("paper_profile_snapshot")
        if not isinstance(snapshot, dict) or not snapshot:
            raise ValueError(f"论文缺少固定 paper_profile_snapshot: {title}")
        profile = paper_profile_from_snapshot(snapshot, paper)
    else:
        try:
            profile = pipeline.parser.parse(
                paper_input,
                prompts["paper_profile_system"],
                prompts["paper_profile_user"],
            )
        except Exception as e:
            print(f"\n解析失败: {title[:30]}... - {e}")
            return None

    # 推荐
    try:
        rec_result = pipeline.recommend(
            paper_input, profile,
            top_k=top_k,
            mode=mode,
            diagnostic_journal_ids=[target_journal.journal_id] if target_journal else None,
            quality_prompts=(
                None
                if reuse_profile_snapshot
                else {
                    "system": prompts.get("paper_quality_assessor_system", ""),
                    "user": prompts.get("paper_quality_assessor_user", ""),
                }
            ),
        )
    except Exception as e:
        print(f"\n推荐失败: {title[:30]}... - {e}")
        return None

    recommendations = rec_result.get("recommendations", [])
    rank_method = rec_result.get("rank_method", "unknown")
    fallback_used = bool(rec_result.get("fallback_used", False))
    if not recommendations:
        llm_pool_size = len(rec_result.get("llm_candidates", []))
        print(
            f"\n推荐结果为空: {title[:30]}... | "
            f"rank_method={rank_method} | llm_pool={llm_pool_size} | "
            "请检查候选召回或未处理异常"
        )
    candidates = rec_result.get("candidates", [])
    rule_ranked = rec_result.get("rule_ranked", [])
    llm_candidates = rec_result.get("llm_candidates", [])
    llm_candidate_ids = set(rec_result.get("llm_candidate_ids", []))
    retrieval_trace = rec_result.get("retrieval_trace", {})
    # 5.3: 透传 LTR 诊断字段(默认 OFF 时 rec_result 里没有这些 key,learned_enabled=False)
    learned_diag = rec_result.get("learned_diagnostics") or {}
    final_rank_source = rec_result.get("final_rank_source")
    learned_enabled = bool(learned_diag.get("status") == "ok")
    llm_role_diag = rec_result.get("llm_role_diagnostics") or {}
    llm_role_candidates = llm_role_diag.get("candidates") or {}
    recommended_journals = [rec.journal.journal_name for rec in recommendations]
    candidate_journal_names = [j.journal_name for j in candidates] if candidates else []
    rule_ranked_names = [j.journal_name for j, s, r in rule_ranked] if rule_ranked else []

    # 标准化 venue 用于比较
    venue_normalized = _normalize_venue(venue)
    candidate_journal_names_norm = [j.lower() for j in candidate_journal_names]
    rule_ranked_names_norm = [j.lower() for j in rule_ranked_names]
    recommended_journals_norm = [j.lower() for j in recommended_journals]
    candidate_by_name = {
        _normalize_venue(j.journal_name): (i + 1, j)
        for i, j in enumerate(candidates)
    }
    rule_by_name = {
        _normalize_venue(j.journal_name): (i + 1, j, s)
        for i, (j, s, r) in enumerate(rule_ranked)
    }
    llm_candidate_ids.update(j.journal_id for j, s, r in llm_candidates)

    # 计算 Hit@K（真正跑10个）
    hit_1 = venue_normalized in recommended_journals_norm[:1] if recommended_journals else False
    hit_3 = venue_normalized in recommended_journals_norm[:3] if recommended_journals else False
    hit_5 = venue_normalized in recommended_journals_norm[:5] if recommended_journals else False
    hit_10 = venue_normalized in recommended_journals_norm[:10] if venue else False

    # 粗排是否命中（实际发表的期刊在 top 50 候选中）
    coarse_hit = venue_normalized in candidate_journal_names_norm if venue else False

    # 粗排候选中是否包含实际发表的期刊（用于分析 RuleScorer 的选择）
    coarse_hit_in_rule_top10 = venue_normalized in rule_ranked_names_norm[:10] if venue else False
    coarse_hit_in_rule_top20 = venue_normalized in rule_ranked_names_norm[:20] if venue else False

    q_level = profile.quality_level or "D"

    # 计算 relevant_rank（1-indexed，未命中为 0）
    relevant_rank = 0
    for i, jname in enumerate(recommended_journals_norm):
        if jname == venue_normalized:
            relevant_rank = i + 1
            break

    # 同领域命中（research_area 匹配任一 subject_tags）
    research_areas_set = {research_area} if research_area else set()
    area_hit_5 = False
    area_hit_10 = False
    for i, rec in enumerate(recommendations[:10]):
        if rec.journal.subject_tags and research_areas_set:
            if set(rec.journal.subject_tags) & research_areas_set:
                if i < 5:
                    area_hit_5 = True
                area_hit_10 = True
                break

    # 同 CCF 档位命中（ccf_level 匹配期刊 ccf_rating）
    level_hit_5 = False
    level_hit_10 = False
    acceptable_journal_hit_5 = False
    for i, rec in enumerate(recommendations[:10]):
        if rec.journal.ccf_rating and ccf_level:
            if rec.journal.ccf_rating.upper() == ccf_level.upper():
                if i < 5:
                    level_hit_5 = True
                level_hit_10 = True
                break

    acceptable_journal_hit_5 = bool(hit_5 or (area_hit_5 and level_hit_5))

    def _retrieval_info(journal_id: str) -> dict:
        trace = retrieval_trace.get(journal_id, {})
        routes = trace.get("routes", {})
        wide_routes = trace.get("wide_routes", {})
        return {
            "retrieval_score": trace.get("total_score"),
            "retrieval_rank": trace.get("retrieval_rank"),
            "retrieval_sources": trace.get("primary_routes", []),
            "retrieval_route_scores": {
                route: {
                    "rank": data.get("rank"),
                    "raw_score": data.get("raw_score"),
                    "weighted_score": data.get("weighted_score"),
                    "normalized_score": data.get("normalized_score"),
                }
                for route, data in routes.items()
            },
            "wide_retrieval_rank": trace.get("wide_retrieval_rank"),
            "wide_retrieval_route_scores": {
                route: {
                    "rank": data.get("rank"),
                    "raw_score": data.get("raw_score"),
                    "weighted_score": data.get("weighted_score"),
                    "normalized_score": data.get("normalized_score"),
                }
                for route, data in wide_routes.items()
            },
        }

    def _find_venue_journal():
        candidate_match = candidate_by_name.get(venue_normalized)
        if candidate_match:
            _, journal = candidate_match
            return journal

        rule_match = rule_by_name.get(venue_normalized)
        if rule_match:
            _, journal, _ = rule_match
            return journal

        store = getattr(getattr(pipeline, "candidate_generator", None), "store", None)
        for journal in getattr(store, "journals", []):
            if _normalize_venue(journal.journal_name) == venue_normalized:
                return journal
        return None

    venue_journal = target_journal or _find_venue_journal()
    venue_retrieval = _retrieval_info(venue_journal.journal_id) if venue_journal else {
        "retrieval_score": None,
        "retrieval_rank": None,
        "retrieval_sources": [],
        "retrieval_route_scores": {},
        "wide_retrieval_rank": None,
        "wide_retrieval_route_scores": {},
    }
    venue_rule = rule_by_name.get(venue_normalized)
    wide_route_scores = venue_retrieval["wide_retrieval_route_scores"]

    def _miss_stage() -> str:
        if hit_5:
            return "final_hit"
        if not venue_journal:
            return "target_not_in_store"
        if not wide_route_scores and venue_retrieval.get("wide_retrieval_rank") is None:
            return "not_in_wide_recall"
        if venue_retrieval.get("retrieval_rank") is None:
            return "wide_recalled_but_not_top50"
        if not venue_rule or (venue_rule[0] or 9999) > 20:
            if bool(venue_journal.journal_id in llm_candidate_ids):
                return "in_llm_but_lost"
            return "rule_suppressed"
        if bool(venue_journal.journal_id in llm_candidate_ids):
            return "in_llm_but_lost"
        return "rule_suppressed"

    venue_diagnostic = {
        "journal_id": venue_journal.journal_id if venue_journal else None,
        "target_journal_id": venue_journal.journal_id if venue_journal else None,
        "gold_area": research_area,
        "parsed_ccf_area": profile.ccf_research_area,
        "area_mismatch": bool(research_area and research_area not in profile.ccf_research_area),
        "abstract_len": len(abstract or ""),
        "target_scope_text_preview": (venue_journal.scope_text[:300] if venue_journal else ""),
        "target_keywords": venue_journal.keywords[:12] if venue_journal else [],
        "target_subject_tags": venue_journal.subject_tags[:5] if venue_journal else [],
        "retrieval_sources": venue_retrieval["retrieval_sources"],
        "retrieval_rank": venue_retrieval["retrieval_rank"],
        "wide_retrieval_rank": venue_retrieval["wide_retrieval_rank"],
        "retrieval_score": venue_retrieval["retrieval_score"],
        "retrieval_route_scores": venue_retrieval["retrieval_route_scores"],
        "wide_retrieval_route_scores": wide_route_scores,
        "coarse_rank": candidate_by_name.get(venue_normalized, (None, None))[0],
        "rule_rank": venue_rule[0] if venue_rule else None,
        "rule_score": venue_rule[2] if venue_rule else None,
        "in_llm_pool": bool(venue_journal and venue_journal.journal_id in llm_candidate_ids),
        "miss_stage": _miss_stage(),
    }
    # 5.3: LTR 启用且 status='ok' 时把 learned_score / learned_rank 写到 gold venue 诊断
    if learned_enabled and venue_journal is not None:
        learned_score_map = learned_diag.get("learned_score") or {}
        learned_rank_map = learned_diag.get("learned_rank") or {}
        venue_diagnostic["learned_score"] = learned_score_map.get(venue_journal.journal_id)
        venue_diagnostic["learned_rank"] = learned_rank_map.get(venue_journal.journal_id)
    if venue_journal is not None and venue_journal.journal_id in llm_role_candidates:
        venue_role = llm_role_candidates[venue_journal.journal_id]
        venue_diagnostic["llm_role_rank"] = venue_role.get("final_rank")
        venue_diagnostic["llm_role_final_score"] = venue_role.get("final_score")
        if "evidence_composite" in venue_role:
            venue_diagnostic["llm_evidence_rank"] = venue_role.get("final_rank")
            venue_diagnostic["llm_evidence_final_score"] = venue_role.get(
                "final_score"
            )
            venue_diagnostic["llm_evidence_composite"] = venue_role.get(
                "evidence_composite"
            )
    paper_profile_snapshot = {
        "title": profile.title,
        "abstract_len": len(abstract or ""),
        "abstract_preview": (abstract or "")[:500],
        "research_area": profile.research_area,
        "ccf_research_area": profile.ccf_research_area,
        "method_type": profile.method_type,
        "paper_type": profile.paper_type,
        "keywords": profile.keywords,
        "novelty": profile.novelty,
        "difficulty_level": profile.difficulty_level,
        "style": profile.style,
        "sections_summary": profile.sections_summary,
        "full_text_summary": profile.full_text_summary,
        "techniques": profile.techniques,
        "datasets": profile.datasets,
        "evaluation_metrics": profile.evaluation_metrics,
        "application_domain": profile.application_domain,
        "novelty_type": profile.novelty_type,
        "quality_level": profile.quality_level,
        "quality_confidence": profile.quality_confidence,
        "quality_reasons": profile.quality_reasons,
        "paper_strength": profile.paper_strength,
        "readiness": profile.readiness,
    }

    return {
        "arxiv": arxiv_id,
        "title": title,
        "abstract_len": len(abstract or ""),
        "venue": venue,
        "ccf_level": ccf_level,
        "research_area": research_area,
        "gold_area": research_area,
        "parsed_ccf_area": profile.ccf_research_area,
        "area_mismatch": bool(research_area and research_area not in profile.ccf_research_area),
        "recommended_journals": recommended_journals[:top_k],
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_5": hit_5,
        "hit_10": hit_10,
        "relevant_rank": relevant_rank,  # MRR 计算用
        "ndcg_gain": 1.0 if relevant_rank > 0 else 0.0,  # NDCG 计算用（二值相关性）
        "coarse_hit": coarse_hit,  # 粗排命中
        "coarse_hit_in_rule_top10": coarse_hit_in_rule_top10,  # 粗排候选在 RuleScorer top10 中
        "coarse_hit_in_rule_top20": coarse_hit_in_rule_top20,  # 粗排候选在 RuleScorer top20 中
        "paper_strength": profile.paper_strength,
        "quality_level": q_level,
        "ccf_research_area": profile.ccf_research_area,
        "paper_profile_snapshot": paper_profile_snapshot,
        "evaluation_status": (
            "fallback" if fallback_used else ("ok" if recommendations else "empty")
        ),
        "rank_method": rank_method,
        "fallback_used": fallback_used,
        "fallback_stage": rec_result.get("fallback_stage", ""),
        "fallback_reason": rec_result.get("fallback_reason", ""),
        "area_match": bool(
            profile.ccf_research_area and research_area in profile.ccf_research_area
        ) if research_area else False,
        "level_match": q_level == ccf_level,
        "area_subject_tag_match": any(
            research_area and any(
                tag in rec.journal.subject_tags
                for tag in (research_area if isinstance(research_area, list) else [research_area])
            )
            for rec in recommendations if hasattr(rec, 'journal')
        ) if research_area and recommendations else False,
        # 新增指标
        "area_hit_5": area_hit_5,
        "area_hit_10": area_hit_10,
        "level_hit_5": level_hit_5,
        "level_hit_10": level_hit_10,
        "same_area_hit_5": area_hit_5,
        "same_area_hit_10": area_hit_10,
        "same_ccf_level_hit_5": level_hit_5,
        "same_ccf_level_hit_10": level_hit_10,
        "acceptable_journal_hit_5": acceptable_journal_hit_5,
        "venue_diagnostic": venue_diagnostic,
        # 每篇论文的详细推荐信息（用于定位问题）
        "recommendations_detail": [
            {
                "rank": i + 1,
                "journal_name": rec.journal.journal_name,
                "journal_id": rec.journal.journal_id,
                "ccf_rating": rec.journal.ccf_rating or "未知",
                "subject_tags": rec.journal.subject_tags[:5],
                "rule_rank": rule_ranked_names.index(rec.journal.journal_name) + 1 if rec.journal.journal_name in rule_ranked_names else -1,
                "rule_score": rule_ranked[rule_ranked_names.index(rec.journal.journal_name)][1] if rec.journal.journal_name in rule_ranked_names else None,
                "llm_score": rec.score,
                "confidence": rec.confidence,
                "match_reasons": rec.match_reasons[:5] if rec.match_reasons else [],
                **_retrieval_info(rec.journal.journal_id),
                # 5.3: LTR 启用时给每条推荐加 learned_score
                **(
                    {"learned_score": (learned_diag.get("learned_score") or {}).get(rec.journal.journal_id)}
                    if learned_enabled else {}
                ),
            }
            for i, rec in enumerate(recommendations[:top_k])
        ],
        "llm_candidates_detail": [
            {
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "candidate_input_rank": index + 1,
                "candidate_rule_score": score,
                "candidate_reasons": reasons,
                **_retrieval_info(journal.journal_id),
                **llm_role_candidates.get(journal.journal_id, {}),
            }
            for index, (journal, score, reasons) in enumerate(llm_candidates)
        ],
        **(
            {
                "llm_role_status": llm_role_diag.get("status"),
                "llm_role": llm_role_diag.get("role"),
                "llm_role_prior_source": llm_role_diag.get("prior_source"),
                "llm_role_fallback_reason": llm_role_diag.get(
                    "fallback_reason", ""
                ),
                **(
                    {
                        "llm_evidence_status": llm_role_diag.get("status"),
                        "llm_evidence_coverage": llm_role_diag.get(
                            "evidence_coverage", 0.0
                        ),
                        "llm_evidence_prior_source": llm_role_diag.get(
                            "prior_source"
                        ),
                        "llm_evidence_fallback_reason": llm_role_diag.get(
                            "fallback_reason", ""
                        ),
                    }
                    if llm_role_diag.get("role") == "evidence"
                    else {}
                ),
            }
            if llm_role_diag
            else {}
        ),
        # 5.3: LTR 启用时 per-paper 顶层加 final_rank_source
        **(
            {"final_rank_source": final_rank_source}
            if learned_enabled or llm_role_diag
            else {}
        ),
    }


def run_evaluation(
    papers: list,
    pipeline: RecommenderPipeline,
    mode: str,
    top_k: int,
    prompts: dict,
    show_progress: bool = True,
    workers: int = 4,
    reuse_profile_snapshots: bool = False,
) -> EvaluationResult:
    """运行评估（并行）"""

    result = EvaluationResult(
        total_count=len(papers),
        mode=mode,
        top_k=top_k,
        hit_at_1=0, hit_at_3=0, hit_at_5=0, hit_at_10=0,
        area_match_count=0,
        area_subject_tag_match_count=0,
        level_match_count=0,
        ndcg_at_10=0.0,
        area_hit_at_5=0, area_hit_at_10=0,
        level_hit_at_5=0, level_hit_at_10=0,
        acceptable_journal_hit_at_5=0,
        coarse_hit_count=0,
        coarse_hit_in_rule_top10_count=0,
        coarse_hit_in_rule_top20_count=0,
        fallback_count=0,
        llm_success_count=0,
        empty_recommendation_count=0,
        level_a_count=0, level_a_hit_at_5=0,
        level_b_count=0, level_b_hit_at_5=0,
        level_c_count=0, level_c_hit_at_5=0,
        level_d_count=0, level_d_hit_at_5=0,
        by_area=defaultdict(lambda: {"total": 0, "hit": 0, "area_match": 0}),
        by_level=defaultdict(lambda: {"total": 0, "hit": 0}),
        paper_results=[],
    )

    results_lock = threading.Lock()

    def update_result(paper_result: dict):
        if paper_result is None:
            return
        with results_lock:
            # 累加 Hit@K
            if paper_result["hit_1"]: result.hit_at_1 += 1
            if paper_result["hit_3"]: result.hit_at_3 += 1
            if paper_result["hit_5"]: result.hit_at_5 += 1
            if paper_result["hit_10"]: result.hit_at_10 += 1

            # 领域匹配（LLM预测的领域 vs 论文实际领域）
            if paper_result["area_match"]:
                result.area_match_count += 1
                area = paper_result["research_area"]
                result.by_area[area]["area_match"] += 1

            # 推荐期刊的 subject_tags 命中论文 research_area
            if paper_result.get("area_subject_tag_match"):
                result.area_subject_tag_match_count += 1

            # Level Match
            if paper_result["level_match"]:
                result.level_match_count += 1

            # MRR 和 NDCG@5 累加
            rank = paper_result.get("relevant_rank", 0)
            if rank > 0:
                import math
                result.mrr += 1.0 / rank
                result.ndcg_at_5 += 1.0 / math.log2(rank + 1)  # 二值相关性: DCG=1/log2(r+1), IDCG=1/log2(2)=1

            # NDCG@10（真正跑满10个，只有10个结果时才计算）
            recommended_journals = paper_result.get("recommended_journals", [])
            if len(recommended_journals) >= 10 and rank > 0:
                result.ndcg_at_10 += 1.0 / math.log2(rank + 1)

            # 同领域/同CCF档位命中
            if paper_result.get("area_hit_5"):
                result.area_hit_at_5 += 1
            if paper_result.get("area_hit_10"):
                result.area_hit_at_10 += 1
            if paper_result.get("level_hit_5"):
                result.level_hit_at_5 += 1
            if paper_result.get("level_hit_10"):
                result.level_hit_at_10 += 1
            if paper_result.get("acceptable_journal_hit_5"):
                result.acceptable_journal_hit_at_5 += 1

            # 粗排命中统计
            if paper_result.get("coarse_hit"):
                result.coarse_hit_count += 1
            if paper_result.get("coarse_hit_in_rule_top10"):
                result.coarse_hit_in_rule_top10_count += 1
            if paper_result.get("coarse_hit_in_rule_top20"):
                result.coarse_hit_in_rule_top20_count += 1

            if paper_result.get("fallback_used"):
                result.fallback_count += 1
            elif (
                str(paper_result.get("rank_method", "")).startswith("llm")
                and paper_result.get("recommended_journals")
                and paper_result.get("llm_role_status") != "neutral_fallback"
            ):
                result.llm_success_count += 1
            if not paper_result.get("recommended_journals"):
                result.empty_recommendation_count += 1

            # 分质量等级统计
            q_level = paper_result["quality_level"]
            if q_level == "A":
                result.level_a_count += 1
                if paper_result["hit_5"]: result.level_a_hit_at_5 += 1
            elif q_level == "B":
                result.level_b_count += 1
                if paper_result["hit_5"]: result.level_b_hit_at_5 += 1
            elif q_level == "C":
                result.level_c_count += 1
                if paper_result["hit_5"]: result.level_c_hit_at_5 += 1
            else:
                result.level_d_count += 1
                if paper_result["hit_5"]: result.level_d_hit_at_5 += 1

            # 按领域统计
            area = paper_result["research_area"]
            result.by_area[area]["total"] += 1
            if paper_result["hit_5"]: result.by_area[area]["hit"] += 1

            # 按CCF等级统计
            ccf = paper_result["ccf_level"]
            result.by_level[ccf]["total"] += 1
            if paper_result["hit_5"]: result.by_level[ccf]["hit"] += 1

            # 保存单篇结果
            result.paper_results.append({
                k: v for k, v in paper_result.items()
                if k not in ["hit_1", "hit_3", "hit_5", "hit_10", "area_match", "level_match", "relevant_rank", "ndcg_gain"]
            })

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_paper,
                paper, pipeline, prompts, mode, top_k, reuse_profile_snapshots,
            ): paper
            for paper in papers
        }

        pbar = tqdm(
            as_completed(futures),
            total=len(papers),
            desc=f"评估 [{mode}/top{top_k}/w{workers}]",
            unit="篇",
            disable=not show_progress,
        )

        for future in pbar:
            paper_result = future.result()
            update_result(paper_result)

            # 更新进度条
            n = len(result.paper_results)
            if n > 0:
                hit_k = getattr(result, f"hit_at_{top_k}", 0)
                mrr_val = result.mrr / n
                ndcg5_val = result.ndcg_at_5 / n
                pbar.set_postfix({
                    f"Hit@{top_k}": f"{hit_k}/{n}({hit_k*100/n:.1f}%)",
                    "MRR": f"{mrr_val:.3f}",
                    "NDCG@5": f"{ndcg5_val:.3f}",
                    "Area@5": f"{result.area_hit_at_5}/{n}",
                    "level_match": f"{result.level_match_count}/{n}",
                })

    return result


def print_report(result: EvaluationResult):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print(f"评估报告 - 模式: {result.mode} | Top-K: {result.top_k}")
    print("=" * 70)

    metrics = calculate_metrics(result)

    print(f"\n总论文数: {result.total_count}")

    print(f"\n--- Hit@K ---")
    print(f"  Hit@1:  {metrics['Hit@1']}")
    print(f"  Hit@3:  {metrics['Hit@3']}")
    print(f"  Hit@5:  {metrics['Hit@5']}")
    print(f"  Hit@10: {metrics['Hit@10']}")

    print(f"\n--- 其他指标 ---")
    total = result.total_count if result.total_count > 0 else 1
    print(f"  领域匹配准确率: {metrics['Area Match Rate']}")
    print(f"  推荐期刊subject_tags命中: {result.area_subject_tag_match_count}/{total} ({result.area_subject_tag_match_count*100/total:.1f}%)")
    print(f"  Level Match Rate: {metrics['Level Match Rate']}")
    print(f"  MRR: {metrics['MRR']}")
    print(f"  NDCG@5: {metrics['NDCG@5']}")
    print(f"  NDCG@10: {metrics['NDCG@10']}")
    print(f"  同领域命中@5: {metrics['同领域命中@5']}")
    print(f"  同领域命中@10: {metrics['同领域命中@10']}")
    print(f"  同CCF档位@5: {metrics['同CCF档位@5']}")
    print(f"  同CCF档位@10: {metrics['同CCF档位@10']}")
    print(f"  可接受期刊命中@5: {metrics['可接受期刊命中@5']}")

    # 粗排命中分析
    print(f"\n--- 粗排命中分析 ---")
    print(f"  粗排命中（top50候选包含实际期刊）: {result.coarse_hit_count}/{total} ({result.coarse_hit_count*100/total:.1f}%)")
    print(f"  粗排命中且在RuleScorer top10中: {result.coarse_hit_in_rule_top10_count}/{result.coarse_hit_count} ({result.coarse_hit_in_rule_top10_count*100/max(result.coarse_hit_count,1):.1f}%)")
    print(f"  粗排命中且在RuleScorer top20中: {result.coarse_hit_in_rule_top20_count}/{result.coarse_hit_count} ({result.coarse_hit_in_rule_top20_count*100/max(result.coarse_hit_count,1):.1f}%)")

    print(f"\n--- 稳定性诊断 ---")
    print(f"  LLM 正常完成: {result.llm_success_count}/{total}")
    print(f"  Rule fallback: {result.fallback_count}/{total}")
    print(f"  空推荐结果: {result.empty_recommendation_count}/{total}")

    print(f"\n--- 分质量等级 Hit@5 ---")
    if result.level_a_count > 0:
        print(f"  A级 (n={result.level_a_count}): {metrics.get('A级 Hit@5', 'N/A')}")
    if result.level_b_count > 0:
        print(f"  B级 (n={result.level_b_count}): {metrics.get('B级 Hit@5', 'N/A')}")
    if result.level_c_count > 0:
        print(f"  C级 (n={result.level_c_count}): {metrics.get('C级 Hit@5', 'N/A')}")
    if result.level_d_count > 0:
        print(f"  D级 (n={result.level_d_count}): {metrics.get('D级 Hit@5', 'N/A')}")

    print(f"\n--- 按领域分布 ---")
    for area, stats in sorted(result.by_area.items()):
        hit_rate = stats["hit"] * 100 / stats["total"] if stats["total"] > 0 else 0
        print(f"  {area}: {stats['hit']}/{stats['total']} ({hit_rate:.1f}%)")

    print(f"\n--- 按CCF等级分布 ---")
    for level in ['A', 'B', 'C']:
        if level in result.by_level:
            stats = result.by_level[level]
            hit_rate = stats["hit"] * 100 / stats["total"] if stats["total"] > 0 else 0
            print(f"  {level}级: {stats['hit']}/{stats['total']} ({hit_rate:.1f}%)")

    print("=" * 70)


def save_results(
    result: EvaluationResult,
    output_dir: str = "data/evaluation/results",
    benchmark_manifest: dict | None = None,
    benchmark_profile: str = "custom",
    benchmark_path: str = "",
    app_config: dict | None = None,
    ltr_info: dict | None = None,
):
    """保存评估结果。

    顶层除原有的 timestamp / mode / top_k / total_count / metrics / paper_results 外,
    5.4.e 新增:
    - environment: 评测环境元信息 (benchmark profile, path, ltr 状态, model, hash 等)
    - metrics 内每个 count 字段旁加 _rate (0~1 小数) 字段
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{result.mode}_top{result.top_k}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    total = result.total_count if result.total_count > 0 else 1

    def _rate(num: int) -> float:
        return round(num / total, 4) if total > 0 else 0.0

    # 转换为可序列化的dict
    result_dict = {
        "timestamp": timestamp,
        "mode": result.mode,
        "top_k": result.top_k,
        "total_count": result.total_count,
        "environment": {
            "benchmark_profile": benchmark_profile,
            "benchmark_path": benchmark_path,
            "paper_count": result.total_count,
            "ltr_enabled": (ltr_info or {}).get("enabled", False),
            "ltr_model_path": (ltr_info or {}).get("model_path"),
            "ltr_model_converged": (ltr_info or {}).get("model_converged"),
            "ltr_disable_reason": (ltr_info or {}).get("disable_reason"),
            "minimax_model": (app_config or {}).get("minimax", {}).get("model"),
            "embedding_model": (app_config or {}).get("ollama", {}).get("embedding_model"),
            "retrieval_target": (app_config or {}).get("candidate_generator", {}).get("retrieval_target"),
            "candidate_generator": {
                "hybrid_scope_weight": (app_config or {}).get("candidate_generator", {}).get("hybrid_scope_weight"),
                "hybrid_typical_weight": (app_config or {}).get("candidate_generator", {}).get("hybrid_typical_weight"),
                "accepted_paper_weight": (app_config or {}).get("candidate_generator", {}).get("accepted_paper_weight"),
                "identity_anchor_weight": (app_config or {}).get("candidate_generator", {}).get("identity_anchor_weight"),
            },
            "ranking": {
                "llm_anchor_guard_enabled": (app_config or {}).get("ranking", {}).get("llm_anchor_guard", {}).get("enabled"),
                "rule_scorer": (app_config or {}).get("ranking", {}).get("rule_scorer", {}),
            },
            "benchmark_manifest_hash": (benchmark_manifest or {}).get("app_config_hash"),
            "prompt_hash": (benchmark_manifest or {}).get("prompt_hash"),
        },
        "metrics": {
            "hit_at_1": result.hit_at_1,
            "hit_at_1_rate": _rate(result.hit_at_1),
            "hit_at_3": result.hit_at_3,
            "hit_at_3_rate": _rate(result.hit_at_3),
            "hit_at_5": result.hit_at_5,
            "hit_at_5_rate": _rate(result.hit_at_5),
            "hit_at_10": result.hit_at_10,
            "hit_at_10_rate": _rate(result.hit_at_10),
            "area_match_count": result.area_match_count,
            "area_match_rate": _rate(result.area_match_count),
            "level_match_count": result.level_match_count,
            "level_match_rate": _rate(result.level_match_count),
            "mrr": result.mrr / total if total > 0 else 0.0,
            "ndcg_at_5": result.ndcg_at_5 / total if total > 0 else 0.0,
            "coarse_hit_count": result.coarse_hit_count,
            "coarse_hit_rate": _rate(result.coarse_hit_count),
            "coarse_hit_in_rule_top10_count": result.coarse_hit_in_rule_top10_count,
            "coarse_hit_in_rule_top20_count": result.coarse_hit_in_rule_top20_count,
            "fallback_count": result.fallback_count,
            "fallback_rate": _rate(result.fallback_count),
            "llm_success_count": result.llm_success_count,
            "llm_success_rate": _rate(result.llm_success_count),
            "empty_recommendation_count": result.empty_recommendation_count,
            "empty_recommendation_rate": _rate(result.empty_recommendation_count),
            "area_subject_tag_match_count": result.area_subject_tag_match_count,
            "area_subject_tag_match_rate": _rate(result.area_subject_tag_match_count),
            "level_a_count": result.level_a_count,
            "level_a_hit_at_5": result.level_a_hit_at_5,
            "level_a_hit_at_5_rate": _rate(result.level_a_hit_at_5) if result.level_a_count > 0 else 0.0,
            "level_b_count": result.level_b_count,
            "level_b_hit_at_5": result.level_b_hit_at_5,
            "level_b_hit_at_5_rate": _rate(result.level_b_hit_at_5) if result.level_b_count > 0 else 0.0,
            "level_c_count": result.level_c_count,
            "level_c_hit_at_5": result.level_c_hit_at_5,
            "level_c_hit_at_5_rate": _rate(result.level_c_hit_at_5) if result.level_c_count > 0 else 0.0,
            "level_d_count": result.level_d_count,
            "level_d_hit_at_5": result.level_d_hit_at_5,
            "level_d_hit_at_5_rate": _rate(result.level_d_hit_at_5) if result.level_d_count > 0 else 0.0,
            "area_hit_at_5": result.area_hit_at_5,
            "area_hit_at_5_rate": _rate(result.area_hit_at_5),
            "area_hit_at_10": result.area_hit_at_10,
            "area_hit_at_10_rate": _rate(result.area_hit_at_10),
            "level_hit_at_5": result.level_hit_at_5,
            "level_hit_at_5_rate": _rate(result.level_hit_at_5),
            "level_hit_at_10": result.level_hit_at_10,
            "level_hit_at_10_rate": _rate(result.level_hit_at_10),
            "acceptable_journal_hit_at_5": result.acceptable_journal_hit_at_5,
            "acceptable_journal_hit_at_5_rate": _rate(result.acceptable_journal_hit_at_5),
        },
        "by_area": dict(result.by_area),
        "by_level": dict(result.by_level),
        "paper_results": result.paper_results,
    }
    if benchmark_manifest is not None:
        result_dict["benchmark_manifest"] = benchmark_manifest

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {filepath}")
    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description="期刊推荐系统评估")
    parser.add_argument("--benchmark-profile", choices=["light30", "full-v2", "full-v2-90", "holdout240", "custom"], default="custom",
                        help="Benchmark profile; light30/full-v2 use default inputs unless --input overrides")
    parser.add_argument("--input", "-i", default=None,
                        help="论文元数据路径")
    parser.add_argument("--mode", "-m", choices=["title", "abstract", "full"], default="abstract",
                        help="推荐模式")
    parser.add_argument("--top-k", "-k", type=int, nargs="+", default=[5],
                        help="Top-K 值，可指定多个如: --top-k 3 5 10")
    parser.add_argument("--papers", "-n", type=int, default=None,
                        help="限制评估论文数量（用于测试）")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存结果")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="并行线程数（默认10）")
    parser.add_argument(
        "--baseline-eval",
        default="",
        help="复用已完成评测中的固定 paper_profile_snapshot，保证排序消融口径一致",
    )

    args = parser.parse_args()
    args.input = resolve_benchmark_input(args.benchmark_profile, args.input)

    # 加载论文
    print(f"加载论文数据: {args.input}")
    papers = load_papers_metadata(args.input)
    if args.papers:
        papers = papers[:args.papers]
    if args.baseline_eval:
        papers = attach_baseline_profile_snapshots(papers, args.baseline_eval)
        print(f"复用固定 PaperProfile 快照: {args.baseline_eval}")
    print(f"共 {len(papers)} 篇论文")

    # 初始化 pipeline
    print("\n初始化推荐系统...")
    pipeline = init_pipeline()

    # 加载 prompts + app_config (5.4.e: 报告里要写 environment 块)
    import yaml
    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    # LTR 状态 (从 pipeline 拿)
    ltr_info = {"enabled": False, "model_path": None, "model_converged": None, "disable_reason": None}
    ltr_cfg = app_config.get("ranking", {}).get("learned_reranker", {}) or {}
    ltr_info["model_path"] = ltr_cfg.get("model_path")
    ltr = getattr(pipeline, "learned_reranker", None)
    if ltr is not None:
        ltr_info["enabled"] = bool(getattr(ltr, "enabled", False))
        ltr_info["disable_reason"] = getattr(ltr, "disable_reason", None)
        ltr_info["model_converged"] = getattr(ltr, "model_converged", None)

    # 运行评估
    for top_k in args.top_k:
        print(f"\n{'='*70}")
        print(f"开始评估 | 模式: {args.mode} | Top-{top_k}")
        print(f"{'='*70}")

        result = run_evaluation(
            papers,
            pipeline,
            args.mode,
            top_k,
            prompts,
            show_progress=True,
            workers=args.workers,
            reuse_profile_snapshots=bool(args.baseline_eval),
        )

        # 打印报告
        print_report(result)

        # 保存结果 (5.4.e 加 environment + *_rate 字段)
        if not args.no_save:
            manifest = build_benchmark_manifest(
                input_path=args.input,
                mode=args.mode,
                top_k=top_k,
                clean_benchmark=False,
                profile_snapshot_reused=bool(args.baseline_eval),
                baseline_eval_path=args.baseline_eval or None,
                workers=args.workers,
            )
            save_results(
                result,
                benchmark_manifest=manifest,
                benchmark_profile=args.benchmark_profile,
                benchmark_path=args.input,
                app_config=app_config,
                ltr_info=ltr_info,
            )

    print("\n评估完成!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Search RuleScorer settings on a fixed comparable evaluation set."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from scripts.run_retrieval_ablation import (
    _accumulate_metrics,
    _accumulate_route_attribution,
    _finalize_metrics,
    _miss_stage,
    _new_metric_accumulator,
    _progress_snapshot,
    _rank_of,
    _target_journal_id,
    _target_route_attribution,
    build_candidate_generator,
    journal_name_to_id_map,
    load_comparable_eval_papers,
    paper_profile_from_metadata,
)
from src.ranker.rule_scorer import RuleScorer


DEFAULT_RULE_TRIALS = (
    {
        "label": "baseline",
        "weights": {},
    },
    {
        "label": "rank0.6|scope0.4|confirm0.2|multi0.1|area0.3",
        "weights": {
            "retrieval_rank_prior": 0.6,
            "strong_scope_rank_bonus": 0.4,
            "scope_typical_confirm_bonus": 0.2,
            "multi_route_bonus": 0.1,
            "research_area_match": 0.3,
            "ccf_research_area_match": 0.3,
        },
    },
    {
        "label": "rank1.0|scope0.8|confirm0.4|multi0.2|area0.4",
        "weights": {
            "retrieval_rank_prior": 1.0,
            "strong_scope_rank_bonus": 0.8,
            "scope_typical_confirm_bonus": 0.4,
            "multi_route_bonus": 0.2,
            "research_area_match": 0.4,
            "ccf_research_area_match": 0.4,
        },
    },
    {
        "label": "rank1.2|scope0.8|confirm0.5|multi0.3|area0.5|typ0.08",
        "weights": {
            "retrieval_rank_prior": 1.2,
            "strong_scope_rank_bonus": 0.8,
            "scope_typical_confirm_bonus": 0.5,
            "multi_route_bonus": 0.3,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.08,
        },
    },
    {
        "label": "rank1.5|scope1.0|confirm0.6|multi0.4|area0.5|typ0.08",
        "weights": {
            "retrieval_rank_prior": 1.5,
            "strong_scope_rank_bonus": 1.0,
            "scope_typical_confirm_bonus": 0.6,
            "multi_route_bonus": 0.4,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.08,
        },
    },
    {
        "label": "rank1.8|scope1.0|confirm0.6|multi0.4|area0.5|typ0.08",
        "weights": {
            "retrieval_rank_prior": 1.8,
            "strong_scope_rank_bonus": 1.0,
            "scope_typical_confirm_bonus": 0.6,
            "multi_route_bonus": 0.4,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.08,
        },
    },
    {
        "label": "rank1.8|scope1.0|typRank0.6|confirm0.5|multi0.3|area0.5|typ0.06",
        "weights": {
            "retrieval_rank_prior": 1.8,
            "strong_scope_rank_bonus": 1.0,
            "strong_typical_rank_bonus": 0.6,
            "scope_typical_confirm_bonus": 0.5,
            "multi_route_bonus": 0.3,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.06,
        },
    },
    {
        "label": "rank1.5|scope0.8|typRank0.8|confirm0.4|multi0.2|area0.5|typ0.06",
        "weights": {
            "retrieval_rank_prior": 1.5,
            "strong_scope_rank_bonus": 0.8,
            "strong_typical_rank_bonus": 0.8,
            "scope_typical_confirm_bonus": 0.4,
            "multi_route_bonus": 0.2,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.06,
        },
    },
    {
        "label": "rank1.2|scope0.8|typRank1.0|confirm0.4|multi0.2|area0.5|typ0.08",
        "weights": {
            "retrieval_rank_prior": 1.2,
            "strong_scope_rank_bonus": 0.8,
            "strong_typical_rank_bonus": 1.0,
            "scope_typical_confirm_bonus": 0.4,
            "multi_route_bonus": 0.2,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.08,
        },
    },
    {
        "label": "rank1.2|scope0.8|confirm0.8|multi0.5|area0.5|typ0.08",
        "weights": {
            "retrieval_rank_prior": 1.2,
            "strong_scope_rank_bonus": 0.8,
            "scope_typical_confirm_bonus": 0.8,
            "multi_route_bonus": 0.5,
            "research_area_match": 0.5,
            "ccf_research_area_match": 0.5,
            "typical_only_penalty": 0.08,
        },
    },
    {
        "label": "rank1.5|scope0.8|confirm0.8|multi0.5|area0.3|typ0.10",
        "weights": {
            "retrieval_rank_prior": 1.5,
            "strong_scope_rank_bonus": 0.8,
            "scope_typical_confirm_bonus": 0.8,
            "multi_route_bonus": 0.5,
            "research_area_match": 0.3,
            "ccf_research_area_match": 0.3,
            "typical_only_penalty": 0.10,
        },
    },
)


def build_rule_trials(max_trials: int | None = None) -> list[dict]:
    trials = list(DEFAULT_RULE_TRIALS)
    if max_trials is not None:
        return trials[:max_trials]
    return trials


def format_rule_trial_status(index: int, total: int, summary: dict) -> str:
    return (
        f"trial {index}/{total} | "
        f"coarse@50={summary['coarse_hit_at_50']} | "
        f"rule@10={summary['rule_hit_at_10']} | "
        f"rule@20={summary['rule_hit_at_20']} | "
        f"rule_mrr={summary['rule_mrr']:.4f} | "
        f"{summary['label']}"
    )


def _write_search_results(output_path: str, results: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def evaluate_rule_trial(
    papers: list[dict],
    generator,
    scorer: RuleScorer,
    journal_name_to_id: dict[str, str],
    mode: str = "abstract",
    candidate_top_k: int = 50,
    show_progress: bool = False,
    live_label: str = "",
    log_every: int = 10,
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
        desc="rule scorer eval",
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
            candidates, retrieval_trace = generator.generate_with_trace(
                query_text,
                profile,
                top_k=candidate_top_k,
                mode=mode,
                diagnostic_journal_ids=[target_id],
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

            paper_results.append({
                "title": paper.get("title", ""),
                "profile_title": profile.title,
                "venue": paper.get("venue", ""),
                "target_journal_id": target_id,
                "retrieval_rank": retrieval_rank,
                "rule_rank": rule_rank,
                "baseline_final_hit_5": final_hit_5,
                "miss_stage": miss_stage,
                "target_route_attribution": target_attribution,
                "retrieval_top5": candidate_ids[:5],
                "rule_top5": rule_ids[:5],
            })
            progress_bar.update(1)
            snapshot = _progress_snapshot(
                evaluated_count=evaluated_count,
                missing_target_count=missing_target_count,
                retrieval=retrieval,
                rule=rule,
                baseline_final_hit_at_5=baseline_final_hit_at_5,
                miss_stage_counts=miss_stage_counts,
            )
            progress_bar.set_postfix(snapshot)
            if (not show_progress) and log_every and evaluated_count % log_every == 0:
                prefix = f"[{live_label}] " if live_label else ""
                print(
                    prefix
                    + "papers="
                    + snapshot["eval"]
                    + f" coarse@50={snapshot['coarse@50']}"
                    + f" rule@20={snapshot['rule@20']}"
                    + f" rule_mrr={snapshot['rule_mrr']}"
                )
    finally:
        progress_bar.close()

    finalized_retrieval = _finalize_metrics(retrieval, evaluated_count)
    finalized_rule = _finalize_metrics(rule, evaluated_count)
    return {
        "evaluated_count": evaluated_count,
        "missing_target_count": missing_target_count,
        "candidate_top_k": candidate_top_k,
        "coarse_hit_at_50": int(finalized_retrieval.get("Hit@50", 0)),
        "rule_hit_at_10": int(finalized_rule.get("Hit@10", 0)),
        "rule_hit_at_20": int(finalized_rule.get("Hit@20", 0)),
        "baseline_final_hit_at_5": baseline_final_hit_at_5,
        "miss_stage_counts": miss_stage_counts,
        "route_attribution": route_attribution,
        "retrieval": finalized_retrieval,
        "rule": finalized_rule,
        "paper_results": paper_results,
    }


def summarize_rule_trial(trial: dict, result: dict) -> dict:
    rule = result["rule"]
    retrieval = result["retrieval"]
    return {
        "label": trial["label"],
        "weights": trial["weights"],
        "coarse_hit_at_50": result["coarse_hit_at_50"],
        "rule_hit_at_10": result["rule_hit_at_10"],
        "rule_hit_at_20": result["rule_hit_at_20"],
        "rule_mrr": rule["MRR"],
        "rule_ndcg_at_5": rule["NDCG@5"],
        "retrieval_mrr": retrieval["MRR"],
        "miss_stage_counts": result["miss_stage_counts"],
        "route_attribution": result["route_attribution"],
        "paper_results": result["paper_results"],
    }


def sort_rule_trials(trials: list[dict]) -> list[dict]:
    return sorted(
        trials,
        key=lambda item: (
            item["rule_hit_at_20"],
            item.get("rule_mrr", 0.0),
            item.get("rule_hit_at_10", 0),
            item.get("rule_ndcg_at_5", 0.0),
            item.get("coarse_hit_at_50", 0),
        ),
        reverse=True,
    )


def run_search(
    papers_path: str,
    baseline_eval_path: str,
    config_path: str,
    output_path: str,
    mode: str = "abstract",
    candidate_top_k: int = 50,
    limit: int | None = None,
    include_vector: bool = True,
    max_trials: int | None = None,
    show_progress: bool = True,
    show_paper_progress: bool = False,
    log_every: int = 10,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    generator = build_candidate_generator(app_config, include_vector=include_vector)
    papers = load_comparable_eval_papers(papers_path, baseline_eval_path, limit=limit)
    name_to_id = journal_name_to_id_map(generator.store)
    base_weights = app_config.get("ranking", {}).get("rule_scorer", {})
    trials = build_rule_trials(max_trials=max_trials)

    summaries = []
    iterator = tqdm(trials, desc="rule scorer search", unit="trial", disable=not show_progress)
    total_trials = len(trials)
    for index, trial in enumerate(iterator, start=1):
        scorer = RuleScorer(
            journals=generator.store.journals,
            weights={**base_weights, **trial["weights"]},
        )
        result = evaluate_rule_trial(
            papers=papers,
            generator=generator,
            scorer=scorer,
            journal_name_to_id=name_to_id,
            mode=mode,
            candidate_top_k=candidate_top_k,
            show_progress=show_paper_progress,
            live_label=trial["label"],
            log_every=log_every,
        )
        summary = summarize_rule_trial(trial, result)
        summaries.append(summary)
        iterator.set_postfix({
            "best@20": max(item["rule_hit_at_20"] for item in summaries),
            "cur@20": summary["rule_hit_at_20"],
            "cur_mrr": f"{summary['rule_mrr']:.3f}",
        })
        status_line = format_rule_trial_status(index, total_trials, summary)
        if show_progress:
            tqdm.write(status_line)
        else:
            print(status_line)

        ranked = sort_rule_trials(summaries)
        partial_results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "papers_path": papers_path,
            "baseline_eval_path": baseline_eval_path,
            "config_path": config_path,
            "mode": mode,
            "candidate_top_k": candidate_top_k,
            "include_vector": include_vector,
            "paper_count": len(papers),
            "trial_count": len(trials),
            "completed_trial_count": len(summaries),
            "best": ranked[:10],
            "trials": summaries,
        }
        _write_search_results(output_path, partial_results)

    ranked = sort_rule_trials(summaries)
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "papers_path": papers_path,
        "baseline_eval_path": baseline_eval_path,
        "config_path": config_path,
        "mode": mode,
        "candidate_top_k": candidate_top_k,
        "include_vector": include_vector,
        "paper_count": len(papers),
        "trial_count": len(trials),
        "best": ranked[:10],
        "trials": summaries,
    }
    _write_search_results(output_path, results)
    return results


def print_summary(results: dict) -> None:
    print(f"Saved: {results.get('output_path', '')}")
    print("rank\trule@20\trule@10\trule_mrr\trule_ndcg5\tlabel")
    for rank, item in enumerate(results["best"][:10], start=1):
        print(
            "\t".join([
                str(rank),
                str(item["rule_hit_at_20"]),
                str(item["rule_hit_at_10"]),
                f"{item['rule_mrr']:.4f}",
                f"{item['rule_ndcg_at_5']:.4f}",
                item["label"],
            ])
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search RuleScorer settings on a fixed evaluation set.")
    parser.add_argument("--papers", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--baseline-eval", required=True)
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", default="abstract", choices=["title", "abstract", "full"])
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--paper-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or f"data/evaluation/results/rule_scorer_search_{args.mode}_{timestamp}.json"
    results = run_search(
        papers_path=args.papers,
        baseline_eval_path=args.baseline_eval,
        config_path=args.config,
        output_path=output,
        mode=args.mode,
        candidate_top_k=args.candidate_top_k,
        limit=args.limit,
        include_vector=not args.no_vector,
        max_trials=args.max_trials,
        show_progress=not args.no_progress,
        show_paper_progress=args.paper_progress,
        log_every=args.log_every,
    )
    results["output_path"] = output
    print_summary(results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Search coarse retrieval fusion settings on a fixed comparable evaluation set."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from scripts.run_retrieval_ablation import (
    build_candidate_generator,
    evaluate_variant,
    journal_name_to_id_map,
    load_comparable_eval_papers,
    route_config_for_mode,
)
from src.ranker.rule_scorer import RuleScorer


DEFAULT_WEIGHT_PAIRS = (
    (0.55, 0.45),
    (0.65, 0.35),
    (0.75, 0.25),
    (0.85, 0.15),
)
DEFAULT_IDENTITY_WEIGHTS = (0.0, 0.03, 0.06, 0.10)
DEFAULT_FUSION_STRATEGIES = ("weighted_minmax", "rrf", "weighted_rrf")
DEFAULT_ROUTE_CONFIGS = {
    "base": {},
    "vector40": {"vector": 40},
    "vector56": {"vector": 56},
    "semantic40": {"bm25": 40, "vector": 40},
    "semantic56": {"bm25": 56, "vector": 56},
}


def build_trials(
    mode: str,
    max_trials: int | None = None,
) -> list[dict]:
    base_cfg = route_config_for_mode(mode)
    trials = []
    for fusion_strategy in DEFAULT_FUSION_STRATEGIES:
        for scope_weight, typical_weight in DEFAULT_WEIGHT_PAIRS:
            for identity_weight in DEFAULT_IDENTITY_WEIGHTS:
                for route_label, overrides in DEFAULT_ROUTE_CONFIGS.items():
                    route_config = base_cfg.copy()
                    route_config.update(overrides)
                    trials.append({
                        "label": (
                            f"{fusion_strategy}|s{scope_weight:.2f}|"
                            f"t{typical_weight:.2f}|id{identity_weight:.2f}|{route_label}"
                        ),
                        "fusion_strategy": fusion_strategy,
                        "hybrid_scope_weight": scope_weight,
                        "hybrid_typical_weight": typical_weight,
                        "identity_anchor_weight": identity_weight,
                        "route_config_label": route_label,
                        "route_config": route_config,
                    })
                    if max_trials and len(trials) >= max_trials:
                        return trials
    return trials


def apply_trial(generator, trial: dict) -> None:
    generator.fusion_strategy = trial["fusion_strategy"]
    generator.hybrid_scope_weight = trial["hybrid_scope_weight"]
    generator.hybrid_typical_weight = trial["hybrid_typical_weight"]
    generator.identity_anchor_weight = trial["identity_anchor_weight"]


def summarize_trial(trial: dict, result: dict) -> dict:
    retrieval = result["retrieval"]
    rule = result["rule"]
    return {
        "label": trial["label"],
        "fusion_strategy": trial["fusion_strategy"],
        "hybrid_scope_weight": trial["hybrid_scope_weight"],
        "hybrid_typical_weight": trial["hybrid_typical_weight"],
        "identity_anchor_weight": trial["identity_anchor_weight"],
        "route_config_label": trial["route_config_label"],
        "route_config": trial["route_config"],
        "coarse_hit_at_50": result["coarse_hit_at_50"],
        "rule_hit_at_20": result["rule_hit_at_20"],
        "retrieval_mrr": retrieval["MRR"],
        "retrieval_ndcg_at_5": retrieval["NDCG@5"],
        "rule_mrr": rule["MRR"],
        "rule_ndcg_at_5": rule["NDCG@5"],
        "miss_stage_counts": result["miss_stage_counts"],
        "route_attribution": result["route_attribution"],
    }


def sort_trials(trials: list[dict]) -> list[dict]:
    return sorted(
        trials,
        key=lambda item: (
            item["coarse_hit_at_50"],
            item["retrieval_mrr"],
            item["retrieval_ndcg_at_5"],
            item["rule_hit_at_20"],
        ),
        reverse=True,
    )


def run_search(
    papers_path: str,
    baseline_eval_path: str,
    config_path: str,
    output_path: str,
    mode: str = "abstract",
    variant: str = "hybrid",
    candidate_top_k: int = 50,
    limit: int | None = None,
    include_vector: bool = True,
    max_trials: int | None = None,
    show_progress: bool = True,
    show_paper_progress: bool = False,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    generator = build_candidate_generator(app_config, include_vector=include_vector)
    scorer = RuleScorer(
        journals=generator.store.journals,
        weights=app_config.get("ranking", {}).get("rule_scorer", {}),
    )
    papers = load_comparable_eval_papers(papers_path, baseline_eval_path, limit=limit)
    name_to_id = journal_name_to_id_map(generator.store)
    trials = build_trials(mode, max_trials=max_trials)

    summaries = []
    iterator = tqdm(trials, desc="fusion search", unit="trial", disable=not show_progress)
    for trial in iterator:
        apply_trial(generator, trial)
        result = evaluate_variant(
            papers=papers,
            generator=generator,
            scorer=scorer,
            journal_name_to_id=name_to_id,
            variant=variant,
            mode=mode,
            candidate_top_k=candidate_top_k,
            show_progress=show_paper_progress,
            route_config=trial["route_config"],
        )
        summary = summarize_trial(trial, result)
        summaries.append(summary)
        iterator.set_postfix({
            "best@50": max(item["coarse_hit_at_50"] for item in summaries),
            "cur@50": summary["coarse_hit_at_50"],
            "cur_mrr": f"{summary['retrieval_mrr']:.3f}",
        })

    ranked = sort_trials(summaries)
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "papers_path": papers_path,
        "baseline_eval_path": baseline_eval_path,
        "config_path": config_path,
        "mode": mode,
        "variant": variant,
        "candidate_top_k": candidate_top_k,
        "include_vector": include_vector,
        "paper_count": len(papers),
        "trial_count": len(trials),
        "best": ranked[:10],
        "trials": summaries,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def print_summary(results: dict) -> None:
    print(f"Saved: {results.get('output_path', '')}")
    print("rank\tcoarse@50\trule@20\tret_mrr\tret_ndcg5\tlabel")
    for rank, item in enumerate(results["best"][:10], start=1):
        print(
            "\t".join([
                str(rank),
                str(item["coarse_hit_at_50"]),
                str(item["rule_hit_at_20"]),
                f"{item['retrieval_mrr']:.4f}",
                f"{item['retrieval_ndcg_at_5']:.4f}",
                item["label"],
            ])
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search hybrid retrieval fusion settings.")
    parser.add_argument("--papers", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--baseline-eval", required=True)
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", default="abstract", choices=["title", "abstract", "full"])
    parser.add_argument("--variant", default="hybrid", choices=["scope", "typical", "hybrid"])
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--paper-progress", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or f"data/evaluation/results/retrieval_fusion_search_{args.mode}_{timestamp}.json"
    results = run_search(
        papers_path=args.papers,
        baseline_eval_path=args.baseline_eval,
        config_path=args.config,
        output_path=output,
        mode=args.mode,
        variant=args.variant,
        candidate_top_k=args.candidate_top_k,
        limit=args.limit,
        include_vector=not args.no_vector,
        max_trials=args.max_trials,
        show_progress=not args.no_progress,
        show_paper_progress=args.paper_progress,
    )
    results["output_path"] = output
    print_summary(results)


if __name__ == "__main__":
    main()

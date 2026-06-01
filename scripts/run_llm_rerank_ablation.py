#!/usr/bin/env python3
"""Run LLM rerank ablations with shared retrieval and RuleScorer settings."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from scripts.run_evaluation import (
    init_pipeline,
    load_papers_metadata,
    print_report,
    run_evaluation,
)
from scripts.run_retrieval_ablation import load_comparable_eval_papers


@dataclass(frozen=True)
class RerankVariant:
    name: str
    prompt_path: str | None
    llm_final_selection: dict
    llm_anchor_guard: dict


BASE_ANCHOR = {
    "enabled": True,
    "protect_rule_rank": 5,
    "max_score_gap": 0.08,
}

FUSION_SELECTION = {
    "enabled": True,
    "strategy": "rule_route_fusion",
    "llm_weight": 0.70,
    "rule_rank_weight": 0.20,
    "route_evidence_weight": 0.10,
}

LLM_RERANK_ABLATION_VARIANTS = {
    "old_prompt": RerankVariant(
        name="old_prompt",
        prompt_path="configs/prompts_llm_legacy.yaml",
        llm_final_selection={"enabled": False, "strategy": "llm_only"},
        llm_anchor_guard=BASE_ANCHOR,
    ),
    "current_prompt": RerankVariant(
        name="current_prompt",
        prompt_path=None,
        llm_final_selection={"enabled": False, "strategy": "llm_only"},
        llm_anchor_guard=BASE_ANCHOR,
    ),
    "current_prompt_rule_route_fusion": RerankVariant(
        name="current_prompt_rule_route_fusion",
        prompt_path=None,
        llm_final_selection=FUSION_SELECTION,
        llm_anchor_guard=BASE_ANCHOR,
    ),
    "current_prompt_strong_anchor_guard": RerankVariant(
        name="current_prompt_strong_anchor_guard",
        prompt_path=None,
        llm_final_selection={"enabled": False, "strategy": "llm_only"},
        llm_anchor_guard={
            "enabled": True,
            "protect_rule_rank": 15,
            "max_score_gap": 0.12,
        },
    ),
}


def load_prompts_for_variant(variant: RerankVariant) -> dict:
    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    if variant.prompt_path:
        with open(variant.prompt_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        prompts.update(overrides)

    return prompts


def apply_variant(pipeline, prompts: dict, variant: RerankVariant):
    pipeline.llm_anchor_guard = deepcopy(variant.llm_anchor_guard)
    pipeline.llm_final_selection = deepcopy(variant.llm_final_selection)
    if pipeline.llm_ranker:
        pipeline.llm_ranker.system_prompt = prompts["llm_ranker_system"]
        pipeline.llm_ranker.user_prompt_template = prompts["llm_ranker_user"]
    return pipeline


def save_variant_result(result, variant_name: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"llm_rerank_ablation_{variant_name}_{result.mode}_top{result.top_k}_{timestamp}.json"
    result_dict = {
        "timestamp": timestamp,
        "variant": variant_name,
        "mode": result.mode,
        "top_k": result.top_k,
        "total_count": result.total_count,
        "metrics": {
            "hit_at_1": result.hit_at_1,
            "hit_at_3": result.hit_at_3,
            "hit_at_5": result.hit_at_5,
            "mrr": result.mrr / result.total_count if result.total_count else 0.0,
            "ndcg_at_5": result.ndcg_at_5 / result.total_count if result.total_count else 0.0,
            "coarse_hit_count": result.coarse_hit_count,
            "coarse_hit_in_rule_top20_count": result.coarse_hit_in_rule_top20_count,
            "acceptable_journal_hit_at_5": result.acceptable_journal_hit_at_5,
        },
        "by_area": dict(result.by_area),
        "by_level": dict(result.by_level),
        "paper_results": result.paper_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM rerank ablation runner")
    parser.add_argument("--input", "-i", default="data/evaluation/papers_metadata_v2.jsonl")
    parser.add_argument(
        "--baseline-eval",
        default="",
        help="Completed run_evaluation JSON used to reuse denominator and paper_profile_snapshot.",
    )
    parser.add_argument("--mode", "-m", choices=["title", "abstract", "full"], default="abstract")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--papers", "-n", type=int, default=None)
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(LLM_RERANK_ABLATION_VARIANTS),
        help="Variant to run. Repeat to run multiple. Default: all variants.",
    )
    parser.add_argument("--output-dir", default="data/evaluation/results")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--clean-benchmark", action="store_true")
    parser.add_argument("--clean-typical-dir", default="data/evaluation/clean_typical_abstracts")
    parser.add_argument("--clean-typical-faiss-path", default=None)
    parser.add_argument("--clean-typical-metadata-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.baseline_eval:
        papers = load_comparable_eval_papers(args.input, args.baseline_eval, limit=args.papers)
        print(
            "fair LLM ablation: reused denominator/profile snapshots from "
            f"{args.baseline_eval} ({len(papers)} papers)"
        )
    else:
        papers = load_papers_metadata(args.input)
        if args.papers:
            papers = papers[:args.papers]

    variants = args.variant or list(LLM_RERANK_ABLATION_VARIANTS)
    saved_paths = []
    summaries = []

    for variant_name in variants:
        variant = LLM_RERANK_ABLATION_VARIANTS[variant_name]
        print(f"\n{'=' * 70}")
        print(f"LLM rerank ablation: {variant.name}")
        print(f"{'=' * 70}")

        pipeline = init_pipeline(
            clean_typical_dir=args.clean_typical_dir if args.clean_benchmark else None,
            clean_typical_faiss_path=args.clean_typical_faiss_path,
            clean_typical_metadata_path=args.clean_typical_metadata_path,
        )
        prompts = load_prompts_for_variant(variant)
        apply_variant(pipeline, prompts, variant)

        result = run_evaluation(
            papers,
            pipeline,
            args.mode,
            args.top_k,
            prompts,
            show_progress=True,
            workers=args.workers,
            reuse_profile_snapshots=bool(args.baseline_eval),
        )
        print_report(result)
        summaries.append({
            "variant": variant_name,
            "evaluated": len(result.paper_results),
            "total": result.total_count,
            "hit_at_5": result.hit_at_5,
            "hit_at_1": result.hit_at_1,
            "mrr": result.mrr / result.total_count if result.total_count else 0.0,
            "acceptable_at_5": result.acceptable_journal_hit_at_5,
            "coarse_hit_count": result.coarse_hit_count,
            "rule_hit_at_20": result.coarse_hit_in_rule_top20_count,
        })
        if not args.no_save:
            saved_paths.append(save_variant_result(result, variant_name, args.output_dir))

    print("\nAblation summary")
    for summary in summaries:
        print(
            f"{summary['variant']}: "
            f"Hit@5={summary['hit_at_5']} | "
            f"Hit@1={summary['hit_at_1']} | "
            f"MRR={summary['mrr']:.4f} | "
            f"acceptable@5={summary['acceptable_at_5']} | "
            f"coarse@50={summary['coarse_hit_count']} | "
            f"rule@20={summary['rule_hit_at_20']} | "
            f"evaluated={summary['evaluated']}/{summary['total']}"
        )
    if summaries:
        coarse_values = {summary["coarse_hit_count"] for summary in summaries}
        rule_values = {summary["rule_hit_at_20"] for summary in summaries}
        evaluated_all = all(summary["evaluated"] == summary["total"] for summary in summaries)
        evaluated_labels = [
            f"{summary['evaluated']}/{summary['total']}"
            for summary in summaries
        ]
        if evaluated_all and len(coarse_values) == 1 and len(rule_values) == 1:
            print("Fairness check: PASS (coarse@50 and rule@20 are invariant across variants)")
        else:
            print(
                "Fairness check: WARN "
                f"(coarse@50={sorted(coarse_values)}, rule@20={sorted(rule_values)}, "
                f"evaluated={evaluated_labels})"
            )
    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare direct LLM ranking with structured-evidence ranking roles."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_evaluation import (  # noqa: E402
    init_pipeline,
    print_report,
    resolve_benchmark_input,
    run_evaluation,
)
from scripts.run_retrieval_ablation import load_comparable_eval_papers  # noqa: E402
from src.ranker.llm_evidence_extractor import LLMEvidenceExtractor  # noqa: E402
from src.ranker.llm_evidence_role_ranker import (  # noqa: E402
    DirectLLMRoleRanker,
    LLMEvidenceRoleRanker,
)


@dataclass(frozen=True)
class LLMRoleVariant:
    name: str
    ranker_role: str
    prior_source: str | None
    ltr_enabled: bool


LLM_ROLE_VARIANTS: Dict[str, LLMRoleVariant] = {
    "llm_ranker_direct": LLMRoleVariant(
        name="llm_ranker_direct",
        ranker_role="direct",
        prior_source=None,
        ltr_enabled=False,
    ),
    "llm_evidence_plus_rule": LLMRoleVariant(
        name="llm_evidence_plus_rule",
        ranker_role="evidence",
        prior_source="rule",
        ltr_enabled=False,
    ),
    "llm_evidence_plus_learned_reranker": LLMRoleVariant(
        name="llm_evidence_plus_learned_reranker",
        ranker_role="evidence",
        prior_source="learned",
        ltr_enabled=True,
    ),
}


def _paper_key(result: dict) -> str:
    title = " ".join(str(result.get("title", "")).casefold().split())
    venue = " ".join(str(result.get("venue", "")).casefold().split())
    return f"{title} | {venue}"


def _display_paper_key(result: dict) -> str:
    return f"{result.get('title', '')} | {result.get('venue', '')}"


def _title_key(title: str) -> str:
    return " ".join(str(title or "").casefold().split())


def load_evidence_snapshot(path: str, allow_partial: bool = False) -> dict:
    """Load a precompute snapshot and index entries by normalized paper title."""
    snapshot_path = Path(path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    papers = payload.get("papers")
    if not isinstance(papers, dict):
        raise ValueError(f"Evidence snapshot has no valid papers object: {path}")

    normalized: dict = {}
    partial: list[str] = []
    for outer_key, entry in papers.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid evidence snapshot paper entry: {outer_key}")
        title_key = _title_key(entry.get("title"))
        if not title_key:
            raise ValueError(f"Evidence snapshot entry has no title: {outer_key}")
        if title_key in normalized:
            raise ValueError(f"Duplicate normalized title in evidence snapshot: {title_key}")
        coverage = float(entry.get("evidence_coverage", 0.0))
        if coverage < 1.0:
            partial.append(f"{title_key} ({coverage:.3f})")
        normalized[title_key] = entry

    if partial and not allow_partial:
        raise ValueError(
            "Evidence snapshot contains partial coverage: " + ", ".join(partial[:5])
        )
    return normalized


def _evidence_fingerprint(row: dict) -> tuple:
    detail_by_id = {
        detail.get("journal_id"): detail
        for detail in row.get("llm_candidates_detail", [])
        if detail.get("journal_id")
    }
    fields = (
        "llm_scope_fit",
        "llm_method_fit",
        "llm_application_fit",
        "llm_journal_position_fit",
        "llm_too_broad_penalty",
        "llm_too_narrow_penalty",
        "evidence",
    )
    fingerprint = []
    for journal_id, detail in sorted(detail_by_id.items()):
        values = []
        for field in fields:
            value = detail.get(field)
            if field == "evidence":
                value = tuple(value) if isinstance(value, list) else ()
            values.append(value)
        fingerprint.append((journal_id, tuple(values)))
    return tuple(fingerprint)


def compare_variant_fairness(
    paper_results_by_variant: Dict[str, Iterable[dict]],
    require_full_evidence: bool = True,
) -> dict:
    """Compare denominator and per-paper retrieval/Rule stages across variants."""
    variants = list(paper_results_by_variant)
    if not variants:
        return {
            "fairness_pass": False,
            "denominator_match": False,
            "reference_variant": None,
            "coarse_mismatches": [],
            "rule_top20_mismatches": [],
            "evidence_mismatches": [],
            "evidence_coverage_failures": [],
            "missing_or_extra_by_variant": {},
        }

    maps = {
        variant: {_paper_key(row): row for row in rows}
        for variant, rows in paper_results_by_variant.items()
    }
    reference_variant = variants[0]
    reference = maps[reference_variant]
    reference_keys = set(reference)
    missing_or_extra: Dict[str, dict] = {}
    coarse_mismatches = set()
    rule_mismatches = set()
    evidence_mismatches = set()
    evidence_coverage_failures = set()

    for variant in variants[1:]:
        current = maps[variant]
        current_keys = set(current)
        missing = sorted(reference_keys - current_keys)
        extra = sorted(current_keys - reference_keys)
        if missing or extra:
            missing_or_extra[variant] = {"missing": missing, "extra": extra}
        for key in sorted(reference_keys & current_keys):
            if bool(reference[key].get("coarse_hit")) != bool(
                current[key].get("coarse_hit")
            ):
                coarse_mismatches.add(_display_paper_key(reference[key]))
            if bool(reference[key].get("coarse_hit_in_rule_top20")) != bool(
                current[key].get("coarse_hit_in_rule_top20")
            ):
                rule_mismatches.add(_display_paper_key(reference[key]))

    evidence_variants = [
        variant
        for variant, rows in maps.items()
        if any(row.get("llm_role") == "evidence" for row in rows.values())
    ]
    for variant in evidence_variants:
        for key, row in maps[variant].items():
            coverage = float(row.get("llm_evidence_coverage", 0.0))
            if coverage < 1.0:
                evidence_coverage_failures.add(
                    f"{variant}: {_display_paper_key(row)} ({coverage:.3f})"
                )
    if len(evidence_variants) >= 2:
        evidence_reference = maps[evidence_variants[0]]
        for variant in evidence_variants[1:]:
            current = maps[variant]
            for key in sorted(set(evidence_reference) & set(current)):
                if _evidence_fingerprint(evidence_reference[key]) != _evidence_fingerprint(
                    current[key]
                ):
                    evidence_mismatches.add(_display_paper_key(evidence_reference[key]))

    denominator_match = not missing_or_extra
    return {
        "fairness_pass": bool(
            denominator_match
            and not coarse_mismatches
            and not rule_mismatches
            and not evidence_mismatches
            and (not require_full_evidence or not evidence_coverage_failures)
        ),
        "denominator_match": denominator_match,
        "reference_variant": reference_variant,
        "coarse_mismatches": sorted(coarse_mismatches),
        "rule_top20_mismatches": sorted(rule_mismatches),
        "evidence_mismatches": sorted(evidence_mismatches),
        "evidence_coverage_failures": sorted(evidence_coverage_failures),
        "missing_or_extra_by_variant": missing_or_extra,
    }


def _load_config() -> tuple[dict, dict]:
    with open("configs/app.yaml", "r", encoding="utf-8") as handle:
        app_config = yaml.safe_load(handle)
    with open("configs/prompts.yaml", "r", encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle)
    return app_config, prompts


def configure_pipeline_for_variant(
    pipeline,
    variant: LLMRoleVariant,
    app_config: dict,
    prompts: dict,
    evidence_snapshot: Optional[dict] = None,
    ltr_score_weight: float = 0.0,
) -> dict:
    """Configure a fresh pipeline for one role-ablation variant.

    ``evidence_snapshot`` (Fix #1): when supplied, the new
    ``LLMEvidenceRoleRanker`` reads pre-computed evidence from it instead of
    calling the LLM extractor. The snapshot is shared verbatim across all
    evidence variants, which guarantees byte-identical evidence.
    """
    pipeline.llm_anchor_guard = {"enabled": False}
    # Use the raw LLMRanker (saved by init_pipeline as
    # ``pipeline.direct_llm_ranker``) so we never accidentally wrap an
    # LLMEvidenceRoleRanker in DirectLLMRoleRanker.
    direct_ranker = getattr(pipeline, "direct_llm_ranker", None) or pipeline.llm_ranker
    learned_reranker = pipeline.learned_reranker

    if variant.ranker_role == "direct":
        if direct_ranker is None:
            raise RuntimeError("Direct LLM role requires an initialized LLM ranker")
        pipeline.learned_reranker = None
        pipeline.llm_ranker = DirectLLMRoleRanker(direct_ranker)
        return {
            "ranker_role": "direct",
            "prior_source": None,
            "ltr_enabled": False,
            "anchor_guard_enabled": False,
            "feature_schema": "not_applicable",
            "evidence_source": "not_applicable",
        }

    if direct_ranker is None:
        raise RuntimeError("Evidence role requires an initialized direct LLM client")
    if variant.ltr_enabled and not (
        learned_reranker and getattr(learned_reranker, "enabled", False)
    ):
        reason = getattr(learned_reranker, "disable_reason", "adapter missing")
        raise RuntimeError(f"LTR variant requested but learned reranker is disabled: {reason}")

    accepted_store = getattr(learned_reranker, "_accepted_paper_store", None)
    extractor = LLMEvidenceExtractor(
        llm=direct_ranker.llm,
        system_prompt=prompts["llm_evidence_extractor_system"],
        user_prompt_template=prompts["llm_evidence_extractor_user"],
        timeout_seconds=app_config.get("ranking", {}).get(
            "llm_ranker_timeout_seconds", 420
        ),
    )
    pipeline.llm_ranker = LLMEvidenceRoleRanker(
        evidence_extractor=extractor,
        journal_store=pipeline.candidate_generator.store,
        accepted_paper_store=accepted_store,
        prior_source=str(variant.prior_source),
        evidence_weight=0.8,
        prior_weight=0.2,
        ltr_score_weight=ltr_score_weight,  # 6.4: LTR score as 3rd formula component
        evidence_snapshot=evidence_snapshot,
    )
    if not variant.ltr_enabled:
        pipeline.learned_reranker = None

    return {
        "ranker_role": "evidence",
        "prior_source": variant.prior_source,
        "ltr_enabled": variant.ltr_enabled,
        "anchor_guard_enabled": False,
        "evidence_weight": 0.8,
        "prior_weight": 0.2,
        "evidence_source": (
            "precomputed_snapshot" if evidence_snapshot is not None
            else "live_llm_call"
        ),
        "feature_schema": (
            "20_dim_ltr_prior_plus_26_dim_diagnostics"
            if variant.ltr_enabled
            else "26_dim_diagnostics_only"
        ),
    }


def _metrics(result) -> dict:
    total = result.total_count or 1
    evidence_neutral_fallback_count = sum(
        1
        for row in result.paper_results
        if row.get("llm_role_status") == "neutral_fallback"
    )
    return {
        "hit_at_1": result.hit_at_1,
        "hit_at_3": result.hit_at_3,
        "hit_at_5": result.hit_at_5,
        "hit_at_5_rate": result.hit_at_5 / total,
        "mrr": result.mrr / total,
        "ndcg_at_5": result.ndcg_at_5 / total,
        "same_area_hit_at_5": result.area_hit_at_5,
        "same_ccf_level_hit_at_5": result.level_hit_at_5,
        "acceptable_journal_hit_at_5": result.acceptable_journal_hit_at_5,
        "coarse_hit_count": result.coarse_hit_count,
        "rule_hit_at_20": result.coarse_hit_in_rule_top20_count,
        "fallback_count": result.fallback_count,
        "evidence_neutral_fallback_count": evidence_neutral_fallback_count,
        "empty_recommendation_count": result.empty_recommendation_count,
    }


def _save_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-profile",
        choices=["light30", "full-v2-90", "holdout240", "custom"],
        default="light30",
    )
    parser.add_argument("--input", "-i", default=None)
    parser.add_argument(
        "--baseline-eval",
        required=True,
        help="Completed evaluation JSON used to freeze denominator and paper_profile_snapshot.",
    )
    parser.add_argument("--mode", choices=["title", "abstract", "full"], default="abstract")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--papers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(LLM_ROLE_VARIANTS),
        help="Repeat to run selected variants. Default: all three.",
    )
    parser.add_argument("--output-dir", default="data/evaluation/results")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--evidence-snapshot",
        default=None,
        help=(
            "Pre-computed evidence JSON (output of precompute_evidence.py). "
            "When supplied, evidence variants skip their own LLM call and "
            "read from this snapshot, guaranteeing byte-identical evidence "
            "across rule/learned variants. Coverage < 1.0 is treated as a "
            "fatal error unless --allow-partial-snapshot is also set."
        ),
    )
    parser.add_argument(
        "--allow-partial-snapshot",
        action="store_true",
        help="Allow evidence snapshot with coverage < 1.0 (debug only).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Override the LTR model path from configs/app.yaml. Use this to "
            "swap between 16-dim base and 22-dim evidence LTR for ablation "
            "comparisons without editing the config. The model file's "
            "feature_dim must match what the LTR adapter expects (16 or 22)."
        ),
    )
    parser.add_argument(
        "--ltr-score-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the actual LTR score in the role ranker's final formula. "
            "When > 0, the role ranker uses (1-rank/N), (ltr_score), and "
            "evidence_composite with these weights renormalized to sum=1. "
            "Default 0 = use only evidence and rank (legacy 6.3 formula)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = resolve_benchmark_input(args.benchmark_profile, args.input)
    papers = load_comparable_eval_papers(
        args.input, args.baseline_eval, limit=args.papers
    )
    if not papers:
        raise RuntimeError("No comparable papers loaded from baseline evaluation")

    app_config, prompts = _load_config()
    if args.model_path:
        # Override the LTR model path before any pipeline is built. The
        # LTR adapter reads model_path at construction time, so we must
        # patch the config dict that init_pipeline will consume.
        app_config.setdefault("ranking", {}).setdefault(
            "learned_reranker", {}
        )["model_path"] = args.model_path
        print(f"Overriding LTR model path → {args.model_path}")
    evidence_snapshot = (
        load_evidence_snapshot(
            args.evidence_snapshot,
            allow_partial=args.allow_partial_snapshot,
        )
        if args.evidence_snapshot
        else None
    )
    if evidence_snapshot is not None:
        missing_titles = [
            paper.get("title", "")
            for paper in papers
            if _title_key(paper.get("title", "")) not in evidence_snapshot
        ]
        if missing_titles:
            raise ValueError(
                "Evidence snapshot is missing benchmark papers: "
                + ", ".join(missing_titles[:5])
            )
    variant_names = args.variant or list(LLM_ROLE_VARIANTS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_results_by_variant: Dict[str, list] = {}
    variant_payloads: Dict[str, dict] = {}
    saved_paths = []

    print(
        f"LLM role ablation: benchmark={args.benchmark_profile}, "
        f"papers={len(papers)}, workers={args.workers}, baseline={args.baseline_eval}"
    )

    for variant_name in variant_names:
        variant = LLM_ROLE_VARIANTS[variant_name]
        print(f"\n{'=' * 72}\nLLM role variant: {variant_name}\n{'=' * 72}")
        pipeline = init_pipeline()
        effective_config = configure_pipeline_for_variant(
            pipeline,
            variant,
            app_config,
            prompts,
            evidence_snapshot=evidence_snapshot,
            ltr_score_weight=args.ltr_score_weight,
        )
        result = run_evaluation(
            papers=papers,
            pipeline=pipeline,
            mode=args.mode,
            top_k=args.top_k,
            prompts=prompts,
            show_progress=True,
            workers=args.workers,
            reuse_profile_snapshots=True,
        )
        print_report(result)
        paper_results_by_variant[variant_name] = result.paper_results
        payload = {
            "timestamp": timestamp,
            "variant": variant_name,
            "benchmark_profile": args.benchmark_profile,
            "benchmark_path": args.input,
            "baseline_eval": args.baseline_eval,
            "mode": args.mode,
            "top_k": args.top_k,
            "workers": args.workers,
            "total_count": result.total_count,
            "effective_config": effective_config,
            "metrics": _metrics(result),
            "by_area": dict(result.by_area),
            "by_level": dict(result.by_level),
            "paper_results": result.paper_results,
        }
        variant_payloads[variant_name] = payload
        if not args.no_save:
            saved_paths.append(
                _save_json(
                    Path(args.output_dir)
                    / f"llm_role_ablation_{variant_name}_{args.mode}_top{args.top_k}_{timestamp}.json",
                    payload,
                )
            )

    fairness = compare_variant_fairness(
        paper_results_by_variant,
        require_full_evidence=not args.allow_partial_snapshot,
    )
    summary = {
        "timestamp": timestamp,
        "benchmark_profile": args.benchmark_profile,
        "benchmark_path": args.input,
        "baseline_eval": args.baseline_eval,
        "mode": args.mode,
        "top_k": args.top_k,
        "workers": args.workers,
        "variants": {
            name: {
                "effective_config": payload["effective_config"],
                "metrics": payload["metrics"],
            }
            for name, payload in variant_payloads.items()
        },
        "fairness": fairness,
    }
    if not args.no_save:
        summary_path = _save_json(
            Path(args.output_dir)
            / f"llm_role_ablation_summary_{args.mode}_top{args.top_k}_{timestamp}.json",
            summary,
        )
        saved_paths.append(summary_path)

    print("\nLLM role ablation summary")
    for name, payload in variant_payloads.items():
        metrics = payload["metrics"]
        print(
            f"{name}: Hit@5={metrics['hit_at_5']}/{payload['total_count']} | "
            f"MRR={metrics['mrr']:.4f} | "
            f"acceptable@5={metrics['acceptable_journal_hit_at_5']} | "
            f"coarse@50={metrics['coarse_hit_count']} | "
            f"rule@20={metrics['rule_hit_at_20']} | "
            f"fallback={metrics['fallback_count']} | "
            f"evidence_neutral={metrics['evidence_neutral_fallback_count']} | "
            f"empty={metrics['empty_recommendation_count']}"
        )
    print(
        "Fairness check: "
        + (
            "PASS (same denominator/stages and byte-identical evidence)"
            if fairness["fairness_pass"]
            else f"FAIL ({json.dumps(fairness, ensure_ascii=False)})"
        )
    )
    for path in saved_paths:
        print(f"Saved: {path}")

    if not fairness["fairness_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

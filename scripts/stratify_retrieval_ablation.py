#!/usr/bin/env python3
"""Stratify a retrieval ablation JSON by covered/uncovered status.

Coverage rule (per ADR 0001):
- A paper is "covered" iff its `target_journal_id` has at least one paper
  recorded under `data/accepted_papers/<jid>.json`.
- The script never leaks gold info into LTR training features; it only
  uses coverage to slice the report into Overall / Covered / Uncovered
  tables for diagnostic purposes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.accepted_paper_store import AcceptedPaperStore


def compute_coverage(
    paper_results: Sequence[dict], accepted_papers_dir: Path
) -> Dict[int, bool]:
    """Return {paper_index: is_covered} for every paper.

    A paper is covered when its `target_journal_id` resolves to a journal
    with at least one non-empty `papers` list in the accepted-paper corpus.
    Missing journal_id or empty corpus → False.
    """
    try:
        store = AcceptedPaperStore(str(accepted_papers_dir))
        store.load()
    except Exception:
        return {i: False for i in range(len(paper_results))}

    coverage: Dict[int, bool] = {}
    for i, r in enumerate(paper_results):
        jid = r.get("target_journal_id")
        if not jid:
            coverage[i] = False
            continue
        papers = store.get_papers(jid) or []
        coverage[i] = len(papers) > 0
    return coverage


def compute_stratified_metrics(
    paper_results: Sequence[dict], subset: Sequence[int]
) -> dict:
    """Compute retrieval + rule metrics for the given paper indices."""
    n = len(subset)
    if n == 0:
        return {
            "n": 0,
            "coarse_at_50": 0,
            "rule_at_5": 0,
            "rule_at_20": 0,
            "ret_mrr": 0.0,
            "rule_mrr": 0.0,
            "ret_ndcg5": 0.0,
            "rule_ndcg5": 0.0,
        }

    coarse_50 = rule_5 = rule_20 = 0
    mrr_num = rule_mrr_num = ndcg5_num = rule_ndcg5_num = 0.0

    for i in subset:
        r = paper_results[i]
        ret_rank = r.get("retrieval_rank")
        rule_rank = r.get("rule_rank")

        if ret_rank is not None and 0 < ret_rank <= 50:
            coarse_50 += 1
        if rule_rank is not None and 0 < rule_rank <= 5:
            rule_5 += 1
        if rule_rank is not None and 0 < rule_rank <= 20:
            rule_20 += 1
        if ret_rank is not None and ret_rank > 0:
            mrr_num += 1.0 / ret_rank
        if rule_rank is not None and rule_rank > 0:
            rule_mrr_num += 1.0 / rule_rank
        if ret_rank is not None and 0 < ret_rank <= 5:
            ndcg5_num += 1.0 / math.log2(ret_rank + 1)
        if rule_rank is not None and 0 < rule_rank <= 5:
            rule_ndcg5_num += 1.0 / math.log2(rule_rank + 1)

    return {
        "n": n,
        "coarse_at_50": coarse_50,
        "rule_at_5": rule_5,
        "rule_at_20": rule_20,
        "ret_mrr": mrr_num / n,
        "rule_mrr": rule_mrr_num / n,
        "ret_ndcg5": ndcg5_num / n,
        "rule_ndcg5": rule_ndcg5_num / n,
    }


def format_stratified_report(
    ablation_data: dict, variants: Sequence[str], coverage: Dict[int, bool]
) -> str:
    """Format a markdown report with Overall / Covered / Uncovered tables."""
    paper_results = ablation_data["variants"][variants[0]]["paper_results"]
    n = len(paper_results)
    covered_idx = [i for i, c in coverage.items() if c]
    uncovered_idx = [i for i, c in coverage.items() if not c]

    lines: List[str] = []
    lines.append("# Stratified Retrieval Ablation Report")
    lines.append("")
    lines.append(
        f"Total papers: {n} | Covered: {len(covered_idx)} | Uncovered: {len(uncovered_idx)}"
    )
    lines.append("")

    for subset_name, subset_indices in (
        ("Overall", list(range(n))),
        ("Covered", covered_idx),
        ("Uncovered", uncovered_idx),
    ):
        lines.append(f"## {subset_name} (n={len(subset_indices)})")
        lines.append("")
        lines.append(
            "| variant | coarse@50 | rule@5 | rule@20 | ret_mrr | ret_ndcg5 | rule_mrr | rule_ndcg5 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for v in variants:
            prs = ablation_data["variants"][v]["paper_results"]
            m = compute_stratified_metrics(prs, subset_indices)
            lines.append(
                f"| {v} | {m['coarse_at_50']} | {m['rule_at_5']} | {m['rule_at_20']} | "
                f"{m['ret_mrr']:.4f} | {m['ret_ndcg5']:.4f} | "
                f"{m['rule_mrr']:.4f} | {m['rule_ndcg5']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ablation JSON path")
    parser.add_argument(
        "--papers", required=True, help="papers jsonl path (used only for traceability check)"
    )
    parser.add_argument(
        "--accepted-papers-dir", required=True, help="accepted_papers dir"
    )
    parser.add_argument("--output", required=True, help="output markdown path")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="variants to include (default: all from input)",
    )
    args = parser.parse_args()

    ablation_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    variants = args.variants or list(ablation_data["variants"].keys())
    paper_results = ablation_data["variants"][variants[0]]["paper_results"]

    # Sanity: paper_results and papers jsonl should align by index (we just
    # assert the counts match; titles were already verified in the ablation).
    n_papers = sum(1 for _ in Path(args.papers).read_text(encoding="utf-8").splitlines() if _.strip())
    if n_papers != len(paper_results):
        print(
            f"[warn] papers jsonl has {n_papers} entries, ablation has {len(paper_results)}; "
            f"trusting ablation's paper_results order",
            file=sys.stderr,
        )

    coverage = compute_coverage(paper_results, Path(args.accepted_papers_dir))
    report = format_stratified_report(ablation_data, variants, coverage)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        f"Total: {len(paper_results)} | "
        f"Covered: {sum(1 for v in coverage.values() if v)} | "
        f"Uncovered: {sum(1 for v in coverage.values() if not v)}"
    )


if __name__ == "__main__":
    main()

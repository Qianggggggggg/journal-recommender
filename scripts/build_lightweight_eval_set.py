#!/usr/bin/env python3
"""Build a 30-paper lightweight evaluation set across CCF areas and levels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


CCF_AREAS = [
    "计算机体系结构/并行与分布计算/存储系统",
    "计算机网络",
    "网络与信息安全",
    "软件工程/系统软件/程序设计语言",
    "数据库/数据挖掘/内容检索",
    "计算机科学理论",
    "计算机图形学与多媒体",
    "人工智能",
    "人机交互与普适计算",
    "交叉/综合/新兴",
]

CCF_LEVELS = ["A", "B", "C"]


def load_jsonl(path: str | Path, source: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source"] = source
            rows.append(row)
    return rows


def paper_area(paper: dict) -> str:
    area = paper.get("research_area")
    if isinstance(area, list):
        return area[0] if area else ""
    return area or ""


def paper_key(paper: dict) -> tuple[str, str]:
    return (
        str(paper.get("title") or "").strip().lower(),
        str(paper.get("venue") or "").strip().lower(),
    )


def is_valid_paper(paper: dict) -> bool:
    return bool(
        paper.get("title")
        and paper.get("venue")
        and paper.get("abstract")
        and paper_area(paper)
        and paper.get("ccf_level")
    )


def strip_private_fields(paper: dict) -> dict:
    return {
        key: value
        for key, value in paper.items()
        if not key.startswith("_")
    }


def select_lightweight_papers(
    primary_papers: Iterable[dict],
    fallback_papers: Iterable[dict],
) -> tuple[list[dict], dict]:
    pools = {
        "primary": list(primary_papers),
        "fallback": list(fallback_papers),
    }
    selected = []
    selected_keys = set()
    coverage = []
    missing = []

    for area in CCF_AREAS:
        for level in CCF_LEVELS:
            chosen = None
            for source in ("primary", "fallback"):
                for paper in pools[source]:
                    if not is_valid_paper(paper):
                        continue
                    if paper_area(paper) != area or paper.get("ccf_level") != level:
                        continue
                    key = paper_key(paper)
                    if key in selected_keys:
                        continue
                    chosen = dict(paper)
                    chosen["_source"] = paper.get("_source", source)
                    break
                if chosen:
                    break

            if chosen:
                selected_keys.add(paper_key(chosen))
                selected.append(strip_private_fields(chosen))
                coverage.append({
                    "research_area": area,
                    "ccf_level": level,
                    "title": chosen.get("title", ""),
                    "venue": chosen.get("venue", ""),
                    "source": chosen.get("_source", "unknown"),
                })
            else:
                missing.append({
                    "research_area": area,
                    "ccf_level": level,
                })

    source_counts = Counter(item["source"] for item in coverage)
    report = {
        "summary": {
            "selected_count": len(selected),
            "expected_count": len(CCF_AREAS) * len(CCF_LEVELS),
            "missing_combo_count": len(missing),
        },
        "source_counts": dict(source_counts),
        "coverage": coverage,
        "missing_combos": missing,
    }
    return selected, report


def validate_lightweight_eval_set(path: str | Path) -> dict:
    papers = load_jsonl(path, "light30")
    combo_counts = Counter((paper_area(paper), paper.get("ccf_level")) for paper in papers)
    expected_combos = {
        (area, level)
        for area in CCF_AREAS
        for level in CCF_LEVELS
    }
    missing = [
        {"research_area": area, "ccf_level": level}
        for area, level in sorted(expected_combos)
        if combo_counts.get((area, level), 0) == 0
    ]
    duplicates = [
        {"research_area": area, "ccf_level": level, "count": count}
        for (area, level), count in sorted(combo_counts.items())
        if count > 1
    ]
    unexpected = [
        {"research_area": area, "ccf_level": level, "count": count}
        for (area, level), count in sorted(combo_counts.items())
        if (area, level) not in expected_combos
    ]
    selected_count = len(papers)
    report = {
        "valid": selected_count == len(expected_combos)
        and not missing
        and not duplicates
        and not unexpected,
        "summary": {
            "selected_count": selected_count,
            "expected_count": len(expected_combos),
            "missing_combo_count": len(missing),
            "duplicate_combo_count": len(duplicates),
            "unexpected_combo_count": len(unexpected),
        },
        "missing_combos": missing,
        "duplicate_combos": duplicates,
        "unexpected_combos": unexpected,
    }
    return report


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 30-paper lightweight eval set.")
    parser.add_argument("--primary", default="data/evaluation/papers_metadata_v2.jsonl")
    parser.add_argument("--fallback", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--output", default="data/evaluation/papers_metadata_light_30.jsonl")
    parser.add_argument(
        "--report",
        default="data/evaluation/papers_metadata_light_30_report.json",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the output light30 file without rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        report = validate_lightweight_eval_set(args.output)
        write_json(args.report, report)
        summary = report["summary"]
        print(
            "lightweight eval validation: "
            f"valid={report['valid']}, "
            f"selected={summary['selected_count']}/{summary['expected_count']}, "
            f"missing={summary['missing_combo_count']}, "
            f"duplicates={summary['duplicate_combo_count']}, "
            f"unexpected={summary['unexpected_combo_count']}, "
            f"input={args.output}, report={args.report}"
        )
        if not report["valid"]:
            raise SystemExit(1)
        return

    primary = load_jsonl(args.primary, "primary")
    fallback = load_jsonl(args.fallback, "fallback")
    selected, report = select_lightweight_papers(primary, fallback)

    write_jsonl(args.output, selected)
    write_json(args.report, report)

    summary = report["summary"]
    print(
        "lightweight eval set: "
        f"selected={summary['selected_count']}/{summary['expected_count']}, "
        f"missing={summary['missing_combo_count']}, "
        f"output={args.output}, report={args.report}"
    )
    if report["missing_combos"]:
        for item in report["missing_combos"]:
            print(f"missing: {item['research_area']} | {item['ccf_level']}")


if __name__ == "__main__":
    main()

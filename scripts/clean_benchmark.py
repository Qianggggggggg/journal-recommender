#!/usr/bin/env python3
"""Create leakage reports and clean typical-abstract snapshots."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.clean_benchmark import (
    build_clean_typical_snapshot,
    detect_typical_leakage,
    load_papers_jsonl,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect test leakage in typical abstracts.")
    parser.add_argument("--input", "-i", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--typical-dir", default="data/typical_abstracts")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write a clean typical-abstract snapshot here. Omit for report-only mode.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Leakage report JSON path. Defaults to data/evaluation/results/clean_benchmark_leakage_<timestamp>.json",
    )
    parser.add_argument("--fail-on-leak", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    papers = load_papers_jsonl(args.input)
    if args.output_dir:
        report = build_clean_typical_snapshot(
            papers,
            args.typical_dir,
            args.output_dir,
            overwrite=not args.no_overwrite,
        )
    else:
        report = detect_typical_leakage(papers, args.typical_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report or f"data/evaluation/results/clean_benchmark_leakage_{timestamp}.json"
    write_report(report, report_path)

    summary = report.summary
    print(f"papers={summary['paper_count']}")
    print(f"typical_files={summary['typical_file_count']}")
    print(f"leaked_papers={summary['leaked_paper_count']}")
    print(f"leaked_entries={summary['leaked_entry_count']}")
    if "removed_entry_count" in summary:
        print(f"removed_entries={summary['removed_entry_count']}")
        print(f"clean_typical_dir={summary['clean_typical_dir']}")
    print(f"report={report_path}")

    if args.fail_on_leak and summary["leaked_entry_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

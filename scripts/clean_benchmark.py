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
    detect_leakage,
    load_papers_jsonl,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect test leakage in typical abstracts and accepted-paper profiles.")
    parser.add_argument("--input", "-i", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--typical-dir", default="data/typical_abstracts")
    parser.add_argument(
        "--accepted-paper-dir",
        default=None,
        help=(
            "Optional directory of accepted-paper profiles (JSON files with a 'papers' "
            "array). When provided, the leakage scan includes this source in addition "
            "to --typical-dir. The accepted-paper scan is report-only; no cleaned "
            "accepted-paper directory is written."
        ),
    )
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
    accepted_dir = args.accepted_paper_dir
    if accepted_dir and not Path(accepted_dir).exists():
        print(
            f"warning: accepted-paper dir does not exist: {accepted_dir}",
            file=sys.stderr,
        )
        accepted_dir = None

    if args.output_dir:
        report = build_clean_typical_snapshot(
            papers,
            args.typical_dir,
            args.output_dir,
            overwrite=not args.no_overwrite,
        )
    else:
        report = detect_leakage(
            papers,
            typical_dir=args.typical_dir,
            accepted_paper_dir=accepted_dir,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report or f"data/evaluation/results/clean_benchmark_leakage_{timestamp}.json"
    write_report(report, report_path)

    summary = report.summary
    print(f"papers={summary['paper_count']}")
    print(f"typical_files={summary.get('typical_file_count', 0)}")
    print(f"accepted_paper_files={summary.get('accepted_paper_file_count', 0)}")
    print(f"leaked_papers={summary['leaked_paper_count']}")
    print(f"leaked_entries={summary['leaked_entry_count']}")
    if "leaked_typical_entry_count" in summary:
        print(f"leaked_typical_entries={summary['leaked_typical_entry_count']}")
        print(f"leaked_accepted_paper_entries={summary['leaked_accepted_paper_entry_count']}")
    if "removed_entry_count" in summary:
        print(f"removed_entries={summary['removed_entry_count']}")
        print(f"clean_typical_dir={summary['clean_typical_dir']}")
    print(f"report={report_path}")

    if args.fail_on_leak and summary["leaked_entry_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

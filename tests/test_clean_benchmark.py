import json
import subprocess
import sys
from pathlib import Path

from src.evaluation.clean_benchmark import (
    build_clean_typical_snapshot,
    detect_leakage,
)


def _write_typical(path: Path, journal_id: str, abstracts: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "journal_id": journal_id,
                "journal_name": f"{journal_id.upper()} Journal",
                "ccf_rating": "B",
                "abstracts": abstracts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_accepted(path: Path, journal_id: str, papers: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "journal_id": journal_id,
                "journal_name": f"{journal_id.upper()} Journal",
                "papers": papers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_detect_leakage_finds_title_overlap_in_typical(tmp_path):
    typical_dir = tmp_path / "typical"
    typical_dir.mkdir()
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "local_metadata_example",
                "abstract": "Goal-Oriented Medium Access with Distributed Belief Processing studies semantic communication.",
            },
            {
                "method_type": "模板",
                "novelty_level": "clean",
                "abstract": "This representative networking paper studies congestion control and routing.",
            },
        ],
    )
    papers = [
        {
            "title": "Goal-Oriented Medium Access with Distributed Belief Processing",
            "abstract": "This paper proposes a distributed belief processing protocol.",
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    report = detect_leakage(papers, typical_dir=typical_dir)

    assert report.summary["paper_count"] == 1
    assert report.summary["leaked_entry_count"] == 1
    assert report.summary["leaked_paper_count"] == 1
    assert report.matches[0]["match_type"] == "title"
    assert report.matches[0]["paper_title"] == papers[0]["title"]
    assert report.matches[0]["typical_file"] == "ton.json"


def test_build_clean_typical_snapshot_removes_leaked_entries_only(tmp_path):
    typical_dir = tmp_path / "typical"
    output_dir = tmp_path / "clean_typical"
    typical_dir.mkdir()
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "local_metadata_example",
                "abstract": "Goal-Oriented Medium Access with Distributed Belief Processing studies semantic communication.",
            },
            {
                "method_type": "模板",
                "novelty_level": "clean",
                "abstract": "This representative networking paper studies congestion control and routing.",
            },
        ],
    )
    papers = [
        {
            "title": "Goal-Oriented Medium Access with Distributed Belief Processing",
            "abstract": "This paper proposes a distributed belief processing protocol.",
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    report = build_clean_typical_snapshot(papers, typical_dir, output_dir)

    original = json.loads((typical_dir / "ton.json").read_text(encoding="utf-8"))
    cleaned = json.loads((output_dir / "ton.json").read_text(encoding="utf-8"))
    assert len(original["abstracts"]) == 2
    assert [item["novelty_level"] for item in cleaned["abstracts"]] == ["clean"]
    assert report.summary["removed_entry_count"] == 1
    assert report.summary["written_file_count"] == 1


def test_detect_leakage_scans_typical_and_accepted_papers(tmp_path):
    typical_dir = tmp_path / "typical"
    accepted_dir = tmp_path / "accepted"
    typical_dir.mkdir()
    accepted_dir.mkdir()

    paper_title = "Goal-Oriented Medium Access with Distributed Belief Processing"
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "local_metadata_example",
                "abstract": f"{paper_title} studies semantic communication.",
            },
        ],
    )
    _write_accepted(
        accepted_dir / "ton.json",
        "ton",
        [
            {
                "title": paper_title,
                "abstract": "Accepted paper about distributed belief processing.",
                "year": 2025,
                "source": "local_evaluation_metadata",
                "doi": "",
                "url": "",
            },
        ],
    )
    papers = [
        {
            "title": paper_title,
            "abstract": "This paper proposes a distributed belief processing protocol.",
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    report = detect_leakage(
        papers,
        typical_dir=typical_dir,
        accepted_paper_dir=accepted_dir,
    )

    source_types = {m["source_type"] for m in report.matches}
    assert source_types == {"typical_abstract", "accepted_paper"}
    typical_match = next(m for m in report.matches if m["source_type"] == "typical_abstract")
    accepted_match = next(m for m in report.matches if m["source_type"] == "accepted_paper")
    assert typical_match["typical_file"] == "ton.json"
    assert accepted_match["accepted_paper_file"] == "ton.json"
    assert accepted_match["journal_id"] == "ton"
    assert accepted_match["year"] == 2025
    assert accepted_match["source"] == "local_evaluation_metadata"
    assert report.summary["leaked_typical_entry_count"] == 1
    assert report.summary["leaked_accepted_paper_entry_count"] == 1
    assert report.summary["leaked_entry_count"] == 2
    assert report.summary["leaked_paper_count"] == 1
    assert report.summary["typical_file_count"] == 1
    assert report.summary["accepted_paper_file_count"] == 1


def test_detect_leakage_abstract_snippet_match_uses_160_chars(tmp_path):
    typical_dir = tmp_path / "typical"
    typical_dir.mkdir()

    paper_abstract = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50  # 203 chars
    assert len(paper_abstract) == 203

    # Build an entry whose haystack contains the first 160 normalized chars of
    # the paper abstract (truncate the first 160 chars and pad with something else).
    snippet_160 = paper_abstract[:160]
    # Sanity: with default 220, the abstract needle would NOT be created (203<220),
    # so the only way to get a match is via the 160-char snippet threshold.
    entry_abstract = f"{snippet_160} followed by filler text."

    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "snippet_match",
                "abstract": entry_abstract,
            },
        ],
    )
    papers = [
        {
            "title": "An Untouched Novel Title For Snippet Match",
            "abstract": paper_abstract,
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    report = detect_leakage(papers, typical_dir=typical_dir)

    abstract_matches = [m for m in report.matches if m["match_type"] == "abstract_snippet"]
    assert abstract_matches, f"Expected an abstract_snippet match, got {report.matches}"
    assert abstract_matches[0]["source_type"] == "typical_abstract"
    assert report.summary["leaked_entry_count"] == 1


def test_detect_leakage_accepted_paper_dir_optional(tmp_path):
    typical_dir = tmp_path / "typical"
    typical_dir.mkdir()
    paper_title = "Goal-Oriented Medium Access with Distributed Belief Processing"
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "local_metadata_example",
                "abstract": f"{paper_title} studies semantic communication.",
            },
        ],
    )
    papers = [
        {
            "title": paper_title,
            "abstract": "This paper proposes a distributed belief processing protocol.",
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    # Only typical -> behaves like the old detector
    only_typical = detect_leakage(papers, typical_dir=typical_dir)
    assert only_typical.summary["leaked_entry_count"] == 1
    assert only_typical.summary["accepted_paper_file_count"] == 0
    assert {m["source_type"] for m in only_typical.matches} == {"typical_abstract"}

    # No leakage anywhere
    clean_papers = [
        {
            "title": "An Unrelated Title That Matches No Entry",
            "abstract": "An abstract containing none of the test strings in any corpus entry.",
            "venue": "Some Other Venue",
        }
    ]
    empty_report = detect_leakage(
        clean_papers,
        typical_dir=typical_dir,
        accepted_paper_dir=tmp_path / "accepted",
    )
    assert empty_report.summary["leaked_entry_count"] == 0
    assert empty_report.summary["leaked_paper_count"] == 0
    assert empty_report.matches == []


def test_detect_leakage_missing_accepted_paper_dir_does_not_raise(tmp_path):
    typical_dir = tmp_path / "typical"
    typical_dir.mkdir()
    paper_title = "Goal-Oriented Medium Access with Distributed Belief Processing"
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "local_metadata_example",
                "abstract": f"{paper_title} studies semantic communication.",
            },
        ],
    )
    papers = [
        {
            "title": paper_title,
            "abstract": "This paper proposes a distributed belief processing protocol.",
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    # None
    report_none = detect_leakage(papers, typical_dir=typical_dir, accepted_paper_dir=None)
    assert report_none.summary["leaked_entry_count"] == 1
    # Non-existent path
    report_missing = detect_leakage(
        papers, typical_dir=typical_dir, accepted_paper_dir=tmp_path / "does_not_exist"
    )
    assert report_missing.summary["leaked_entry_count"] == 1
    # Empty dir
    empty_dir = tmp_path / "empty_accepted"
    empty_dir.mkdir()
    report_empty = detect_leakage(
        papers, typical_dir=typical_dir, accepted_paper_dir=empty_dir
    )
    assert report_empty.summary["leaked_entry_count"] == 1


def test_build_clean_typical_snapshot_unchanged_behavior(tmp_path):
    typical_dir = tmp_path / "typical"
    output_dir = tmp_path / "clean_typical"
    accepted_dir = tmp_path / "accepted"
    typical_dir.mkdir()
    accepted_dir.mkdir()

    paper_title = "Goal-Oriented Medium Access with Distributed Belief Processing"
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "真实代表论文",
                "novelty_level": "local_metadata_example",
                "abstract": f"{paper_title} studies semantic communication.",
            },
            {
                "method_type": "模板",
                "novelty_level": "clean",
                "abstract": "This representative networking paper studies congestion control and routing.",
            },
        ],
    )
    # Accepted paper also contains the same title - it should NOT be cleaned, only
    # reported (snapshot is report-only for accepted papers).
    _write_accepted(
        accepted_dir / "ton.json",
        "ton",
        [
            {
                "title": paper_title,
                "abstract": "An accepted paper with the leaked title.",
                "year": 2025,
                "source": "local_evaluation_metadata",
                "doi": "",
                "url": "",
            },
        ],
    )
    papers = [
        {
            "title": paper_title,
            "abstract": "This paper proposes a distributed belief processing protocol.",
            "venue": "IEEE/ACM Transactions on Networking",
        }
    ]

    report = build_clean_typical_snapshot(papers, typical_dir, output_dir)

    cleaned = json.loads((output_dir / "ton.json").read_text(encoding="utf-8"))
    # Snapshot is a typical-abstract dir only: no "papers" key, abstracts filtered.
    assert "abstracts" in cleaned
    assert "papers" not in cleaned
    assert [item["novelty_level"] for item in cleaned["abstracts"]] == ["clean"]
    assert report.summary["removed_entry_count"] == 1
    assert report.summary["written_file_count"] == 1
    # Accepted paper file was NOT created in the clean directory.
    assert not (output_dir / "accepted_ton.json").exists()


def test_cli_accepted_paper_dir_argument(tmp_path):
    paper_title = "Unique Title For Cli Smoke Test XYZ"
    paper_abstract = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50  # 203 chars

    papers_path = tmp_path / "papers.jsonl"
    papers_path.write_text(
        json.dumps(
            {
                "title": paper_title,
                "abstract": paper_abstract,
                "venue": "Some Venue",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    typical_dir = tmp_path / "typical"
    typical_dir.mkdir()
    _write_typical(
        typical_dir / "ton.json",
        "ton",
        [
            {
                "method_type": "模板",
                "novelty_level": "clean",
                "abstract": "An unrelated representative paper.",
            },
        ],
    )

    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    _write_accepted(
        accepted_dir / "ton.json",
        "ton",
        [
            {
                "title": paper_title,
                "abstract": paper_abstract,
                "year": 2025,
                "source": "local_evaluation_metadata",
                "doi": "",
                "url": "",
            },
        ],
    )

    report_path = tmp_path / "report.json"
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "clean_benchmark.py"),
        "--input",
        str(papers_path),
        "--typical-dir",
        str(typical_dir),
        "--accepted-paper-dir",
        str(accepted_dir),
        "--report",
        str(report_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    source_types = {m["source_type"] for m in payload["matches"]}
    assert "accepted_paper" in source_types
    assert payload["summary"]["leaked_accepted_paper_entry_count"] == 1

import json
from pathlib import Path

from src.evaluation.clean_benchmark import (
    build_clean_typical_snapshot,
    detect_typical_leakage,
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


def test_detect_typical_leakage_finds_title_overlap(tmp_path):
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

    report = detect_typical_leakage(papers, typical_dir)

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

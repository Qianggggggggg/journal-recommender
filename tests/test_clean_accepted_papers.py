import json
from pathlib import Path

from scripts.clean_accepted_papers import (
    classify_non_research_title,
    clean_accepted_papers,
    sanitize_metadata_text,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_profile(
    directory: Path,
    journal_id: str,
    journal_name: str,
    papers: list[dict],
) -> Path:
    path = directory / f"{journal_id}.json"
    path.write_text(
        json.dumps(
            {
                "journal_id": journal_id,
                "journal_name": journal_name,
                "papers": papers,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_non_research_filter_keeps_real_index_research():
    assert classify_non_research_title(
        "Parallel and scalable Dunn Index for the validation of big data clusters"
    ) is None
    assert (
        classify_non_research_title(
            "2020 Index IEEE Transactions on Computers Vol. 69"
        )
        == "index"
    )
    assert classify_non_research_title("RETRACTED ARTICLE: Unsafe result") == "retraction"
    assert classify_non_research_title("Corrigendum: Prior result") == "correction"
    assert classify_non_research_title("Introduction to the Special Issue on AI") == "editorial"
    assert classify_non_research_title("Journal Name: Preface") == "editorial"
    assert (
        classify_non_research_title(
            "Journal Name Special Issue on Reliable Computing"
        )
        == "editorial"
    )
    assert (
        classify_non_research_title(
            "Future trends: A message from the new Editor-in-Chief"
        )
        == "editorial"
    )
    assert (
        classify_non_research_title(
            "Communication from the Editor-in-Chief: State of the Journal"
        )
        == "editorial"
    )
    assert (
        classify_non_research_title("2019 IEEE Information Theory Society Paper Award")
        == "front_matter"
    )
    assert classify_non_research_title("In memoriam: A. Researcher") == "front_matter"

    assert (
        sanitize_metadata_text(
            "Abstract We use <i>graph</i> learning &amp; optimization.",
            abstract=True,
        )
        == "We use graph learning & optimization."
    )


def test_clean_removes_benchmark_non_research_and_cross_journal_duplicates(tmp_path):
    benchmark = tmp_path / "papers_metadata_660_balanced.jsonl"
    benchmark_abstract = "benchmark abstract content " * 20
    _write_jsonl(
        benchmark,
        [
            {
                "title": "A benchmark paper that must not leak",
                "abstract": benchmark_abstract,
                "doi": "10.1/benchmark",
            }
        ],
    )
    benchmark_before = benchmark.read_bytes()
    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    duplicate = {
        "title": "A duplicated research paper",
        "abstract": "valid duplicated abstract " * 20,
        "year": 2024,
        "source": "openalex",
        "doi": "10.1/duplicate",
        "url": "https://example.test/duplicate",
    }
    first = _write_profile(
        accepted_dir,
        "journal",
        "Journal",
        [
            {
                "title": "A benchmark paper that must not leak",
                "abstract": "different abstract",
                "doi": "10.1/benchmark",
            },
            {
                "title": "2020 Index IEEE Transactions on Tests Vol. 1",
                "abstract": "index metadata",
            },
            duplicate,
        ],
    )
    second = _write_profile(
        accepted_dir,
        "journal_2",
        "Journal",
        [dict(duplicate)],
    )

    report, _ = clean_accepted_papers(
        benchmark_path=benchmark,
        accepted_dir=accepted_dir,
        apply=True,
    )

    assert benchmark.read_bytes() == benchmark_before
    assert report["benchmark"]["unchanged"] is True
    assert report["summary"]["remaining_benchmark_overlap_count"] == 0
    assert report["summary"]["remaining_non_research_count"] == 0
    assert report["summary"]["remaining_low_quality_abstract_count"] == 0
    assert report["summary"]["remaining_cross_journal_duplicate_group_count"] == 0
    assert report["coverage_before"]["paper_count"] == 4
    assert report["coverage_after"]["paper_count"] == 1
    assert len(json.loads(first.read_text())["papers"]) == 1
    assert json.loads(second.read_text())["papers"] == []


def test_dry_run_does_not_rewrite_profiles(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        benchmark,
        [{"title": "Benchmark title long enough to match", "abstract": "x" * 200}],
    )
    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    profile = _write_profile(
        accepted_dir,
        "journal",
        "Journal",
        [
            {
                "title": "Benchmark title long enough to match",
                "abstract": "accepted abstract",
            }
        ],
    )
    before = profile.read_bytes()

    report, _ = clean_accepted_papers(
        benchmark_path=benchmark,
        accepted_dir=accepted_dir,
        apply=False,
    )

    assert report["mode"] == "dry-run"
    assert report["summary"]["removed_entry_count"] == 1
    assert profile.read_bytes() == before


def test_clean_normalizes_markup_and_removes_short_abstract(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        benchmark,
        [{"title": "Unrelated benchmark title", "abstract": "benchmark " * 40}],
    )
    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    profile = _write_profile(
        accepted_dir,
        "journal",
        "Journal",
        [
            {
                "title": "A <i>valid</i> research paper",
                "abstract": "Abstract " + "substantive research content " * 20,
                "year": 2025,
                "source": "openalex",
            },
            {
                "title": "A paper without a usable abstract",
                "abstract": "conference-version note only",
                "year": 2025,
                "source": "openalex",
            },
        ],
    )

    report, _ = clean_accepted_papers(
        benchmark_path=benchmark,
        accepted_dir=accepted_dir,
        apply=True,
    )

    papers = json.loads(profile.read_text())["papers"]
    assert len(papers) == 1
    assert papers[0]["title"] == "A valid research paper"
    assert papers[0]["abstract"].startswith("substantive research content")
    assert report["summary"]["normalized_entry_count"] == 1
    assert report["summary"]["removal_primary_counts"]["short_abstract"] == 1

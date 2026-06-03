import json
from pathlib import Path

from scripts.expand_accepted_paper_corpus import (
    CandidatePaper,
    candidate_from_openalex_work,
    reconstruct_openalex_abstract,
    build_semantic_scholar_request,
    build_blacklist,
    filter_candidates,
    merge_profile,
    target_uncovered_venues,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_target_uncovered_venues_uses_benchmarks_and_skips_existing_profiles(tmp_path):
    benchmark = tmp_path / "papers.jsonl"
    _write_jsonl(
        benchmark,
        [
            {"title": "A", "abstract": "alpha abstract", "venue": "Journal A"},
            {"title": "B", "abstract": "beta abstract", "venue": "Journal B"},
            {"title": "D", "abstract": "delta abstract", "venue": "Journal D"},
            {"title": "C", "abstract": "gamma abstract", "venue": "Unknown Journal"},
        ],
    )
    accepted_dir = tmp_path / "accepted"
    accepted_dir.mkdir()
    (accepted_dir / "ja.json").write_text(
        json.dumps(
            {
                "journal_id": "ja",
                "journal_name": "Journal A",
                "papers": [
                    {"title": "Existing", "abstract": "existing abstract", "year": 2024}
                ],
            }
        ),
        encoding="utf-8",
    )
    (accepted_dir / "jd.json").write_text(
        json.dumps(
            {
                "journal_id": "jd",
                "journal_name": "Journal D",
                "papers": [],
            }
        ),
        encoding="utf-8",
    )

    journals = {
        "journal a": {"journal_id": "ja", "journal_name": "Journal A"},
        "journal b": {"journal_id": "jb", "journal_name": "Journal B"},
        "journal d": {"journal_id": "jd", "journal_name": "Journal D"},
    }

    targets = target_uncovered_venues([benchmark], accepted_dir, journals)

    assert targets == [
        {"journal_id": "jb", "journal_name": "Journal B", "benchmark_count": 1},
        {"journal_id": "jd", "journal_name": "Journal D", "benchmark_count": 1},
    ]


def test_filter_candidates_excludes_test_leakage_and_wrong_venues(tmp_path):
    benchmark = tmp_path / "papers.jsonl"
    leaked_abstract = "This is a benchmark paper abstract with enough words to create a stable snippet. " * 4
    _write_jsonl(
        benchmark,
        [
            {
                "title": "Benchmark Title",
                "abstract": leaked_abstract,
                "venue": "Journal B",
            }
        ],
    )
    blacklist = build_blacklist([benchmark])
    candidates = [
        CandidatePaper(
            title="Benchmark Title",
            abstract="different abstract",
            venue="Journal B",
            year=2024,
            doi="10.1/leak-title",
            url="https://example.test/leak-title",
        ),
        CandidatePaper(
            title="Different title",
            abstract=leaked_abstract,
            venue="Journal B",
            year=2024,
            doi="10.1/leak-abstract",
            url="https://example.test/leak-abstract",
        ),
        CandidatePaper(
            title="Wrong venue",
            abstract="good abstract " * 80,
            venue="Other Journal",
            year=2024,
            doi="10.1/wrong",
            url="https://example.test/wrong",
        ),
        CandidatePaper(
            title="Good paper",
            abstract="good abstract " * 80,
            venue="Journal B",
            year=2024,
            doi="10.1/good",
            url="https://example.test/good",
        ),
    ]

    accepted = filter_candidates(
        candidates,
        target_venue="Journal B",
        blacklist=blacklist,
        existing_titles=set(),
        limit=3,
    )

    assert [paper.title for paper in accepted] == ["Good paper"]


def test_merge_profile_preserves_existing_and_limits_new_papers(tmp_path):
    profile_path = tmp_path / "jb.json"
    profile_path.write_text(
        json.dumps(
            {
                "journal_id": "jb",
                "journal_name": "Journal B",
                "papers": [
                    {
                        "title": "Existing",
                        "abstract": "existing abstract",
                        "year": 2023,
                        "source": "manual",
                        "doi": "",
                        "url": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    added = merge_profile(
        profile_path=profile_path,
        journal_id="jb",
        journal_name="Journal B",
        new_papers=[
            CandidatePaper(
                title="Existing",
                abstract="duplicate abstract",
                venue="Journal B",
                year=2024,
                doi="10.1/duplicate",
                url="https://example.test/duplicate",
            ),
            CandidatePaper(
                title="New 1",
                abstract="new abstract one",
                venue="Journal B",
                year=2024,
                doi="10.1/new1",
                url="https://example.test/new1",
            ),
            CandidatePaper(
                title="New 2",
                abstract="new abstract two",
                venue="Journal B",
                year=2025,
                doi="10.1/new2",
                url="https://example.test/new2",
            ),
        ],
        target_total=2,
        source="semantic_scholar",
    )

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    assert added == 1
    assert [paper["title"] for paper in data["papers"]] == ["Existing", "New 1"]
    assert data["papers"][1]["source"] == "semantic_scholar"


def test_semantic_scholar_request_uses_api_key_header_without_query_leakage():
    request = build_semantic_scholar_request(
        "https://api.semanticscholar.org/graph/v1/paper/search?query=test",
        api_key="secret-key",
    )

    assert "secret-key" in request.headers.values()
    assert "secret-key" not in request.full_url


def test_reconstruct_openalex_abstract_orders_inverted_index_tokens():
    inverted = {
        "Second": [1],
        "First": [0],
        "Third": [2],
    }

    assert reconstruct_openalex_abstract(inverted) == "First Second Third"


def test_candidate_from_openalex_work_requires_exact_source_name():
    work = {
        "title": "A backup-source paper",
        "publication_year": 2024,
        "doi": "https://doi.org/10.1000/example",
        "id": "https://openalex.org/W123",
        "abstract_inverted_index": {
            "This": [0],
            "paper": [1],
            "has": [2],
            "an": [3],
            "abstract": [4],
        },
        "primary_location": {
            "source": {"display_name": "Computer Communications"},
        },
    }

    candidate = candidate_from_openalex_work(work, target_venue="Computer Communications")

    assert candidate == CandidatePaper(
        title="A backup-source paper",
        abstract="This paper has an abstract",
        venue="Computer Communications",
        year=2024,
        doi="10.1000/example",
        url="https://openalex.org/W123",
    )
    assert candidate_from_openalex_work(work, target_venue="Other Venue") is None

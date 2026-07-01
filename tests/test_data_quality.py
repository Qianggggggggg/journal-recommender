"""Regression checks for local journal and typical-abstract data quality."""
import json
from pathlib import Path


def _load_journals():
    return [
        json.loads(line)
        for line in Path("data/processed/journals.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_duplicate_journal_names_are_explicit_alias_ids():
    journals = _load_journals()
    ids_by_name = {}
    for journal in journals:
        name = journal["journal_name"].strip().lower()
        ids_by_name.setdefault(name, []).append(journal["journal_id"])

    duplicates = {
        name: sorted(ids)
        for name, ids in ids_by_name.items()
        if len(ids) > 1
    }
    assert duplicates == {
        "data & knowledge engineering": ["dke", "dke_2"],
        "information processing letters": ["ipl", "ipl_2"],
        "international journal of intelligent systems": ["ijis", "ijis_2"],
    }


def test_journal_metadata_has_complete_scope_signals():
    for journal in _load_journals():
        assert journal.get("scope_text", "").strip()
        assert len(journal.get("keywords") or []) >= 10
        assert journal.get("subject_tags")


def test_low_accuracy_journals_use_real_accepted_paper_examples():
    for journal_id in ["ton", "toit", "sicomp", "tap", "bmcbioinformatics"]:
        path = Path(f"data/accepted_papers/{journal_id}.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        papers = data["papers"]

        assert len(papers) >= 5
        assert all((item.get("title") or "").strip() for item in papers)
        assert all((item.get("abstract") or "").strip() for item in papers)
        assert all((item.get("source") or "").strip() for item in papers)

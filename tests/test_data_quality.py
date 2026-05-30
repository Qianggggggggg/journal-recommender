"""Regression checks for local journal and typical-abstract data quality."""
import json
from collections import Counter
from pathlib import Path


def _load_journals():
    return [
        json.loads(line)
        for line in Path("data/processed/journals.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_journal_names_are_not_ambiguous():
    journals = _load_journals()
    names = Counter(journal["journal_name"].strip().lower() for journal in journals)

    assert [name for name, count in names.items() if count > 1] == []


def test_low_accuracy_journal_metadata_contains_subdomain_terms():
    journals = {journal["journal_id"]: journal for journal in _load_journals()}

    expected_terms = {
        "ton": ["goal-oriented communication", "age of information"],
        "toit": ["firmware version identification", "IoT intrusion detection"],
        "tdsc": ["coded blockchain", "blockchain for IoT"],
        "sicomp": ["meta-complexity", "minimum circuit size problem"],
        "appliedintelligence": ["high-utility pattern mining", "LiDAR semantic segmentation"],
        "bmcbioinformatics": ["multi-omics integration", "cancer subtype classification"],
    }

    for journal_id, terms in expected_terms.items():
        journal_text = " ".join([
            journals[journal_id]["scope_text"],
            " ".join(journals[journal_id]["keywords"]),
        ]).lower()
        for term in terms:
            assert term.lower() in journal_text


def test_typical_abstracts_include_real_local_examples_for_low_accuracy_journals():
    for journal_id in ["ton", "toit", "sicomp", "tap", "bmcbioinformatics"]:
        path = Path(f"data/typical_abstracts/{journal_id}.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        abstracts = data["abstracts"]

        assert len(abstracts) == 4
        assert any(item.get("method_type") == "真实代表论文" for item in abstracts)
        assert all((item.get("abstract") or "").strip() for item in abstracts)

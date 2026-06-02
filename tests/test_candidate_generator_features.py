"""Tests for CandidateGenerator.attach_features (Task 4.1.d)."""
import json
from pathlib import Path

from src.journals.accepted_paper_store import AcceptedPaperStore
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.retriever.candidate_generator import CandidateGenerator
from src.ranker.feature_builder import FEATURE_NAMES


class _DummyRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=10):
        return self.results[:top_k]


def _make_generator_with_journals(journal_specs: list[tuple[str, str | None]]) -> CandidateGenerator:
    """构造一个最小可用的 CandidateGenerator,store 里塞入指定期刊。"""
    store = JournalStore()
    for jid, ccf in journal_specs:
        store.add_journal(Journal(journal_id=jid, journal_name=jid.upper(), ccf_rating=ccf))
    return CandidateGenerator(
        store=store,
        bm25_retriever=_DummyRetriever([]),
        retrieval_target="scope_text",
    )


def test_candidate_generator_attach_features_uses_internal_store():
    """CandidateGenerator.attach_features 必须用 self.store 解析 journal,而不是要求 caller 传 store。"""
    gen = _make_generator_with_journals([("a", "A"), ("b", "B")])
    trace = {
        "a": {"total_score": 0.5, "routes": {"scope_bm25": {"rank": 2, "raw_score": 1.0, "normalized_score": 0.5, "weighted_score": 0.2}}},
        "b": {"total_score": 0.3, "routes": {}},
    }
    paper = PaperProfile(title="T", abstract="A", paper_strength=0.6)

    gen.attach_features(
        trace=trace,
        paper_profile=paper,
        rule_ranks={"a": 1, "b": 2},
        rule_scores={"a": 0.7, "b": 0.4},
        accepted_paper_store=None,
    )

    assert "features" in trace["a"]
    assert "features" in trace["b"]
    assert len(trace["a"]["features"]) == 20
    assert trace["a"]["feature_names"] == FEATURE_NAMES
    rule_idx = FEATURE_NAMES.index("rule_rank")
    assert trace["a"]["features"][rule_idx] == 1.0
    assert trace["b"]["features"][rule_idx] == 2.0


def test_candidate_generator_attach_features_marks_corpus_membership(tmp_path: Path):
    """accepted_paper_store 传入时,candidate_in_accepted_corpus 必须按 jid 正确填充。"""
    payload = {"journal_id": "a", "journal_name": "A", "papers": [{"title": "x", "abstract": "y" * 50}]}
    (tmp_path / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    accepted_store = AcceptedPaperStore(str(tmp_path))
    accepted_store.load()

    gen = _make_generator_with_journals([("a", "A"), ("b", "B")])
    trace = {
        "a": {"total_score": 0.5, "routes": {}},
        "b": {"total_score": 0.3, "routes": {}},
    }
    paper = PaperProfile(title="T", abstract="A")

    gen.attach_features(
        trace=trace,
        paper_profile=paper,
        rule_ranks=None,
        rule_scores=None,
        accepted_paper_store=accepted_store,
    )

    corpus_idx = FEATURE_NAMES.index("candidate_in_accepted_corpus")
    assert trace["a"]["features"][corpus_idx] == 1.0
    assert trace["b"]["features"][corpus_idx] == 0.0

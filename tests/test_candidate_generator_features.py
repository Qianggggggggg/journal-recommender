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


def test_attach_features_supports_26_dim_schema():
    """When feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE and evidence is
    supplied, each trace entry's features array must be 26 long."""
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE

    journals = [_journal("j1"), _journal("j2")]
    store = StubJournalStore(journals)
    cg = CandidateGenerator(
        store=store,
        bm25_retriever=_DummyRetriever([]),
        retrieval_target="scope_text",
    )
    trace = {
        "j1": {"retrieval_rank": 1, "routes": {}},
        "j2": {"retrieval_rank": 2, "routes": {}},
    }
    paper_evidence = {
        "j1": {"scope_fit": 0.9, "method_fit": 0.8, "application_fit": 0.7,
               "journal_position_fit": 0.85, "too_broad_penalty": 0.1, "too_narrow_penalty": 0.05},
        "j2": {"scope_fit": 0.4, "method_fit": 0.3, "application_fit": 0.5,
               "journal_position_fit": 0.2, "too_broad_penalty": 0.0, "too_narrow_penalty": 0.0},
    }
    profile = PaperProfile(title="P1")

    cg.attach_features(
        trace=trace,
        paper_profile=profile,
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 0.9, "j2": 0.8},
        feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE,
        llm_evidence_by_journal=paper_evidence,
    )

    assert len(trace["j1"]["features"]) == 26
    assert trace["j1"]["feature_names"] == FEATURE_NAMES_WITH_LLM_EVIDENCE
    # Last 6 entries should be the evidence values
    assert trace["j1"]["features"][20:] == [0.9, 0.8, 0.7, 0.85, 0.1, 0.05]
    assert len(trace["j2"]["features"]) == 26
    assert trace["j2"]["features"][20:] == [0.4, 0.3, 0.5, 0.2, 0.0, 0.0]


def _journal(jid: str) -> Journal:
    """小工具:创建一个只设 jid/name 的 Journal 实例。"""
    return Journal(journal_id=jid, journal_name=jid.upper())


class StubJournalStore:
    """最小可用的 JournalStore stub,提供 get_journal 与 journals。"""

    def __init__(self, journals):
        self._journals = list(journals)

    def get_journal(self, journal_id):
        for journal in self._journals:
            if journal.journal_id == journal_id:
                return journal
        return None

    @property
    def journals(self):
        return list(self._journals)


# ---------------------------------------------------------------------------
# 阶段 6.5 (P2-mini): 28-dim schema support in CandidateGenerator.attach_features
# ---------------------------------------------------------------------------


def test_attach_features_supports_28_dim_schema(tmp_path: Path):
    """CandidateGenerator.attach_features 接受 28-dim feature_names,
    输出 28 维 features。"""
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY

    gen = _make_generator_with_journals([("j1", "C")])
    # j1 加上 subject_tags 才能让 area_exclusivity 非零
    j1 = next(j for j in gen.store._journals if j.journal_id == "j1")
    j1.subject_tags = ["AI"]
    paper = PaperProfile(title="t", research_area=["AI"])
    trace = {"j1": {"retrieval_rank": 1, "routes": {}}}

    gen.attach_features(
        trace, paper,
        rule_ranks={"j1": 1},
        rule_scores={"j1": 0.5},
        feature_names=FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
        paper_anchor_area="AI",
        n_matching_in_pool=1,
    )

    assert len(trace["j1"]["features"]) == 28
    assert trace["j1"]["feature_names"] == FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY
    # tier C → 1.5; area_exclusivity 1/1 → 1.0
    assert trace["j1"]["features"][-2] == 1.5
    assert trace["j1"]["features"][-1] == 1.0


def test_attach_features_paper_anchor_area_passed_through_to_build_features(
    tmp_path: Path,
):
    """CandidateGenerator.attach_features 把 paper_anchor_area + n_matching
    透传给底层 attach_features_to_trace (验证 build_features 收到这两个参数)。"""
    from src.ranker.feature_builder import (
        FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
    )
    from src.ranker import feature_builder as fb

    captured_kwargs = {}
    orig_attach = fb.attach_features_to_trace

    def spy_attach(trace, paper_profile, journal_store, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return orig_attach(trace, paper_profile, journal_store, *args, **kwargs)

    import src.retriever.candidate_generator as cg_module
    monkey = __import__("pytest").MonkeyPatch()
    monkey.setattr(cg_module, "attach_features_to_trace", spy_attach)

    gen = _make_generator_with_journals([("j1", "B")])
    j1 = next(j for j in gen.store._journals if j.journal_id == "j1")
    j1.subject_tags = ["DB"]
    paper = PaperProfile(title="t", research_area=["AI"])
    trace = {"j1": {"retrieval_rank": 1, "routes": {}}}

    gen.attach_features(
        trace, paper,
        rule_ranks={"j1": 1},
        rule_scores={"j1": 0.5},
        feature_names=FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
        paper_anchor_area="DB",
        n_matching_in_pool=2,
    )
    monkey.undo()

    assert "paper_anchor_area" in captured_kwargs
    assert captured_kwargs["paper_anchor_area"] == "DB"
    assert "n_matching_in_pool" in captured_kwargs
    assert captured_kwargs["n_matching_in_pool"] == 2

"""召回模块测试"""
import pytest
import tempfile
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.candidate_generator import CandidateGenerator
from src.papers.paper_model import PaperProfile


class DummyRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=30):
        return self.results[:top_k]


def test_bm25_retriever():
    """测试 BM25 召回"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(store_path=f"{tmpdir}/journals.jsonl")
        journal = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai"],
            keywords=["machine learning"],
            scope_text="Artificial intelligence and machine learning",
            journal_profile="AI Journal artificial intelligence machine learning",
        )
        store.add_journal(journal)

        retriever = BM25Retriever(store)
        retriever.build_index()

        results = retriever.retrieve("machine learning", top_k=10)
        assert len(results) >= 1
        assert results[0][0].journal_id == "ai-journal"


def test_candidate_generator_merge():
    """测试候选生成器合并"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(store_path=f"{tmpdir}/journals.jsonl")
        journal1 = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai"],
            journal_profile="AI Journal",
        )
        journal2 = Journal(
            journal_id="cv-journal",
            journal_name="CV Journal",
            subject_tags=["cv"],
            journal_profile="CV Journal",
        )
        store.add_journal(journal1)
        store.add_journal(journal2)

        generator = CandidateGenerator(store, BM25Retriever(store))

        profile = PaperProfile(
            title="Deep Learning",
            research_area=["ai"],
            method_type="method",
        )
        candidates = generator.generate("deep learning artificial intelligence", profile, top_k=10)
        assert len(candidates) <= 10
        assert all(isinstance(j, Journal) for j in candidates)


def test_scope_text_mode_does_not_use_typical_routes():
    """scope_text 模式保持消融基线，不混入 typical 摘要召回。"""
    store = JournalStore()
    scope_journal = Journal(
        journal_id="scope",
        journal_name="Scope Journal",
        journal_profile="scope boundary",
    )
    typical_journal = Journal(
        journal_id="typical",
        journal_name="Typical Journal",
        journal_profile="semantic expansion",
    )
    store.add_journals([scope_journal, typical_journal])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(scope_journal, 10.0)]),
        retrieval_target="scope_text",
        typical_bm25_retriever=DummyRetriever([(typical_journal, 10.0)]),
    )

    candidates, trace = generator.generate_with_trace(
        "scope boundary",
        PaperProfile(title="scope boundary"),
        top_k=10,
    )

    assert [j.journal_id for j in candidates] == ["scope"]
    assert "scope_bm25" in trace["scope"]["routes"]
    assert trace["scope"]["retrieval_rank"] == 1
    assert all(not route.startswith("typical_") for route in trace["scope"]["routes"])


def test_typical_mode_mixes_scope_and_typical_routes_with_trace():
    """typical_abstracts 模式用 scope 做边界，用 typical 做语义补召回。"""
    store = JournalStore()
    scope_journal = Journal(
        journal_id="scope",
        journal_name="Scope Journal",
        journal_profile="scope boundary",
    )
    typical_journal = Journal(
        journal_id="typical",
        journal_name="Typical Journal",
        journal_profile="semantic expansion",
    )
    store.add_journals([scope_journal, typical_journal])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(scope_journal, 10.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(typical_journal, 10.0)]),
        typical_text_retriever=DummyRetriever([(typical_journal, 8.0)]),
    )

    candidates, trace = generator.generate_with_trace(
        "semantic expansion",
        PaperProfile(title="semantic expansion"),
        top_k=10,
    )

    candidate_ids = {j.journal_id for j in candidates}
    assert {"scope", "typical"} <= candidate_ids
    assert "scope_bm25" in trace["scope"]["routes"]
    assert "typical_bm25" in trace["typical"]["routes"]
    assert "typical_text" in trace["typical"]["routes"]
    assert "identity_anchor" in trace["typical"]["routes"]
    assert all("retrieval_rank" in trace[journal_id] for journal_id in candidate_ids)


def test_diagnostic_journal_trace_keeps_wide_rank_outside_top_k():
    """评估诊断期刊即使不进最终 top_k，也要保留宽召回 trace。"""
    store = JournalStore()
    first = Journal(journal_id="first", journal_name="First Journal", journal_profile="dominant match")
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        journal_profile="diagnostic match",
    )
    store.add_journals([first, target])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(first, 10.0), (target, 9.0)]),
        retrieval_target="scope_text",
    )

    candidates, trace = generator.generate_with_trace(
        "diagnostic match",
        PaperProfile(title="diagnostic match"),
        top_k=1,
        diagnostic_journal_ids=["target"],
    )

    assert [j.journal_id for j in candidates] == ["first"]
    assert "target" in trace
    assert trace["target"]["wide_retrieval_rank"] == 2
    assert "retrieval_rank" not in trace["target"]

"""召回模块测试"""
import pytest
import tempfile
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.candidate_generator import CandidateGenerator
from src.papers.paper_model import PaperProfile


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
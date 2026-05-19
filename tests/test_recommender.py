"""推荐流程测试"""
import pytest
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperInput, PaperProfile
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.bm25_retriever import BM25Retriever
from src.ranker.rule_scorer import RuleScorer
from src.recommender.pipeline import RecommenderPipeline


def test_pipeline_integration():
    """测试完整流程（不含 LLM）"""
    from src.journals.journal_store import JournalStore
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(store_path=f"{tmpdir}/journals.jsonl")
        journal = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai"],
            keywords=["machine learning"],
            scope_text="Artificial intelligence",
            journal_profile="AI Journal artificial intelligence",
            target_paper_type=["method"],
            quartile="Q1",
        )
        store.add_journal(journal)

        generator = CandidateGenerator(
            store, BM25Retriever(store), embedding_retriever=None
        )
        scorer = RuleScorer()
        pipeline = RecommenderPipeline(
            candidate_generator=generator,
            rule_scorer=scorer,
        )

        paper_input = PaperInput(title="Deep Learning for AI")
        profile = PaperProfile(
            title="Deep Learning for AI",
            research_area=["ai"],
            method_type="method",
        )

        result = pipeline.recommend(paper_input, profile, mode="title")
        assert "recommendations" in result
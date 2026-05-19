"""排序模块测试"""
import pytest
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperProfile
from src.ranker.rule_scorer import RuleScorer


def test_rule_scorer():
    """测试规则打分"""
    scorer = RuleScorer()
    journal = Journal(
        journal_id="ai-journal",
        journal_name="AI Journal",
        subject_tags=["ai"],
        target_paper_type=["method", "experiment"],
        quartile="Q1",
        oa_type="full_oa",
    )
    profile = PaperProfile(
        title="Deep Learning",
        research_area=["ai"],
        method_type="method",
        paper_type="application",
    )

    score, reasons = scorer.score(journal, profile, oa_preference="any")
    assert score > 0
    assert len(reasons) >= 2  # 至少领域匹配和分区加分


def test_rule_scorer_rank():
    """测试规则排序"""
    scorer = RuleScorer()
    journals = [
        Journal(journal_id="j1", journal_name="J1", subject_tags=["ai"], quartile="Q1"),
        Journal(journal_id="j2", journal_name="J2", subject_tags=["cv"], quartile="Q2"),
        Journal(journal_id="j3", journal_name="J3", subject_tags=["ai"], quartile="Q2"),
    ]
    profile = PaperProfile(title="Test", research_area=["ai"])

    ranked = scorer.rank(journals, profile, top_k=2)
    assert len(ranked) <= 2
    # AI 期刊应该在前面
    assert ranked[0][0].journal_id == "j1"
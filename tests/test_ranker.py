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
        scope_text="Deep learning and artificial intelligence application research",
        target_paper_type=["method", "experiment"],
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
        Journal(journal_id="j1", journal_name="J1", subject_tags=["ai"]),
        Journal(journal_id="j2", journal_name="J2", subject_tags=["cv"]),
        Journal(journal_id="j3", journal_name="J3", subject_tags=["ai"]),
    ]
    profile = PaperProfile(title="Test", research_area=["ai"])

    ranked = scorer.rank(journals, profile, top_k=2)
    assert len(ranked) <= 2
    # AI 期刊应该在前面
    assert ranked[0][0].journal_id == "j1"


def test_rule_scorer_prefers_scope_evidence_over_typical_only():
    """同等规则分下，scope 边界证据应优先于 pure typical 补召回。"""
    scorer = RuleScorer()
    scope_journal = Journal(journal_id="scope", journal_name="Scope Journal")
    typical_journal = Journal(journal_id="typical", journal_name="Typical Journal")
    profile = PaperProfile(title="Neutral Title")
    retrieval_trace = {
        "scope": {
            "routes": {
                "scope_bm25": {"rank": 1, "weighted_score": 0.2},
            }
        },
        "typical": {
            "routes": {
                "typical_bm25": {"rank": 1, "weighted_score": 0.2},
            }
        },
    }

    ranked = scorer.rank(
        [typical_journal, scope_journal],
        profile,
        top_k=2,
        retrieval_trace=retrieval_trace,
    )

    assert ranked[0][0].journal_id == "scope"
    assert ranked[0][1] > ranked[1][1]
    assert any("范围文本" in reason for reason in ranked[0][2])


def test_rule_scorer_treats_identity_anchor_as_expansion_not_boundary():
    """identity_anchor 只作为扩展证据，不应等同于 scope 边界。"""
    scorer = RuleScorer()
    scope_journal = Journal(journal_id="scope", journal_name="Scope Journal")
    identity_journal = Journal(journal_id="identity", journal_name="Identity Journal")
    profile = PaperProfile(title="Neutral Title")
    retrieval_trace = {
        "scope": {
            "routes": {
                "scope_bm25": {"rank": 1, "weighted_score": 0.15},
            }
        },
        "identity": {
            "routes": {
                "identity_anchor": {"rank": 1, "weighted_score": 0.15},
            }
        },
    }

    ranked = scorer.rank(
        [identity_journal, scope_journal],
        profile,
        top_k=2,
        retrieval_trace=retrieval_trace,
    )

    assert ranked[0][0].journal_id == "scope"
    assert any("仅有补充语义证据" in reason for reason in ranked[1][2])


def test_rule_scorer_does_not_score_research_area_directly():
    """research_area 只作为解释信号，不直接制造分数优势。"""
    scorer = RuleScorer()
    profile = PaperProfile(title="Neutral Title", research_area=["人工智能"])
    journal = Journal(
        journal_id="area",
        journal_name="Area Journal",
        subject_tags=["人工智能"],
    )

    score, reasons = scorer.score(journal, profile)

    assert score == 0
    assert any("领域标签对齐" in reason for reason in reasons)

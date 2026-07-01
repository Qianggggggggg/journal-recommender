"""排序模块测试"""
import pytest
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperProfile
from src.ranker.llm_ranker import LLMRanker
from src.ranker.rule_scorer import RuleScorer


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _RecordingLLM:
    def __init__(self):
        self.user_prompt = ""
        self.timeout = None

    def chat_auto(self, system_prompt, user_prompt, timeout=200):
        self.user_prompt = user_prompt
        self.timeout = timeout
        return _FakeLLMResponse(
            '{"rankings":[{"journal_id":"target","score":0.9,"reasons":["ok"],"confidence":0.8}]}'
        )


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


def test_rule_scorer_can_use_configured_retrieval_rank_prior():
    """配置 retrieval_rank_prior 后，强粗排证据应能进入规则排序。"""
    scorer = RuleScorer(
        weights={
            "scope_boundary_evidence": 0.0,
            "typical_scope_synergy": 0.0,
            "retrieval_rank_prior": 1.0,
        }
    )
    target = Journal(journal_id="target", journal_name="Target Journal")
    distractor = Journal(journal_id="distractor", journal_name="Distractor Journal")
    profile = PaperProfile(title="Neutral Title")
    retrieval_trace = {
        "target": {
            "retrieval_rank": 1,
            "routes": {
                "scope_vector": {"rank": 12, "weighted_score": 0.0},
            },
        },
        "distractor": {
            "retrieval_rank": 50,
            "routes": {},
        },
    }

    ranked = scorer.rank(
        [distractor, target],
        profile,
        top_k=2,
        retrieval_trace=retrieval_trace,
    )

    assert ranked[0][0].journal_id == "target"
    assert any("粗排排名证据" in reason for reason in ranked[0][2])


def test_rule_scorer_can_use_configured_strong_typical_rank_bonus():
    """典型摘要强命中可作为可配置软信号，补足 scope 边界较弱的召回。"""
    scorer = RuleScorer(
        weights={
            "scope_boundary_evidence": 0.0,
            "typical_scope_synergy": 0.0,
            "retrieval_rank_prior": 0.0,
            "strong_scope_rank_bonus": 0.0,
            "strong_typical_rank_bonus": 1.0,
        }
    )
    target = Journal(journal_id="target", journal_name="Target Journal")
    distractor = Journal(journal_id="distractor", journal_name="Distractor Journal")
    profile = PaperProfile(title="Neutral Title")
    retrieval_trace = {
        "target": {
            "retrieval_rank": 11,
            "routes": {
                "scope_vector": {"rank": 29, "weighted_score": 0.04},
                "typical_vector": {"rank": 2, "weighted_score": 0.08},
                "typical_bm25": {"rank": 3, "weighted_score": 0.05},
            },
        },
        "distractor": {
            "retrieval_rank": 20,
            "routes": {},
        },
    }

    ranked = scorer.rank(
        [distractor, target],
        profile,
        top_k=2,
        retrieval_trace=retrieval_trace,
    )

    assert ranked[0][0].journal_id == "target"
    assert any("强典型摘要召回证据" in reason for reason in ranked[0][2])


def test_rule_scorer_research_area_weight_is_positive_only():
    """领域对齐配置为正向软信号，不命中时不扣分。"""
    scorer = RuleScorer(weights={"research_area_match": 0.5})
    profile = PaperProfile(title="Neutral Title", research_area=["人工智能"])
    matched = Journal(
        journal_id="matched",
        journal_name="Matched Journal",
        subject_tags=["人工智能"],
    )
    unmatched = Journal(
        journal_id="unmatched",
        journal_name="Unmatched Journal",
        subject_tags=["网络与信息安全"],
    )

    matched_score, matched_reasons = scorer.score(matched, profile)
    unmatched_score, unmatched_reasons = scorer.score(unmatched, profile)

    assert matched_score == 0.5
    assert unmatched_score == 0
    assert any("领域标签对齐" in reason for reason in matched_reasons)
    assert not unmatched_reasons


def test_llm_ranker_does_not_expose_internal_retrieval_fields():
    """LLM 精排输入不应暴露内部召回强度字段，避免模型被诊断字段牵引。"""
    llm = _RecordingLLM()
    ranker = LLMRanker(
        llm,
        "system",
        "候选期刊：{journals_info}\n论文：{title}\n总数：{total_candidates}",
    )
    journal = Journal(
        journal_id="target",
        journal_name="Target Journal",
        scope_text="database systems and entity linking",
    )

    ranker.rank(
        [(journal, 1.0, ["scope matched"])],
        PaperProfile(title="Entity linking"),
        retrieval_trace={
            "target": {
                "routes": {
                    "scope_vector": {"weighted_score": 0.2},
                    "typical_vector": {"weighted_score": 0.1},
                }
            }
        },
    )

    assert "retrieval_sources_summary" not in llm.user_prompt
    assert "scope_boundary_strength" not in llm.user_prompt
    assert "typical_expansion_strength" not in llm.user_prompt
    assert "rule_score" not in llm.user_prompt


def test_llm_ranker_uses_configured_timeout():
    llm = _RecordingLLM()
    ranker = LLMRanker(
        llm,
        "system",
        "候选期刊：{journals_info}\n论文：{title}\n总数：{total_candidates}",
        timeout_seconds=420,
    )
    journal = Journal(journal_id="target", journal_name="Target Journal")

    ranker.rank([(journal, 1.0, ["ok"])], PaperProfile(title="Entity linking"))

    assert llm.timeout == 420

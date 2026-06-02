"""推荐流程测试"""
import pytest
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperInput, PaperProfile
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.bm25_retriever import BM25Retriever
from src.ranker.rule_scorer import RuleScorer
from src.recommender.pipeline import RecommenderPipeline


class DummyGenerator:
    def __init__(self, candidates, trace):
        self.candidates = candidates
        self.trace = trace

    def generate_with_trace(
        self,
        query_text,
        paper_profile,
        top_k=40,
        mode="abstract",
        diagnostic_journal_ids=None,
    ):
        return self.candidates[:top_k], self.trace


class FixedRuleScorer:
    def __init__(self, ranked):
        self.ranked = ranked
        self.received_trace = None

    def rank(self, journals, paper_profile, oa_preference="any", top_k=10, retrieval_trace=None):
        self.received_trace = retrieval_trace
        journal_ids = {j.journal_id for j in journals}
        return [item for item in self.ranked if item[0].journal_id in journal_ids][:top_k]


class FixedLLMRanker:
    def __init__(self, ranked):
        self.ranked = ranked
        self.received_top_k = None

    def rank(self, candidates, paper_profile, top_k=5, retrieval_trace=None):
        self.received_top_k = top_k
        candidate_ids = {journal.journal_id for journal, _, _ in candidates}
        return [
            item for item in self.ranked if item[0].journal_id in candidate_ids
        ][:top_k], "llm"


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


def test_pipeline_returns_llm_pool_and_prefers_scope_supplements():
    """rule top20 外的 scope 边界候选应优先进入 LLM 候选池。"""
    journals = [
        Journal(journal_id=f"j{i}", journal_name=f"Journal {i}")
        for i in range(20)
    ]
    scope_late = Journal(journal_id="scope-late", journal_name="Scope Late")
    typical_late = Journal(journal_id="typical-late", journal_name="Typical Late")
    identity_late = Journal(journal_id="identity-late", journal_name="Identity Late")
    journals.extend([scope_late, typical_late, identity_late])

    trace = {
        **{
            journal.journal_id: {
                "retrieval_rank": i + 1,
                "total_score": 0.5,
                "primary_routes": ["scope_bm25"],
                "routes": {"scope_bm25": {"rank": i + 1, "weighted_score": 0.1}},
            }
            for i, journal in enumerate(journals[:20])
        },
        "scope-late": {
            "retrieval_rank": 21,
            "total_score": 0.2,
            "primary_routes": ["scope_bm25"],
            "routes": {"scope_bm25": {"rank": 3, "weighted_score": 0.08}},
        },
        "typical-late": {
            "retrieval_rank": 22,
            "total_score": 0.2,
            "primary_routes": ["typical_bm25"],
            "routes": {"typical_bm25": {"rank": 1, "weighted_score": 0.2}},
        },
        "identity-late": {
            "retrieval_rank": 23,
            "total_score": 0.2,
            "primary_routes": ["identity_anchor"],
            "routes": {"identity_anchor": {"rank": 1, "weighted_score": 0.2}},
        },
    }
    ranked = [
        (journal, 1.0 - i * 0.01, [])
        for i, journal in enumerate(journals[:20])
    ] + [
        (scope_late, 0.2, []),
        (typical_late, 0.1, []),
        (identity_late, 0.1, []),
    ]
    scorer = FixedRuleScorer(ranked)
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=scorer,
    )

    result = pipeline.recommend(
        PaperInput(title="Test"),
        PaperProfile(title="Test"),
        top_k=5,
    )

    assert scorer.received_trace is trace
    assert "scope-late" in result["llm_candidate_ids"]
    assert "typical-late" not in result["llm_candidate_ids"]
    assert "identity-late" not in result["llm_candidate_ids"]


def test_pipeline_keeps_full_llm_ranking_and_restores_close_rule_anchor():
    """LLM Top5 边缘结果应允许靠前 Rule 候选在分差很小时回到最终列表。"""
    protected = Journal(journal_id="protected", journal_name="Protected Journal")
    llm_top = [
        Journal(journal_id=f"llm-{i}", journal_name=f"LLM Journal {i}")
        for i in range(5)
    ]
    journals = [protected] + llm_top
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [
        (journal, 1.0 - i * 0.05, [])
        for i, journal in enumerate(journals)
    ]
    llm_ranked = [
        (llm_top[0], 0.95, ["llm"], 0.8),
        (llm_top[1], 0.94, ["llm"], 0.8),
        (llm_top[2], 0.93, ["llm"], 0.8),
        (llm_top[3], 0.92, ["llm"], 0.8),
        (llm_top[4], 0.90, ["llm"], 0.8),
        (protected, 0.84, ["rule anchor"], 0.7),
    ]
    llm_ranker = FixedLLMRanker(llm_ranked)
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=llm_ranker,
        llm_anchor_guard={
            "enabled": True,
            "protect_rule_rank": 5,
            "max_score_gap": 0.08,
        },
    )

    result = pipeline.recommend(PaperInput(title="Test"), PaperProfile(title="Test"), top_k=5)

    assert llm_ranker.received_top_k == len(journals)
    assert "protected" in [rec.journal.journal_id for rec in result["recommendations"]]


def test_pipeline_does_not_restore_rule_anchor_when_llm_score_gap_is_large():
    """LLM 明显判低的靠前 Rule 候选不应被硬塞进 Top5。"""
    protected = Journal(journal_id="protected", journal_name="Protected Journal")
    llm_top = [
        Journal(journal_id=f"llm-{i}", journal_name=f"LLM Journal {i}")
        for i in range(5)
    ]
    journals = [protected] + llm_top
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [
        (journal, 1.0 - i * 0.05, [])
        for i, journal in enumerate(journals)
    ]
    llm_ranked = [
        (llm_top[0], 0.95, ["llm"], 0.8),
        (llm_top[1], 0.94, ["llm"], 0.8),
        (llm_top[2], 0.93, ["llm"], 0.8),
        (llm_top[3], 0.92, ["llm"], 0.8),
        (llm_top[4], 0.90, ["llm"], 0.8),
        (protected, 0.70, ["rule anchor"], 0.7),
    ]
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        llm_anchor_guard={
            "enabled": True,
            "protect_rule_rank": 5,
            "max_score_gap": 0.08,
        },
    )

    result = pipeline.recommend(PaperInput(title="Test"), PaperProfile(title="Test"), top_k=5)

    assert "protected" not in [rec.journal.journal_id for rec in result["recommendations"]]


# ---------------------------------------------------------------------------
# Task 5.3 — LTR 接入 + 默认 OFF bit-equal
# ---------------------------------------------------------------------------


class _StubLTRAdapter:
    """Test double for LTRAdapter,只实现 pipeline 用到的 enabled + compute_scores。"""

    def __init__(self, enabled: bool, compute_scores_fn=None, disable_reason=None):
        self._enabled = enabled
        self._compute_scores_fn = compute_scores_fn
        self._disable_reason = disable_reason

    @property
    def enabled(self):
        return self._enabled

    @property
    def disable_reason(self):
        return self._disable_reason

    def compute_scores(self, paper_profile, llm_candidates, retrieval_trace, rule_ranks, rule_scores):
        if self._compute_scores_fn is not None:
            return self._compute_scores_fn(llm_candidates)
        return list(llm_candidates), {
            "learned_score": {},
            "learned_rank": {},
            "status": "fallback_disabled",
        }


def test_pipeline_default_off_omits_ltr_fields():
    """5.3 强不变量:learned_reranker=None 时 result 字典**完全不写**新 key。

    baseline 5.2 的 result schema 必须 bit-equal,确保默认关闭时零回归。
    """
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(5)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, []) for i, journal in enumerate(journals)]
    llm_ranked = [(journals[i], 0.9 - i * 0.05, ["llm"], 0.8) for i in range(5)]

    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        # 关键:learned_reranker 不传,默认 None
    )
    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=5)

    assert result["rank_method"] == "llm"
    assert "learned_diagnostics" not in result
    assert "final_rank_source" not in result


def test_pipeline_with_ltr_reranks_llm_candidates():
    """LTR 启用且 compute_scores 成功时,result 写 learned_diagnostics + final_rank_source。"""
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(5)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, []) for i, journal in enumerate(journals)]
    llm_ranked = [(journals[i], 0.9 - i * 0.05, ["llm"], 0.8) for i in range(5)]

    def _stub_rerank(candidates):
        diag = {
            "learned_score": {c[0].journal_id: 0.5 + i * 0.1 for i, c in enumerate(candidates)},
            "learned_rank": {c[0].journal_id: i + 1 for i, c in enumerate(candidates)},
            "status": "ok",
        }
        return list(candidates), diag

    ltr = _StubLTRAdapter(enabled=True, compute_scores_fn=_stub_rerank)

    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        learned_reranker=ltr,
    )
    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=5)

    assert result["final_rank_source"] == "llm_after_learned_rerank"
    assert result["learned_diagnostics"]["status"] == "ok"
    # 5 本 learned_rank 都 1..5
    ranks = sorted(result["learned_diagnostics"]["learned_rank"].values())
    assert ranks == [1, 2, 3, 4, 5]


def test_pipeline_with_ltr_disabled_falls_back():
    """LTRAdapter.enabled=False → 跳过 LTR 路径,baseline 路径走原样。"""
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(5)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, []) for i, journal in enumerate(journals)]
    llm_ranked = [(journals[i], 0.9 - i * 0.05, ["llm"], 0.8) for i in range(5)]

    ltr = _StubLTRAdapter(enabled=False, disable_reason="disabled in config")

    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        learned_reranker=ltr,
    )
    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=5)

    assert result["rank_method"] == "llm"
    assert "learned_diagnostics" not in result
    assert "final_rank_source" not in result


def test_pipeline_with_ltr_missing_model_falls_back():
    """LTRAdapter.enabled=False 因为 model 缺失 → 走原 LLM 路径。"""
    from src.ranker.ltr_adapter import LTRAdapter

    ltr = LTRAdapter(
        config={"enabled": True, "model_path": "/nonexistent/learning_to_ranker.json"},
        journal_store=None,
    )
    assert ltr.enabled is False
    assert ltr.disable_reason is not None
    assert "not found" in ltr.disable_reason.lower()

    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(5)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, []) for i, journal in enumerate(journals)]
    llm_ranked = [(journals[i], 0.9 - i * 0.05, ["llm"], 0.8) for i in range(5)]

    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        learned_reranker=ltr,
    )
    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=5)

    assert result["rank_method"] == "llm"
    assert "learned_diagnostics" not in result
    assert "final_rank_source" not in result

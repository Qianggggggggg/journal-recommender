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
        self.attach_features_calls: list[dict] = []

    def generate_with_trace(
        self,
        query_text,
        paper_profile,
        top_k=40,
        mode="abstract",
        diagnostic_journal_ids=None,
    ):
        return self.candidates[:top_k], self.trace

    def attach_features(
        self,
        trace,
        paper_profile,
        rule_ranks,
        rule_scores,
        accepted_paper_store=None,
        feature_names=None,
        llm_evidence_by_journal=None,
    ):
        """6.4: 记录 kwargs。默认 no-op(不写入 features),所以 trace[jid]
        不会增加 ``features`` 字段。
        """
        self.attach_features_calls.append(
            {
                "feature_names": feature_names,
                "llm_evidence_by_journal": llm_evidence_by_journal,
            }
        )


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


class FailingLLMRanker:
    def rank(self, candidates, paper_profile, top_k=5, retrieval_trace=None):
        from src.ranker.llm_ranker import LLMRankerError

        raise LLMRankerError("rankings 为空")


class UnexpectedFailingLLMRanker:
    def rank(self, candidates, paper_profile, top_k=5, retrieval_trace=None):
        raise TypeError("unexpected score type")


class DiagnosticLLMRanker:
    def __init__(self):
        self.received_learned_ranks = None
        self.received_learned_scores = None

    def rank_with_diagnostics(
        self,
        candidates,
        paper_profile,
        top_k=5,
        retrieval_trace=None,
        rule_ranks=None,
        rule_scores=None,
        learned_ranks=None,
        learned_scores=None,
    ):
        self.received_learned_ranks = learned_ranks
        self.received_learned_scores = learned_scores
        ranked = [
            (journal, 0.9 - index * 0.1, ["evidence"], 0.8)
            for index, (journal, _score, _reasons) in enumerate(candidates)
        ]
        return ranked[:top_k], "llm_evidence_rule", {
            "status": "ok",
            "prior_source": "rule",
            "candidates": {
                journal.journal_id: {"final_rank": index + 1}
                for index, (journal, _score, _reasons) in enumerate(candidates)
            },
        }


def test_pipeline_passes_per_call_evidence_diagnostics_without_changing_direct_ranker_path():
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(3)]
    trace = {
        journal.journal_id: {"retrieval_rank": index + 1, "routes": {}}
        for index, journal in enumerate(journals)
    }
    ranked = [(journal, 1.0 - index * 0.1, []) for index, journal in enumerate(journals)]
    diagnostic_ranker = DiagnosticLLMRanker()
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=diagnostic_ranker,
        llm_anchor_guard={"enabled": False},
    )

    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=2)

    assert result["rank_method"] == "llm_evidence_rule"
    assert result["llm_role_diagnostics"]["status"] == "ok"
    assert result["llm_role_diagnostics"]["candidates"]["j1"]["final_rank"] == 2
    assert diagnostic_ranker.received_learned_ranks == {}


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


def test_pipeline_falls_back_to_rule_top_k_when_llm_ranking_fails():
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(8)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, [f"rule-{i}"]) for i, journal in enumerate(journals)]
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FailingLLMRanker(),
    )

    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=5)

    assert result["rank_method"] == "rule_fallback"
    assert result["fallback_used"] is True
    assert result["fallback_stage"] == "llm_ranking"
    assert "rankings 为空" in result["fallback_reason"]
    assert [rec.journal.journal_id for rec in result["recommendations"]] == [
        "j0",
        "j1",
        "j2",
        "j3",
        "j4",
    ]
    assert [rec.match_reasons for rec in result["recommendations"]] == [
        ["rule-0"],
        ["rule-1"],
        ["rule-2"],
        ["rule-3"],
        ["rule-4"],
    ]


def test_pipeline_falls_back_when_llm_stage_raises_unexpected_error():
    journal = Journal(journal_id="j0", journal_name="J0")
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator([journal], {"j0": {"routes": {}}}),
        rule_scorer=FixedRuleScorer([(journal, 1.0, ["rule"])]),
        llm_ranker=UnexpectedFailingLLMRanker(),
    )

    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=1)

    assert result["rank_method"] == "rule_fallback"
    assert result["recommendations"][0].journal.journal_id == "j0"
    assert "unexpected score type" in result["fallback_reason"]


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


def test_pipeline_passes_learned_ranks_to_diagnostic_ranker():
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(3)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, []) for i, journal in enumerate(journals)]
    learned_ranks = {"j0": 3, "j1": 1, "j2": 2}

    def _stub_rerank(candidates):
        return list(candidates), {
            "learned_score": {jid: 0.5 for jid in learned_ranks},
            "learned_rank": learned_ranks,
            "status": "ok",
        }

    diagnostic_ranker = DiagnosticLLMRanker()
    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=diagnostic_ranker,
        learned_reranker=_StubLTRAdapter(enabled=True, compute_scores_fn=_stub_rerank),
        llm_anchor_guard={"enabled": False},
    )

    pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=2)

    assert diagnostic_ranker.received_learned_ranks == learned_ranks
    # 6.4: learned_scores must flow through the pipeline to the role ranker
    # so LLMEvidenceRoleRanker can use it as a 3rd formula component.
    assert diagnostic_ranker.received_learned_scores == {
        jid: 0.5 for jid in learned_ranks
    }


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


# ---------------------------------------------------------------------------
# Task 6.4 — Pipeline threads evidence_lookup to 22-dim attach_features
# ---------------------------------------------------------------------------


class _AttachingCandidateGenerator(DummyGenerator):
    """DummyGenerator that records attach_features() kwargs and writes requested features.

    Mimics the real CandidateGenerator.attach_features contract: it builds a
    feature vector of length ``len(feature_names)`` per journal id.
    """

    def __init__(self, candidates, trace, store):
        super().__init__(candidates, trace)
        self.store = store
        self.attach_features_calls: list[dict] = []

    def attach_features(
        self,
        trace,
        paper_profile,
        rule_ranks,
        rule_scores,
        accepted_paper_store=None,
        feature_names=None,
        llm_evidence_by_journal=None,
    ):
        self.attach_features_calls.append(
            {
                "feature_names": feature_names,
                "llm_evidence_by_journal": llm_evidence_by_journal,
            }
        )
        # Mimic the real attach_features: write a features list of the requested dim.
        from src.ranker.feature_builder import FEATURE_NAMES, FEATURE_NAMES_WITH_LLM_EVIDENCE

        dim = len(feature_names) if feature_names is not None else len(FEATURE_NAMES)
        for jid, entry in trace.items():
            evidence = (llm_evidence_by_journal or {}).get(jid) or {}
            evidence_values = [
                evidence.get("scope_fit", 0.5),
                evidence.get("method_fit", 0.5),
                evidence.get("application_fit", 0.5),
                evidence.get("journal_position_fit", 0.5),
                evidence.get("too_broad_penalty", 0.0),
                evidence.get("too_narrow_penalty", 0.0),
            ]
            # Base 16-dim vector is all zeros; pad with evidence values to hit dim.
            base = [0.0] * len(FEATURE_NAMES)
            full = base + evidence_values
            entry["features"] = full[:dim]
            entry["feature_names"] = (
                list(feature_names)
                if feature_names is not None
                else list(FEATURE_NAMES)
            )


def test_pipeline_attaches_22_dim_features_when_evidence_supplied():
    """Evidence lookup selects the 22-dimensional feature schema."""
    from src.journals.journal_model import Journal
    from src.journals.journal_store import JournalStore

    journal_a = Journal(journal_id="j1", journal_name="J1")
    journal_b = Journal(journal_id="j2", journal_name="J2")
    store = JournalStore()
    store.add_journal(journal_a)
    store.add_journal(journal_b)

    trace = {
        "j1": {"retrieval_rank": 1, "routes": {}},
        "j2": {"retrieval_rank": 2, "routes": {}},
    }
    ranked = [(journal_a, 1.0, []), (journal_b, 0.8, [])]
    snapshot = {
        "test paper": {
            "evidence": {
                "j1": {
                    "scope_fit": 0.9,
                    "method_fit": 0.8,
                    "application_fit": 0.7,
                    "journal_position_fit": 0.85,
                    "too_broad_penalty": 0.1,
                    "too_narrow_penalty": 0.05,
                },
                "j2": {
                    "scope_fit": 0.4,
                    "method_fit": 0.3,
                    "application_fit": 0.5,
                    "journal_position_fit": 0.2,
                    "too_broad_penalty": 0.0,
                    "too_narrow_penalty": 0.0,
                },
            },
        }
    }

    llm_ranked = [
        (journal_a, 0.9, ["llm"], 0.8),
        (journal_b, 0.8, ["llm"], 0.8),
    ]
    gen = _AttachingCandidateGenerator([journal_a, journal_b], trace, store)
    pipeline = RecommenderPipeline(
        candidate_generator=gen,
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        evidence_lookup=snapshot,
        feature_schema="22_dim_with_llm_evidence",
    )

    result = pipeline.recommend(
        PaperInput(title="Test Paper", abstract="...", mode="abstract"),
        PaperProfile(title="Test Paper"),
        top_k=5,
        mode="abstract",
    )

    # Pipeline called attach_features once, with the 22-dim schema and the
    # per-paper evidence dict.
    assert len(gen.attach_features_calls) == 1
    call = gen.attach_features_calls[0]
    assert call["feature_names"] is not None
    assert len(call["feature_names"]) == 22
    assert call["llm_evidence_by_journal"] == snapshot["test paper"]["evidence"]

    # Per-journal features are now 22 long.
    retrieval_trace = result["retrieval_trace"]
    for jid, entry in retrieval_trace.items():
        if "features" in entry:
            assert len(entry["features"]) == 22
    assert len(retrieval_trace["j1"]["features"]) == 22
    assert len(retrieval_trace["j2"]["features"]) == 22


def test_pipeline_omits_evidence_schema_when_paper_not_in_snapshot():
    """When the paper title isn't in evidence_lookup, pipeline falls back to
    16-dim base schema when no matching evidence exists."""
    from src.journals.journal_model import Journal
    from src.journals.journal_store import JournalStore

    journal_a = Journal(journal_id="j1", journal_name="J1")
    store = JournalStore()
    store.add_journal(journal_a)

    trace = {"j1": {"retrieval_rank": 1, "routes": {}}}
    ranked = [(journal_a, 1.0, [])]
    llm_ranked = [(journal_a, 0.9, ["llm"], 0.8)]
    gen = _AttachingCandidateGenerator([journal_a], trace, store)
    pipeline = RecommenderPipeline(
        candidate_generator=gen,
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
        evidence_lookup={},  # paper not in snapshot
        feature_schema="22_dim_with_llm_evidence",
    )

    result = pipeline.recommend(
        PaperInput(title="Unseen Paper", abstract="...", mode="abstract"),
        PaperProfile(title="Unseen Paper"),
        top_k=5,
        mode="abstract",
    )

    assert len(gen.attach_features_calls) == 1
    call = gen.attach_features_calls[0]
    # feature_names is None → 16-dim default
    assert call["feature_names"] is None
    # features still attached (16-dim)
    assert len(result["retrieval_trace"]["j1"]["features"]) == 16


def test_pipeline_surfaces_learned_score_in_result_when_ltr_enabled():
    """When LTR is enabled, result dict exposes learned_score / learned_rank."""
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

    # learned_score/learned_rank surfaced in result
    assert "learned_score" in result
    assert "learned_rank" in result
    assert set(result["learned_score"].keys()) == {f"j{i}" for i in range(5)}
    assert sorted(result["learned_rank"].values()) == [1, 2, 3, 4, 5]


def test_pipeline_omits_learned_score_when_ltr_off():
    """Backward compat: when learned_reranker is None, learned_score key is absent."""
    journals = [Journal(journal_id=f"j{i}", journal_name=f"J{i}") for i in range(3)]
    trace = {journal.journal_id: {"routes": {}} for journal in journals}
    ranked = [(journal, 1.0 - i * 0.05, []) for i, journal in enumerate(journals)]
    llm_ranked = [(journals[i], 0.9 - i * 0.05, ["llm"], 0.8) for i in range(3)]

    pipeline = RecommenderPipeline(
        candidate_generator=DummyGenerator(journals, trace),
        rule_scorer=FixedRuleScorer(ranked),
        llm_ranker=FixedLLMRanker(llm_ranked),
    )
    result = pipeline.recommend(PaperInput(title="T"), PaperProfile(title="T"), top_k=3)

    assert "learned_score" not in result
    assert "learned_rank" not in result

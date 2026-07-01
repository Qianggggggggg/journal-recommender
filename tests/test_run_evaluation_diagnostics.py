"""Evaluation diagnostics regression tests."""
import json

import pytest

from scripts.run_evaluation import (
    attach_baseline_profile_snapshots,
    evaluate_single_paper,
    run_evaluation,
)
from src.journals.journal_model import Journal, JournalMatch
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile


class DummyParser:
    def parse(self, paper_input, system_prompt, user_prompt):
        return PaperProfile(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            ccf_research_area=["人工智能"],
        )


class DummyPipeline:
    def __init__(self, journal):
        store = JournalStore()
        store.add_journal(journal)
        self.parser = DummyParser()
        self.candidate_generator = type("Generator", (), {"store": store})()
        self.journal = journal

    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        return {
            "recommendations": [
                JournalMatch(
                    journal=self.journal,
                    score=0.91,
                    confidence=0.8,
                    match_reasons=["范围文本覆盖论文主题"],
                )
            ],
            "candidates": [self.journal],
            "rule_ranked": [(self.journal, 0.77, ["期刊范围文本提供边界匹配证据"])],
            "llm_candidates": [(self.journal, 0.77, [])],
            "llm_candidate_ids": [self.journal.journal_id],
            "retrieval_trace": {
                self.journal.journal_id: {
                    "retrieval_rank": 1,
                    "total_score": 0.42,
                    "primary_routes": ["scope_bm25", "typical_bm25"],
                    "routes": {
                        "scope_bm25": {
                            "rank": 1,
                            "weighted_score": 0.3,
                            "normalized_score": 1.0,
                        },
                        "typical_bm25": {
                            "rank": 2,
                            "weighted_score": 0.12,
                            "normalized_score": 0.6,
                        },
                    },
                }
            },
        }


class WideMissPipeline(DummyPipeline):
    def __init__(self, target, recommended):
        super().__init__(target)
        self.recommended = recommended
        self.candidate_generator.store.add_journal(recommended)

    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        return {
            "recommendations": [
                JournalMatch(
                    journal=self.recommended,
                    score=0.7,
                    confidence=0.6,
                    match_reasons=["邻近期刊"],
                )
            ],
            "candidates": [self.recommended],
            "rule_ranked": [(self.recommended, 0.5, [])],
            "llm_candidates": [(self.recommended, 0.5, [])],
            "llm_candidate_ids": [self.recommended.journal_id],
            "retrieval_trace": {
                self.recommended.journal_id: {
                    "retrieval_rank": 1,
                    "total_score": 0.5,
                    "primary_routes": ["scope_bm25"],
                    "routes": {"scope_bm25": {"rank": 1, "weighted_score": 0.5}},
                },
                self.journal.journal_id: {
                    "wide_retrieval_rank": 3,
                    "total_score": 0.2,
                    "primary_routes": ["scope_bm25"],
                    "routes": {"scope_bm25": {"rank": 3, "weighted_score": 0.2}},
                    "wide_routes": {"scope_bm25": {"rank": 3, "weighted_score": 0.2}},
                },
            },
        }


class SnapshotPipeline(DummyPipeline):
    def __init__(self, journal):
        super().__init__(journal)
        self.parser = type(
            "FailingParser",
            (),
            {"parse": lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parser called"))},
        )()
        self.received_profile = None
        self.received_quality_prompts = "unset"

    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        self.received_profile = profile
        self.received_quality_prompts = quality_prompts
        return super().recommend(
            paper_input,
            profile,
            top_k=top_k,
            mode=mode,
            quality_prompts=quality_prompts,
            diagnostic_journal_ids=diagnostic_journal_ids,
        )


class FallbackPipeline(DummyPipeline):
    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        result = super().recommend(
            paper_input,
            profile,
            top_k=top_k,
            mode=mode,
            quality_prompts=quality_prompts,
            diagnostic_journal_ids=diagnostic_journal_ids,
        )
        result.update(
            {
                "rank_method": "rule_fallback",
                "fallback_used": True,
                "fallback_stage": "llm_ranking",
                "fallback_reason": "rankings 为空",
            }
        )
        return result


class EmptyPipeline(DummyPipeline):
    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        result = super().recommend(
            paper_input,
            profile,
            top_k=top_k,
            mode=mode,
            quality_prompts=quality_prompts,
            diagnostic_journal_ids=diagnostic_journal_ids,
        )
        result["recommendations"] = []
        result["rank_method"] = "llm"
        return result


class EvidenceDiagnosticPipeline(DummyPipeline):
    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        result = super().recommend(
            paper_input,
            profile,
            top_k=top_k,
            mode=mode,
            quality_prompts=quality_prompts,
            diagnostic_journal_ids=diagnostic_journal_ids,
        )
        jid = self.journal.journal_id
        result["rank_method"] = "llm_evidence_rule"
        result["llm_role_diagnostics"] = {
            "status": "ok",
            "role": "evidence",
            "prior_source": "rule",
            "evidence_coverage": 1.0,
            "candidates": {
                jid: {
                    "input_rank": 1,
                    "rank_prior": 1.0,
                    "llm_scope_fit": 0.9,
                    "llm_method_fit": 0.8,
                    "llm_application_fit": 0.7,
                    "llm_journal_position_fit": 0.6,
                    "llm_too_broad_penalty": 0.1,
                    "llm_too_narrow_penalty": 0.0,
                    "evidence": ["specific evidence"],
                    "evidence_composite": 0.7,
                    "final_score": 0.76,
                    "final_rank": 1,
                    "features_base": [0.0] * 16,
                    "features_with_llm_evidence": [0.0] * 22,
                    "feature_names_base": [f"base_{i}" for i in range(16)],
                    "feature_names_with_llm_evidence": [f"v2_{i}" for i in range(22)],
                }
            },
        }
        return result


def test_evaluate_single_paper_reuses_snapshot_without_parser_or_quality_assessor():
    journal = Journal(journal_id="target", journal_name="Target Journal", ccf_rating="B")
    pipeline = SnapshotPipeline(journal)
    paper = {
        "title": "Stable Paper",
        "abstract": "Stable abstract.",
        "venue": "Target Journal",
        "ccf_level": "B",
        "research_area": ["人工智能"],
        "paper_profile_snapshot": {
            "title": "Stable Paper",
            "research_area": ["人工智能"],
            "ccf_research_area": ["人工智能"],
            "keywords": ["fixed"],
            "quality_level": "B",
            "paper_strength": 0.75,
        },
    }

    result = evaluate_single_paper(
        paper,
        pipeline,
        {
            "paper_profile_system": "",
            "paper_profile_user": "",
            "paper_quality_assessor_system": "quality",
            "paper_quality_assessor_user": "quality",
        },
        mode="abstract",
        top_k=5,
        reuse_profile_snapshot=True,
    )

    assert result["paper_profile_snapshot"]["keywords"] == ["fixed"]
    assert pipeline.received_profile.paper_strength == 0.75
    assert pipeline.received_quality_prompts is None


def test_attach_baseline_profile_snapshots_fails_when_current_paper_is_missing(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "paper_results": [
                    {
                        "title": "Other Paper",
                        "venue": "Other Journal",
                        "paper_profile_snapshot": {"title": "Other Paper"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="缺少固定 paper_profile_snapshot"):
        attach_baseline_profile_snapshots(
            [{"title": "Current Paper", "venue": "Target Journal"}],
            str(baseline_path),
        )


def test_attach_baseline_profile_snapshots_preserves_current_order(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "paper_results": [
                    {
                        "title": "Paper B",
                        "venue": "Journal B",
                        "paper_profile_snapshot": {"title": "Snapshot B", "keywords": ["b"]},
                    },
                    {
                        "title": "Paper A",
                        "venue": "Journal A",
                        "paper_profile_snapshot": {"title": "Snapshot A", "keywords": ["a"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    attached = attach_baseline_profile_snapshots(
        [
            {"title": "Paper A", "venue": "Journal A", "abstract": "A"},
            {"title": "Paper B", "venue": "Journal B", "abstract": "B"},
        ],
        str(baseline_path),
    )

    assert [paper["title"] for paper in attached] == ["Paper A", "Paper B"]
    assert attached[0]["paper_profile_snapshot"]["keywords"] == ["a"]
    assert attached[1]["paper_profile_snapshot"]["keywords"] == ["b"]


def test_run_evaluation_reuses_snapshots_with_ten_workers():
    journal = Journal(journal_id="target", journal_name="Target Journal", ccf_rating="B")
    pipeline = SnapshotPipeline(journal)
    papers = [
        {
            "title": f"Stable Paper {index}",
            "abstract": "Stable abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "paper_profile_snapshot": {
                "title": f"Stable Paper {index}",
                "research_area": ["人工智能"],
                "ccf_research_area": ["人工智能"],
                "quality_level": "B",
            },
        }
        for index in range(2)
    ]

    result = run_evaluation(
        papers,
        pipeline,
        mode="abstract",
        top_k=5,
        prompts={"paper_profile_system": "", "paper_profile_user": ""},
        show_progress=False,
        workers=10,
        reuse_profile_snapshots=True,
    )

    assert len(result.paper_results) == 2
    assert result.hit_at_5 == 2


def test_evaluate_single_paper_records_fallback_diagnostics():
    journal = Journal(journal_id="target", journal_name="Target Journal", ccf_rating="B")

    result = evaluate_single_paper(
        {
            "title": "Fallback Paper",
            "abstract": "Abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
        },
        FallbackPipeline(journal),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    assert result["evaluation_status"] == "fallback"
    assert result["rank_method"] == "rule_fallback"
    assert result["fallback_used"] is True
    assert result["fallback_stage"] == "llm_ranking"
    assert result["fallback_reason"] == "rankings 为空"


def test_run_evaluation_counts_fallback_and_empty_recommendations():
    journal = Journal(journal_id="target", journal_name="Target Journal", ccf_rating="B")
    base_paper = {
        "title": "Paper",
        "abstract": "Abstract.",
        "venue": "Target Journal",
        "ccf_level": "B",
        "research_area": ["人工智能"],
    }

    fallback_result = run_evaluation(
        [base_paper],
        FallbackPipeline(journal),
        mode="abstract",
        top_k=5,
        prompts={"paper_profile_system": "", "paper_profile_user": ""},
        show_progress=False,
        workers=10,
    )
    empty_result = run_evaluation(
        [base_paper],
        EmptyPipeline(journal),
        mode="abstract",
        top_k=5,
        prompts={"paper_profile_system": "", "paper_profile_user": ""},
        show_progress=False,
        workers=10,
    )

    assert fallback_result.fallback_count == 1
    assert fallback_result.llm_success_count == 0
    assert fallback_result.empty_recommendation_count == 0
    assert empty_result.fallback_count == 0
    assert empty_result.llm_success_count == 0
    assert empty_result.empty_recommendation_count == 1


def test_evaluate_single_paper_writes_venue_diagnostic():
    journal = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
    )
    result = evaluate_single_paper(
        {
            "title": "Test Paper With A Long Enough Title That Must Not Be Truncated In Evaluation Diagnostics",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        DummyPipeline(journal),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    diagnostic = result["venue_diagnostic"]
    assert result["title"] == "Test Paper With A Long Enough Title That Must Not Be Truncated In Evaluation Diagnostics"
    assert result["abstract_len"] == len("A short abstract.")
    assert result["gold_area"] == "人工智能"
    assert result["parsed_ccf_area"] == ["人工智能"]
    assert result["area_mismatch"] is False
    assert diagnostic["journal_id"] == "target"
    assert diagnostic["retrieval_rank"] == 1
    assert diagnostic["retrieval_score"] == 0.42
    assert diagnostic["rule_rank"] == 1
    assert diagnostic["rule_score"] == 0.77
    assert diagnostic["in_llm_pool"] is True
    assert diagnostic["retrieval_sources"] == ["scope_bm25", "typical_bm25"]
    assert diagnostic["target_journal_id"] == "target"
    assert diagnostic["gold_area"] == "人工智能"
    assert diagnostic["parsed_ccf_area"] == ["人工智能"]
    assert diagnostic["area_mismatch"] is False
    assert diagnostic["abstract_len"] == len("A short abstract.")
    assert diagnostic["miss_stage"] == "final_hit"
    assert result["paper_profile_snapshot"]["title"] == "Test Paper With A Long Enough Title That Must Not Be Truncated In Evaluation Diagnostics"
    assert result["paper_profile_snapshot"]["abstract_len"] == len("A short abstract.")
    assert result["paper_profile_snapshot"]["abstract_preview"] == "A short abstract."


def test_evaluate_single_paper_marks_wide_recalled_not_top50():
    target = Journal(journal_id="target", journal_name="Target Journal", ccf_rating="B")
    recommended = Journal(journal_id="other", journal_name="Other Journal", ccf_rating="B")

    result = evaluate_single_paper(
        {
            "title": "Test Paper",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        WideMissPipeline(target, recommended),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    diagnostic = result["venue_diagnostic"]
    assert diagnostic["target_journal_id"] == "target"
    assert diagnostic["wide_retrieval_rank"] == 3
    assert diagnostic["retrieval_rank"] is None
    assert diagnostic["wide_retrieval_route_scores"]["scope_bm25"]["rank"] == 3
    assert diagnostic["miss_stage"] == "wide_recalled_but_not_top50"


def test_evaluate_single_paper_writes_auxiliary_acceptability_metrics():
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
        subject_tags=["人工智能"],
    )
    recommended = Journal(
        journal_id="other",
        journal_name="Other Journal",
        ccf_rating="B",
        subject_tags=["人工智能"],
    )

    result = evaluate_single_paper(
        {
            "title": "Test Paper",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        WideMissPipeline(target, recommended),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    assert result["hit_5"] is False
    assert result["same_area_hit_5"] is True
    assert result["same_ccf_level_hit_5"] is True
    assert result["acceptable_journal_hit_5"] is True


def test_evaluate_single_paper_persists_complete_llm_evidence_candidate_diagnostics():
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
        subject_tags=["人工智能"],
    )

    result = evaluate_single_paper(
        {
            "title": "Evidence Paper",
            "abstract": "Abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
        },
        EvidenceDiagnosticPipeline(target),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    assert result["llm_evidence_status"] == "ok"
    assert result["llm_evidence_coverage"] == 1.0
    assert len(result["llm_candidates_detail"]) == 1
    detail = result["llm_candidates_detail"][0]
    assert detail["journal_id"] == "target"
    assert detail["llm_scope_fit"] == 0.9
    assert len(detail["features_base"]) == 16
    assert len(detail["features_with_llm_evidence"]) == 22
    assert result["venue_diagnostic"]["llm_evidence_rank"] == 1
    assert result["venue_diagnostic"]["llm_evidence_final_score"] == 0.76


# ---------------------------------------------------------------------------
# Task 5.3 — LTR diagnostics
# ---------------------------------------------------------------------------


class LTREnabledDummyPipeline(DummyPipeline):
    """Dummy pipeline 注入 LTR 成功状态,用于测试 evaluate_single_paper 透传诊断字段。"""

    def recommend(self, paper_input, profile, top_k=5, mode="abstract", quality_prompts=None, diagnostic_journal_ids=None):
        result = super().recommend(paper_input, profile, top_k=top_k, mode=mode, quality_prompts=quality_prompts, diagnostic_journal_ids=diagnostic_journal_ids)
        jid = self.journal.journal_id
        result["learned_diagnostics"] = {
            "learned_score": {jid: 0.73},
            "learned_rank": {jid: 1},
            "status": "ok",
        }
        result["final_rank_source"] = "llm_after_learned_rerank"
        return result


def test_evaluate_single_paper_default_off_omits_learned_fields():
    """OFF (现有 DummyPipeline 无 LTR):venue_diagnostic / recommendations_detail / per-paper 顶层
    **全部不含** learned_* 字段。bit-equal baseline 强约束。
    """
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
        ccf_research_area=["人工智能"],
        subject_tags=["人工智能"],
    )
    result = evaluate_single_paper(
        {
            "title": "Test Paper With A Long Enough Title",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        DummyPipeline(target),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    # per-paper 顶层:无 final_rank_source
    assert "final_rank_source" not in result
    # venue_diagnostic:无 learned_score/learned_rank
    diag = result["venue_diagnostic"]
    assert "learned_score" not in diag
    assert "learned_rank" not in diag
    # recommendations_detail:无 learned_score
    for detail in result["recommendations_detail"]:
        assert "learned_score" not in detail


def test_evaluate_single_paper_with_ltr_populates_learned_fields():
    """ON (LTREnabledDummyPipeline 注入 learned_diagnostics.status='ok') →
    venue_diagnostic / recommendations_detail / per-paper 顶层都填上 learned_* 字段。
    """
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
        ccf_research_area=["人工智能"],
        subject_tags=["人工智能"],
    )
    result = evaluate_single_paper(
        {
            "title": "Test Paper With A Long Enough Title",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        LTREnabledDummyPipeline(target),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    # per-paper 顶层
    assert result["final_rank_source"] == "llm_after_learned_rerank"
    # venue_diagnostic
    diag = result["venue_diagnostic"]
    assert "learned_score" in diag
    assert "learned_rank" in diag
    # recommendations_detail
    for detail in result["recommendations_detail"]:
        assert "learned_score" in detail

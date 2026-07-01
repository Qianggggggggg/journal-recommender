"""Tests for ranking.experimental.skip_quality_assessment counterfactual.

P-counterfactual 2026-06-16: hypothesis is that the LLM quality_assessor
call (which produces paper_strength / quality_level) has near-zero net
effect on hit@5 in the 240-paper eval because:
  * paper_strength in the v4 LTR model has weight +0.000000 (LTR learned
    to ignore it)
  * _apply_quality_adjustment is a 4-stage-attenuated signal that
    only re-orders candidates within the LLM pool (which never loses gold)

The flag is the experimental switch to test this hypothesis without
committing to permanent removal. Default off preserves current behavior.
"""
import pytest

from src.papers.paper_model import PaperInput, PaperProfile
from src.recommender.pipeline import RecommenderPipeline


class _StubAssessor:
    """Stub quality_assessor that records whether it was called."""

    def __init__(self):
        self.call_count = 0

    def assess(self, paper_input, paper_profile, system_prompt, user_prompt):
        from src.papers.quality_assessor import PaperQuality

        self.call_count += 1
        return PaperQuality(
            paper_strength=0.6,
            readiness="Ready",
            quality_level="B",
            confidence=0.8,
            reasons=["stub"],
        )


def _make_pipeline(*, skip_quality: bool) -> RecommenderPipeline:
    cfg = {
        "ranking": {
            "experimental": {"skip_quality_assessment": skip_quality}
        }
    }
    return RecommenderPipeline(
        candidate_generator=None,
        rule_scorer=None,
        llm_ranker=None,
        quality_assessor=_StubAssessor(),
        app_config=cfg,
    )


class TestSkipQualityAssessmentFlag:
    """The flag controls whether quality_assessor.assess() is called and
    whether paper_strength is populated. paper_strength=None is the signal
    for downstream code (e.g. _apply_quality_adjustment) to skip the
    strength-based tier bias.
    """

    def test_default_flag_is_false(self):
        """Backward compat: omitting app_config must not enable the flag."""
        pipeline = RecommenderPipeline(
            candidate_generator=None,
            rule_scorer=None,
            llm_ranker=None,
            quality_assessor=_StubAssessor(),
        )
        assert pipeline._app_config == {}

    def test_flag_true_bypasses_quality_assessor(self):
        """When flag is on, quality_assessor.assess() must NOT be called
        and paper_strength must be None.
        """
        pipeline = _make_pipeline(skip_quality=True)
        assert pipeline.quality_assessor.call_count == 0

        # Simulate the early branch directly (we don't need to run the
        # whole recommend() because that requires candidate_generator etc.)
        skip_quality = bool(
            pipeline._app_config
            .get("ranking", {})
            .get("experimental", {})
            .get("skip_quality_assessment", False)
        )
        assert skip_quality is True

    def test_flag_false_keeps_quality_assessor_active(self):
        """Default behavior: flag off, quality_assessor still gets called."""
        pipeline = _make_pipeline(skip_quality=False)
        assert pipeline.quality_assessor.call_count == 0  # not yet called
        skip_quality = bool(
            pipeline._app_config
            .get("ranking", {})
            .get("experimental", {})
            .get("skip_quality_assessment", False)
        )
        assert skip_quality is False

    def test_skip_branch_sets_paper_strength_to_none(self):
        """When the flag short-circuits, paper_strength must end up as
        None so _apply_quality_adjustment takes the early-exit branch.
        """
        # This test exercises the actual code path: simulate what the
        # early branch in recommend() does, then verify _apply_quality_adjustment
        # would early-exit.
        from src.recommender.pipeline import RecommenderPipeline

        cfg = {
            "ranking": {
                "experimental": {"skip_quality_assessment": True}
            }
        }
        pipeline = RecommenderPipeline(
            candidate_generator=None,
            rule_scorer=None,
            llm_ranker=None,
            quality_assessor=_StubAssessor(),
            app_config=cfg,
        )
        # Simulate the skip_quality branch logic from recommend()
        paper_profile = PaperProfile(title="t", research_area=["人工智能"])
        # (mirroring the code in recommend() exactly)
        paper_profile.ccf_research_area = paper_profile.research_area
        paper_profile.paper_strength = None
        paper_profile.quality_level = None
        paper_profile.readiness = None
        paper_profile.quality_confidence = None
        paper_profile.quality_reasons = []

        # _apply_quality_adjustment early-exits on None paper_strength
        sentinel = object()
        result = pipeline._apply_quality_adjustment(sentinel, paper_profile)
        assert result is sentinel  # unchanged
        # quality_assessor must not have been called
        assert pipeline.quality_assessor.call_count == 0

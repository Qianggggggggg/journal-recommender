"""Regression test for PaperProfile.quality_confidence accepting None.

Bug 2026-06-16: skip_quality_assessment=true pipeline sets
paper_profile.quality_confidence = None (pipeline.py:118). When
run_evaluation.py later tries to reconstruct PaperProfile from
the saved paper_profile_snapshot via PaperProfile(**values),
pydantic raised ValidationError because the field was typed as
required `float`.

Fix: PaperProfile.quality_confidence must be Optional[float] like
its siblings (paper_strength, quality_level, readiness).

This test pins the contract: PaperProfile accepts None for
quality_confidence so paper_profile_from_snapshot can rebuild it.
"""
import pytest
from pydantic import ValidationError

from src.papers.paper_model import PaperProfile


class TestQualityConfidenceOptional:
    """PaperProfile.quality_confidence must accept None."""

    def test_none_is_accepted(self):
        """skip_quality_assessment branch sets confidence=None;
        PaperProfile must accept it without raising."""
        profile = PaperProfile(title="t", quality_confidence=None)
        assert profile.quality_confidence is None

    def test_default_is_none_not_zero(self):
        """After the fix, default must be None (not 0.0).
        0.0 is a meaningful 'low confidence' value; None is 'not assessed'.
        Downstream (pdf_exporter.py:112) already treats None distinctly.
        """
        profile = PaperProfile(title="t")
        assert profile.quality_confidence is None

    def test_explicit_zero_still_works(self):
        """Callers that pass 0.0 explicitly must still work."""
        profile = PaperProfile(title="t", quality_confidence=0.0)
        assert profile.quality_confidence == 0.0

    def test_valid_float_still_works(self):
        """Normal 0.85-style confidence must round-trip."""
        profile = PaperProfile(title="t", quality_confidence=0.85)
        assert profile.quality_confidence == 0.85

    def test_from_dict_round_trip(self):
        """Reproduce the bug scenario: PaperProfile(**dict_with_none)."""
        # This mirrors paper_profile_from_snapshot's PaperProfile(**values) call
        values = {
            "title": "t",
            "abstract": "",
            "research_area": ["人工智能"],
            "paper_strength": None,
            "quality_level": None,
            "readiness": None,
            "quality_confidence": None,  # the offending field
            "quality_reasons": [],
        }
        # Before the fix: ValidationError on quality_confidence
        # After the fix: succeeds
        profile = PaperProfile(**values)
        assert profile.quality_confidence is None
        assert profile.paper_strength is None


class TestQualityConfidenceInvalidTypes:
    """Negative tests: structurally non-numeric inputs still rejected.
    (Note: pydantic v2 coerces str/bool to float, so we only test
    types it cannot coerce — list/dict.)
    """

    @pytest.mark.parametrize("bad", [[0.8], {"v": 0.8}, object()])
    def test_non_numeric_rejected(self, bad):
        with pytest.raises(ValidationError):
            PaperProfile(title="t", quality_confidence=bad)

"""Tests for the strength-aware CCF-tier quality adjustment multiplier.

P1 (2026-06-16 diagnostic): the previous implementation applied
``{"A": 1.05, "B": 1.02, "C": 1.0}`` uniformly regardless of paper
strength. On holdout240 this systematically suppressed C-tier
application-oriented journals (HCI, security engineering, education) and
cost 20-30pp of hit@5 vs A-tier. The new function flips the bias for
weak papers (strength<0.5) so the multiplier prefers C-tier journals
when the paper itself is not strong enough for top venues.
"""
import pytest

from src.utils.text import (
    quality_adjustment_factor,
    quality_adjustment_multiplier,
)


class TestQualityAdjustmentMultiplier:
    """Lock the (strength, ccf_rating) -> multiplier calibration table.

    The intent of P1 is that:
      * Strong paper (strength >= 0.5) -> A > B > C (preference for top venues)
      * Weak paper   (strength <  0.5) -> C > B > A (preference for matching venue tier)
      * Boundary strength == 0.5     -> A > B > C (uses the strong-paper table; 0.5
        is the neutral point of quality_adjustment_factor so no extra boost/reduce
        is needed; tier preference is the only signal)
      * Unknown / missing ccf_rating -> 1.0 (no bias)
    """

    # ---- Strong paper: A > B > C (preserves current behavior) ----

    @pytest.mark.parametrize("strength", [0.5, 0.6, 0.7, 0.85, 1.0])
    def test_strong_paper_prefers_a_tier(self, strength):
        a = quality_adjustment_multiplier(strength, "A")
        b = quality_adjustment_multiplier(strength, "B")
        c = quality_adjustment_multiplier(strength, "C")
        assert a > b > c, (
            f"strength={strength}: expected A > B > C, got A={a} B={b} C={c}"
        )
        assert a == pytest.approx(1.05)
        assert b == pytest.approx(1.02)
        assert c == pytest.approx(1.0)

    # ---- Weak paper: C > B > A (counter-bias; this is the new behavior) ----

    @pytest.mark.parametrize("strength", [0.0, 0.2, 0.35, 0.49])
    def test_weak_paper_prefers_c_tier(self, strength):
        a = quality_adjustment_multiplier(strength, "A")
        b = quality_adjustment_multiplier(strength, "B")
        c = quality_adjustment_multiplier(strength, "C")
        assert c > b > a, (
            f"strength={strength}: expected C > B > A, got A={a} B={b} C={c}"
        )
        assert a == pytest.approx(0.95)
        assert b == pytest.approx(1.0)
        assert c == pytest.approx(1.05)

    # ---- Boundary at strength=0.5: falls into the strong branch ----

    def test_boundary_strength_half_uses_strong_branch(self):
        """strength == 0.5 is the neutral point of quality_adjustment_factor
        (returns 1.0). Tier preference uses the strong-paper table so that
        'decent paper' is rewarded with A-tier when scope fits.
        """
        assert quality_adjustment_multiplier(0.5, "A") == pytest.approx(1.05)
        assert quality_adjustment_multiplier(0.5, "B") == pytest.approx(1.02)
        assert quality_adjustment_multiplier(0.5, "C") == pytest.approx(1.0)

    # ---- Unknown / None / empty CCF: no bias ----

    @pytest.mark.parametrize("rating", [None, "", "unknown", "D", "未知"])
    def test_unknown_ccf_rating_returns_neutral(self, rating):
        for strength in (0.0, 0.4, 0.5, 0.7):
            m = quality_adjustment_multiplier(strength, rating)
            assert m == pytest.approx(1.0), (
                f"strength={strength}, rating={rating}: expected 1.0, got {m}"
            )

    # ---- Stability: the multiplier must be a pure function, no I/O ----

    def test_is_pure_function(self):
        """Same inputs -> same output. Important because the multiplier is
        called inside hot loops in the rule scorer.
        """
        for _ in range(3):
            assert quality_adjustment_multiplier(0.4, "A") == pytest.approx(0.95)
            assert quality_adjustment_multiplier(0.7, "C") == pytest.approx(1.0)


class TestQualityAdjustmentMultiplierIntegration:
    """Verify the multiplier composes correctly with the existing
    quality_adjustment_factor so the combined adjustment behaves as
    described in pipeline.py::_apply_quality_adjustment.
    """

    def test_weak_paper_c_tier_gets_combined_boost(self):
        """For a weak paper (strength=0.3), C-tier should get a combined
        adjustment >= 1.0 (the new behavior), while A-tier gets a
        combined adjustment < 1.0.

        base = 1.0 + 0.2 * (0.3 - 0.5) = 0.96
        weak branch: A=0.95, C=1.05
        combined:    A = 0.96 * 0.95 = 0.912 < 1.0
                     C = 0.96 * 1.05 = 1.008 >= 1.0
        """
        strength = 0.3
        base = quality_adjustment_factor(strength)
        a_combined = base * quality_adjustment_multiplier(strength, "A")
        c_combined = base * quality_adjustment_multiplier(strength, "C")
        assert c_combined > a_combined
        assert a_combined < 1.0
        assert c_combined >= 1.0

    def test_strong_paper_a_tier_gets_combined_boost(self):
        """For a strong paper (strength=0.7), A-tier should get a combined
        adjustment > C-tier.

        base = 1.0 + 0.2 * (0.7 - 0.5) = 1.04
        strong branch: A=1.05, C=1.0
        combined:      A = 1.04 * 1.05 = 1.092
                       C = 1.04 * 1.0  = 1.04
        """
        strength = 0.7
        base = quality_adjustment_factor(strength)
        a_combined = base * quality_adjustment_multiplier(strength, "A")
        c_combined = base * quality_adjustment_multiplier(strength, "C")
        assert a_combined > c_combined

"""Tests for the weighted evidence_composite formula (P0' 2026-06-16).

Old formula (equal-weight):
    fit_mean = mean(scope_fit, method_fit, application_fit, journal_position_fit)
    penalty_mean = mean(too_broad_penalty, too_narrow_penalty)
    composite = max(0, min(1, fit_mean - penalty_mean))

New formula (weighted, derived from discrimination analysis):
    weighted_fit = 0.35*scope_fit + 0.25*application_fit
                  + 0.20*journal_position_fit + 0.20*method_fit
    penalty_mean = mean(too_broad_penalty, too_narrow_penalty)  # unchanged
    composite = max(0, min(1, weighted_fit - penalty_mean))

Fallback: if evidence_field_weights is absent or weights sum != 1.0
(within tolerance), fall back to the equal-weight formula.

The contract here is the _evidence_composite classmethod on
LLMEvidenceRoleRanker. It must read the weights from app_config
and produce the correct weighted score.
"""
import pytest

from src.ranker.llm_evidence_role_ranker import LLMEvidenceRoleRanker


def _evidence(**kwargs):
    """Build a minimal evidence dict with sensible defaults."""
    base = {
        "scope_fit": 0.7,
        "method_fit": 0.7,
        "application_fit": 0.7,
        "journal_position_fit": 0.7,
        "too_broad_penalty": 0.0,
        "too_narrow_penalty": 0.0,
    }
    base.update(kwargs)
    return base


# Note on calling pattern:
# _evidence_composite is a @classmethod. When accessed via the class,
# Python auto-binds `cls`. So we call it WITHOUT explicit class first:
#   LLMEvidenceRoleRanker._evidence_composite(ev, weights=...)
# If we manually pass class first, Python sees it as a duplicate bind.
COMPOSITE = LLMEvidenceRoleRanker._evidence_composite


class TestEqualWeightFallback:
    """No weights configured → fall back to equal-weight mean."""

    def test_no_weights_uses_equal_weight(self):
        """Empty config → all 4 fit fields equal weight 0.25."""
        ev = _evidence(scope_fit=0.8, method_fit=0.6,
                       application_fit=0.7, journal_position_fit=0.9,
                       too_broad_penalty=0.1, too_narrow_penalty=0.0)
        # Equal-weight mean = (0.8+0.6+0.7+0.9)/4 = 0.75
        # penalty_mean = (0.1+0.0)/2 = 0.05
        # composite = 0.75 - 0.05 = 0.70
        result = COMPOSITE(ev, weights=None)
        assert abs(result - 0.70) < 1e-9


class TestWeightedFormula:
    """With explicit weights → weighted_fit, then subtract penalty_mean."""

    def test_basic_weighted(self):
        """All fields 0.7, no penalty → weighted = 0.7, composite = 0.7."""
        weights = {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }
        ev = _evidence()  # all 0.7
        result = COMPOSITE(ev, weights=weights)
        assert abs(result - 0.7) < 1e-9

    def test_scope_fit_dominates(self):
        """scope_fit higher than others → weighted shifts toward scope_fit."""
        weights = {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }
        # scope_fit=1.0 (max weight 0.35), others=0.0
        ev = _evidence(scope_fit=1.0, method_fit=0.0,
                       application_fit=0.0, journal_position_fit=0.0)
        # weighted = 0.35*1.0 + 0.25*0.0 + 0.20*0.0 + 0.20*0.0 = 0.35
        result = COMPOSITE(ev, weights=weights)
        assert abs(result - 0.35) < 1e-9

    def test_method_fit_weakest(self):
        """method_fit high but method has lowest weight → modest contribution."""
        weights = {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }
        # Only method_fit=1.0, others=0.0 → weighted = 0.20
        ev = _evidence(scope_fit=0.0, method_fit=1.0,
                       application_fit=0.0, journal_position_fit=0.0)
        result = COMPOSITE(ev, weights=weights)
        assert abs(result - 0.20) < 1e-9

    def test_penalty_unchanged(self):
        """Penalty subtracts as equal-weight mean regardless of fit weights."""
        weights = {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }
        ev = _evidence(too_broad_penalty=0.2, too_narrow_penalty=0.0)
        # weighted = 0.7, penalty_mean = (0.2+0.0)/2 = 0.1
        # composite = 0.7 - 0.1 = 0.6
        result = COMPOSITE(ev, weights=weights)
        assert abs(result - 0.6) < 1e-9

    def test_output_clamped_to_unit_interval(self):
        """Result must stay in [0, 1]."""
        weights = {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }
        # All fields 0 except huge penalty → should clamp to 0
        ev = _evidence(scope_fit=0.0, method_fit=0.0,
                       application_fit=0.0, journal_position_fit=0.0,
                       too_broad_penalty=1.0, too_narrow_penalty=1.0)
        result = COMPOSITE(ev, weights=weights)
        assert result == 0.0


class TestWeightValidation:
    """Bad weights config → fall back to equal-weight."""

    def test_empty_weights_dict_uses_equal(self):
        ev = _evidence(scope_fit=0.8, method_fit=0.6,
                       application_fit=0.7, journal_position_fit=0.9,
                       too_broad_penalty=0.0)
        # Equal-weight: 0.75, penalty_mean=0.0, composite=0.75
        result = COMPOSITE(ev, weights={})
        assert abs(result - 0.75) < 1e-9

    def test_weights_not_summing_to_one_uses_equal(self):
        """If weights sum != 1.0, fall back to equal-weight."""
        # Weights sum to 0.5, not 1.0
        weights = {
            "scope_fit": 0.20,
            "application_fit": 0.15,
            "journal_position_fit": 0.10,
            "method_fit": 0.05,
        }
        ev = _evidence(scope_fit=0.8, method_fit=0.6,
                       application_fit=0.7, journal_position_fit=0.9,
                       too_broad_penalty=0.0)
        # Equal-weight: 0.75, penalty_mean=0.0, composite=0.75
        result = COMPOSITE(ev, weights=weights)
        assert abs(result - 0.75) < 1e-9

    def test_weights_summing_to_one_uses_weighted(self):
        """If weights sum to 1.0 (within tolerance), use weighted."""
        weights = {
            "scope_fit": 0.4,
            "application_fit": 0.3,
            "journal_position_fit": 0.2,
            "method_fit": 0.1,
        }
        # scope_fit=1.0, others=0.0 → weighted = 0.4*1.0 + 0 = 0.4
        ev = _evidence(scope_fit=1.0, method_fit=0.0,
                       application_fit=0.0, journal_position_fit=0.0,
                       too_broad_penalty=0.0)
        result = COMPOSITE(ev, weights=weights)
        assert abs(result - 0.4) < 1e-9


class TestRealisticComparison:
    """Compare weighted vs equal-weight on a realistic evidence dict."""

    def test_weighted_changes_ordering(self):
        """Weighted formula should rank differently from equal-weight.

        This is the key behavioral change: with weights, scope_fit has
        more impact than method_fit on the composite score.
        """
        weights = {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }
        # Two candidates:
        # A: high scope_fit, low method_fit → weighted should favor
        # B: high method_fit, low scope_fit → equal-weight would tie
        ev_a = _evidence(scope_fit=0.9, method_fit=0.5,
                         application_fit=0.7, journal_position_fit=0.7)
        ev_b = _evidence(scope_fit=0.5, method_fit=0.9,
                         application_fit=0.7, journal_position_fit=0.7)
        # Equal-weight: both = 0.7 (tie)
        # Weighted A: 0.35*0.9 + 0.25*0.7 + 0.20*0.7 + 0.20*0.5
        #           = 0.315 + 0.175 + 0.140 + 0.100 = 0.730
        # Weighted B: 0.35*0.5 + 0.25*0.7 + 0.20*0.7 + 0.20*0.9
        #           = 0.175 + 0.175 + 0.140 + 0.180 = 0.670
        result_a = COMPOSITE(ev_a, weights=weights)
        result_b = COMPOSITE(ev_b, weights=weights)
        assert result_a > result_b, (
            f"Weighted should prefer scope_fit=0.9 over method_fit=0.9: "
            f"A={result_a} vs B={result_b}"
        )
        # Sanity: equal-weight would tie at 0.7
        equal_a = COMPOSITE(ev_a, weights=None)
        equal_b = COMPOSITE(ev_b, weights=None)
        assert abs(equal_a - equal_b) < 1e-9

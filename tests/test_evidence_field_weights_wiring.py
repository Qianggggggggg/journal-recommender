"""Integration test: app_config yaml → ranker.evidence_field_weights.

This is a wiring test, not a unit test. It verifies that when the yaml
config has `evidence_role.evidence_field_weights`, the LLMEvidenceRoleRanker
constructor receives it and stores it on the instance. Without this,
the formula change is silently no-op'd by the loader.
"""
import pytest
import yaml

from src.ranker.llm_evidence_role_ranker import LLMEvidenceRoleRanker


# Stub classes to bypass heavy init dependencies
class _StubExtractor:
    pass

class _StubJournalStore:
    pass


def _make_ranker_from_yaml(yaml_text: str) -> LLMEvidenceRoleRanker:
    """Construct a ranker from a yaml config string."""
    cfg = yaml.safe_load(yaml_text)
    evidence_cfg = cfg["ranking"]["evidence_role"]
    weights = evidence_cfg.get("evidence_field_weights")
    return LLMEvidenceRoleRanker(
        evidence_extractor=_StubExtractor(),
        journal_store=_StubJournalStore(),
        prior_source=evidence_cfg.get("prior_source", "rule"),
        evidence_weight=float(evidence_cfg.get("evidence_weight", 0.8)),
        prior_weight=float(evidence_cfg.get("prior_weight", 0.2)),
        ltr_score_weight=float(evidence_cfg.get("ltr_score_weight", 0.0)),
        evidence_field_weights=weights,
    )


class TestYamlWiring:
    """app_config.yaml.evidence_role.evidence_field_weights → ranker instance."""

    def test_weights_pass_through(self):
        """When yaml has evidence_field_weights, ranker stores them."""
        yaml_text = """
ranking:
  evidence_role:
    evidence_weight: 0.55
    prior_weight: 0.35
    ltr_score_weight: 0.10
    prior_source: learned
    evidence_field_weights:
      scope_fit: 0.35
      application_fit: 0.25
      journal_position_fit: 0.20
      method_fit: 0.20
"""
        ranker = _make_ranker_from_yaml(yaml_text)
        assert ranker.evidence_field_weights == {
            "scope_fit": 0.35,
            "application_fit": 0.25,
            "journal_position_fit": 0.20,
            "method_fit": 0.20,
        }

    def test_absent_weights_pass_through_as_none(self):
        """When yaml omits evidence_field_weights, ranker stores None."""
        yaml_text = """
ranking:
  evidence_role:
    evidence_weight: 0.55
    prior_weight: 0.35
    ltr_score_weight: 0.10
    prior_source: learned
"""
        ranker = _make_ranker_from_yaml(yaml_text)
        assert ranker.evidence_field_weights is None

    def test_weights_actually_used_in_formula(self):
        """End-to-end: yaml weights flow into _evidence_composite result."""
        yaml_text = """
ranking:
  evidence_role:
    evidence_weight: 0.55
    prior_weight: 0.35
    ltr_score_weight: 0.10
    prior_source: learned
    evidence_field_weights:
      scope_fit: 0.35
      application_fit: 0.25
      journal_position_fit: 0.20
      method_fit: 0.20
"""
        ranker = _make_ranker_from_yaml(yaml_text)

        # A: high scope_fit, low method_fit
        ev_a = {
            "scope_fit": 0.9, "method_fit": 0.5,
            "application_fit": 0.7, "journal_position_fit": 0.7,
            "too_broad_penalty": 0.0, "too_narrow_penalty": 0.0,
        }
        ev_b = {
            "scope_fit": 0.5, "method_fit": 0.9,
            "application_fit": 0.7, "journal_position_fit": 0.7,
            "too_broad_penalty": 0.0, "too_narrow_penalty": 0.0,
        }
        score_a = LLMEvidenceRoleRanker._evidence_composite(
            ev_a, weights=ranker.evidence_field_weights
        )
        score_b = LLMEvidenceRoleRanker._evidence_composite(
            ev_b, weights=ranker.evidence_field_weights
        )
        # Weighted: scope_fit=0.9 → 0.730; method_fit=0.9 → 0.670
        assert score_a > score_b
        assert abs(score_a - 0.730) < 1e-9
        assert abs(score_b - 0.670) < 1e-9

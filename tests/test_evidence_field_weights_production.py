"""Tests for production configs/app.yaml evidence_field_weights state.

History:
- P0-2 (2026-06-18): evidence_field_weights added to yaml with
  scope_fit=0.35 / application_fit=0.25 / journal_position_fit=0.20 /
  method_fit=0.20. Discrimination analysis showed scope/application
  ~2x the power of method_fit.
- Rollback (2026-06-18, hit@5 regression -6): yaml key removed; ranker
  falls back to equal-weight mean of fit_fields.

This test now locks the yaml to its CURRENT state (key absent) and
verifies that:
1. The yaml state is intentional (either present-with-correct-sum or
   absent-as-fallback). A test failure here forces a discussion instead
   of a silent drift back into either direction.
2. When the key IS present (used by ablation experiments), the
   sum=1.0 contract holds — otherwise _evidence_composite silently
   uses equal-weight mean, defeating the purpose.
"""
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_app_yaml() -> dict:
    with open(REPO_ROOT / "configs" / "app.yaml") as f:
        return yaml.safe_load(f)


def _get_weights() -> dict | None:
    cfg = _load_app_yaml()
    return (
        cfg.get("ranking", {})
        .get("evidence_role", {})
        .get("evidence_field_weights")
    )


# ---- Current production state (2026-06-18 rollback) ------------------


def test_production_yaml_evidence_field_weights_is_either_absent_or_valid():
    """Lock the yaml state: either the key is absent (current
    production) OR present with sum=1.0. Catch silent drift.

    - If absent: ranker uses equal-weight mean fallback.
    - If present: weights must cover all 4 fit fields and sum to 1.0.
    """
    weights = _get_weights()
    if weights is None:
        return  # explicit fallback state — valid

    required = {"scope_fit", "method_fit", "application_fit", "journal_position_fit"}
    missing = required - set(weights.keys())
    assert not missing, (
        f"evidence_field_weights present but missing fields: {missing}"
    )
    s = sum(weights.values())
    assert abs(s - 1.0) < 1e-6, (
        f"evidence_field_weights sum={s}, must be 1.0 (within 1e-6). "
        f"Otherwise _evidence_composite silently uses equal-weight mean."
    )
    for field, w in weights.items():
        assert w > 0, f"{field} weight must be > 0, got {w}"


def test_rollback_baseline_state_is_documented():
    """Pin the current rollback decision in a test so future readers
    know this was deliberate, not an oversight.

    2026-06-18 P0-2 hit@5 regression: with evidence_field_weights set,
    holdout240 hit@5 dropped from 159 to 153 (-6, -2.5pp). MRR was
    unchanged (0.439→0.441). The weighted formula was hurting more
    than helping in aggregate. Decision: roll back the key, keep P0-1
    (accepted top_k fix) which was a pure config correctness fix.

    Note (2026-06-18): on rerun AFTER accepted_corpus was expanded
    (~100 journals refilled between baseline and rerun), hit@5
    stays at 153 even with evidence_field_weights absent. So the
    159→153 shift is dominated by accepted_corpus expansion (which
    dilutes gold's normalized_score in min-max fusion), not by the
    evidence_field_weights change. P0-2 rollback is still the right
    call (no upside observed), but the 159 number should be
    re-measured once the corpus stabilizes.

    If you re-enable evidence_field_weights, pair this with an
    ablation run and document the result.
    """
    weights = _get_weights()
    assert weights is None, (
        "evidence_field_weights re-enabled. Run a holdout240 ablation "
        "(baseline registry: holdout240_post_p01_fix = 159) before "
        "merging — see commit message 8384dc2."
    )


# ---- Ablation-mode contract (for evidence_field_weights = ON) ---------


def test_evidence_field_weights_wiring_round_trip():
    """When yaml DOES set the key, the ranker must consume it.
    Mirrors test_evidence_field_weights_wiring.py but driven by yaml
    instead of inline strings.
    """
    import importlib
    import yaml as _yaml

    cfg = _load_app_yaml()
    weights = (
        cfg.get("ranking", {})
        .get("evidence_role", {})
        .get("evidence_field_weights")
    )
    if weights is None:
        # Skip — production is in rollback state.
        import pytest
        pytest.skip("evidence_field_weights absent in production yaml")

    # Re-import ranker module fresh to pick up the yaml-driven config.
    ranker_mod = importlib.import_module("src.ranker.llm_evidence_role_ranker")
    from src.ranker.llm_evidence_role_ranker import LLMEvidenceRoleRanker

    class _StubExtractor:
        pass

    class _StubJournalStore:
        pass

    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=_StubExtractor(),
        journal_store=_StubJournalStore(),
        prior_source="learned",
        evidence_weight=0.55,
        prior_weight=0.35,
        ltr_score_weight=0.10,
        evidence_field_weights=weights,
    )
    assert ranker.evidence_field_weights == weights

"""Tests for production configs/app.yaml evidence_field_weights.

P0-2 (2026-06-18): evidence_field_weights code path exists but the
production yaml never sets the key, so the ranker silently falls back
to equal-weight mean of fit_fields. Discrimination analysis showed
scope_fit / application_fit have ~2x the discriminative power of
method_fit, so weighted aggregation should be the production default.

These tests lock the yaml state so the wiring doesn't silently regress.
"""
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_app_yaml() -> dict:
    with open(REPO_ROOT / "configs" / "app.yaml") as f:
        return yaml.safe_load(f)


def test_production_yaml_sets_evidence_field_weights():
    """P0-2: production config must enable evidence_field_weights.
    Without this key, the ranker silently uses equal-weight mean.
    """
    cfg = _load_app_yaml()
    weights = (
        cfg.get("ranking", {})
        .get("evidence_role", {})
        .get("evidence_field_weights")
    )
    assert weights is not None, (
        "P0-2 regression: configs/app.yaml ranking.evidence_role."
        "evidence_field_weights is missing. The ranker will fall back to "
        "equal-weight mean, losing the discrimination signal."
    )


def test_production_weights_cover_all_fit_fields():
    """All 4 fit fields must have a weight (else fallback to equal)."""
    cfg = _load_app_yaml()
    weights = (
        cfg.get("ranking", {})
        .get("evidence_role", {})
        .get("evidence_field_weights")
    )
    assert weights is not None
    required = {"scope_fit", "method_fit", "application_fit", "journal_position_fit"}
    missing = required - set(weights.keys())
    assert not missing, f"Missing fit field weights: {missing}"


def test_production_weights_sum_to_one_within_tolerance():
    """_evidence_composite requires weights sum=1.0 (within 1e-6) to
    actually use them. Otherwise the ranker falls back silently.
    """
    cfg = _load_app_yaml()
    weights = (
        cfg.get("ranking", {})
        .get("evidence_role", {})
        .get("evidence_field_weights")
    )
    assert weights is not None
    s = sum(weights.values())
    assert abs(s - 1.0) < 1e-6, (
        f"evidence_field_weights sum={s}, must be 1.0 (within 1e-6). "
        f"Otherwise _evidence_composite silently uses equal-weight mean."
    )


def test_production_weights_all_positive():
    """Negative or zero weights would silently flip discrimination signs."""
    cfg = _load_app_yaml()
    weights = (
        cfg.get("ranking", {})
        .get("evidence_role", {})
        .get("evidence_field_weights")
    )
    assert weights is not None
    for field, w in weights.items():
        assert w > 0, f"{field} weight must be > 0, got {w}"

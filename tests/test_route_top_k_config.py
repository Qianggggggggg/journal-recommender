"""Tests for route_top_k.abstract config invariants.

P0-1 (2026-06-18): accepted_vector top_k was set to 56 in app.yaml but the
largest accepted-paper corpus per journal is 15 (avg 9.5). The misconfigured
top_k silently degrades to "fetch all available" but the number itself is
misleading and inconsistent with sibling routes. Lock the production config
to sane upper bounds relative to actual corpus scale.
"""
import json
from pathlib import Path

import pytest  # noqa: F401  (used by skipif below)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_accepted_corpus_stats() -> dict[str, int]:
    accepted_dir = REPO_ROOT / "data" / "accepted_papers"
    counts: dict[str, int] = {}
    for p in sorted(accepted_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            counts[p.stem] = len(d.get("papers", []))
        except Exception:
            continue
    return counts


def _load_app_yaml() -> dict:
    import yaml  # local import to keep this test independent of API path
    with open(REPO_ROOT / "configs" / "app.yaml") as f:
        return yaml.safe_load(f)


def test_accepted_vector_top_k_within_corpus_scale():
    """P0-1: accepted_vector top_k must not exceed the largest journal's
    corpus. Otherwise the value is a copy-paste leftover and silently
    degrades to "fetch all" while making the config harder to reason about.
    """
    cfg = _load_app_yaml()
    route_top_k = (
        cfg.get("candidate_generator", {})
        .get("route_top_k", {})
        .get("abstract", {})
    )
    accepted_vector_k = route_top_k.get("accepted_vector")
    assert accepted_vector_k is not None, (
        "configs/app.yaml route_top_k.abstract.accepted_vector missing"
    )
    counts = _load_accepted_corpus_stats()
    assert counts, "accepted-paper corpus is empty; cannot validate top_k"
    max_corpus = max(counts.values())
    # Allow a small headroom (e.g. if someone fills the corpus to a slightly
    # larger size later), but flag clearly oversized values.
    assert accepted_vector_k <= max_corpus + 5, (
        f"P0-1 regression: accepted_vector top_k={accepted_vector_k} but the "
        f"largest journal has only {max_corpus} accepted papers (avg "
        f"{sum(counts.values()) / len(counts):.1f}, "
        f"{len(counts)} journals total). Set top_k close to the corpus max."
    )


def test_accepted_bm25_top_k_within_corpus_scale():
    """P0-1 sibling check: accepted_bm25 should be similarly bounded."""
    cfg = _load_app_yaml()
    route_top_k = (
        cfg.get("candidate_generator", {})
        .get("route_top_k", {})
        .get("abstract", {})
    )
    accepted_bm25_k = route_top_k.get("accepted_bm25")
    assert accepted_bm25_k is not None, (
        "configs/app.yaml route_top_k.abstract.accepted_bm25 missing"
    )
    counts = _load_accepted_corpus_stats()
    max_corpus = max(counts.values())
    assert accepted_bm25_k <= max_corpus + 5, (
        f"accepted_bm25 top_k={accepted_bm25_k} exceeds corpus max "
        f"{max_corpus}. Same copy-paste concern as accepted_vector."
    )


def test_route_top_k_keys_are_consistent():
    """All four abstract route keys must be present and ints."""
    cfg = _load_app_yaml()
    route_top_k = (
        cfg.get("candidate_generator", {})
        .get("route_top_k", {})
        .get("abstract", {})
    )
    expected = {"bm25", "vector", "text", "accepted_bm25", "accepted_vector"}
    assert expected.issubset(route_top_k.keys()), (
        f"Missing keys in route_top_k.abstract: {expected - route_top_k.keys()}"
    )
    for k, v in route_top_k.items():
        assert isinstance(v, int), f"{k} must be int, got {type(v).__name__}"
        assert v > 0, f"{k} must be positive, got {v}"

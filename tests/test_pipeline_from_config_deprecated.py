"""P1-1 (2026-06-18): RecommenderPipeline.from_config is deprecated.

Previously constructed a RuleScorer with code-default weights (8 critical
weights = 0 in RuleScorer.DEFAULT_WEIGHTS), producing a half-disabled
scorer. No production path used this classmethod — it was dead code with
a footgun. Now raises NotImplementedError to make the deprecation loud.
"""
import pytest


def test_from_config_raises_not_implemented():
    """Calling from_config must raise NotImplementedError, not silently
    build a half-disabled pipeline."""
    from src.recommender.pipeline import RecommenderPipeline

    with pytest.raises(NotImplementedError) as exc_info:
        RecommenderPipeline.from_config()
    # The error message must point at the alternative (DI).
    msg = str(exc_info.value).lower()
    assert "dependency injection" in msg or "di" in msg, (
        f"Error message should mention DI alternative, got: {exc_info.value!r}"
    )


def test_from_config_never_instantiates_rule_scorer_with_defaults():
    """Regression: the old impl called RuleScorer() with no kwargs,
    inheriting DEFAULT_WEIGHTS (8 critical weights=0). If anyone
    accidentally restores that behavior, this test will fail because
    the deprecation raise must happen first."""
    from src.recommender.pipeline import RecommenderPipeline

    # We can't easily assert "didn't construct RuleScorer()" — but if
    # from_config returned instead of raising, the rule_scorer would
    # have DEFAULT_WEIGHTS (all-zero). Just confirm raise is loud.
    try:
        result = RecommenderPipeline.from_config()
    except NotImplementedError:
        return  # expected path
    # If we got here without raise, the deprecation has been broken.
    pytest.fail(
        "from_config returned an object without raising; "
        "deprecation has been silently reverted."
    )

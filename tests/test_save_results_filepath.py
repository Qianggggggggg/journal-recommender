"""Regression test for save_results when --output/filepath is provided.

Bug 2026-06-16 (holdout240 v2 ablation): When `filepath` is passed to
save_results, the if-branch (line 1254-1256) skips the else-branch where
`timestamp = datetime.now().strftime(...)` is assigned. But line 1270's
`result_dict["timestamp"] = timestamp` references it unconditionally,
causing UnboundLocalError.

The contract: save_results must write a valid JSON file at the given
filepath regardless of whether the caller provided an explicit path or
relied on auto-naming. Both paths must produce a result with a `timestamp`
field.
"""
import json
from dataclasses import asdict
from datetime import datetime

from scripts.run_evaluation import EvaluationResult, save_results


def _make_dummy_result() -> EvaluationResult:
    """Build a minimal EvaluationResult with just the fields save_results touches."""
    return EvaluationResult(
        mode="abstract", top_k=5, total_count=1,
        hit_at_1=1, hit_at_3=2, hit_at_5=3, hit_at_10=3,
        area_match_count=1, area_subject_tag_match_count=1,
        level_match_count=0,
        mrr=0.5, ndcg_at_5=0.6,
        coarse_hit_count=1,
        coarse_hit_in_rule_top10_count=1, coarse_hit_in_rule_top20_count=1,
        fallback_count=0, llm_success_count=1,
        empty_recommendation_count=0,
        paper_results=[], by_area={}, by_level={},
    )


class TestSaveResultsWithExplicitFilepath:
    """Bug regression: --output filepath must not crash on UnboundLocalError."""

    def test_explicit_filepath_writes_valid_json(self, tmp_path):
        """filepath is given → file is written, contains timestamp."""
        out = tmp_path / "subdir" / "v2_result.json"
        result = _make_dummy_result()
        # Before the fix: this raises UnboundLocalError on line 1270
        save_results(
            result,
            filepath=str(out),
            benchmark_profile="holdout240",
            benchmark_path="data/evaluation/papers_metadata_holdout240.jsonl",
        )
        assert out.exists(), "File should be written to filepath"
        with open(out) as f:
            data = json.load(f)
        assert "timestamp" in data, "Saved JSON must have timestamp"
        assert isinstance(data["timestamp"], str)
        assert data["mode"] == "abstract"
        assert data["top_k"] == 5
        assert data["metrics"]["hit_at_5"] == 3

    def test_auto_path_still_works(self, tmp_path):
        """No filepath → auto-naming path still works (regression guard)."""
        result = _make_dummy_result()
        save_results(
            result,
            output_dir=str(tmp_path),
            benchmark_profile="holdout240",
            benchmark_path="data/evaluation/papers_metadata_holdout240.jsonl",
        )
        files = list(tmp_path.glob("eval_*.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            data = json.load(f)
        assert "timestamp" in data

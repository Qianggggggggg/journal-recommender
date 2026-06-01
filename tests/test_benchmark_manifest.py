import json
from pathlib import Path

import pytest

from scripts.run_evaluation import EvaluationResult, save_results


def test_build_benchmark_manifest_hashes_configs_and_records_models(tmp_path):
    from src.evaluation.benchmark_manifest import build_benchmark_manifest, hash_file

    input_path = tmp_path / "papers.jsonl"
    app_config_path = tmp_path / "app.yaml"
    prompts_path = tmp_path / "prompts.yaml"

    input_path.write_text('{"title": "paper"}\n', encoding="utf-8")
    app_config_path.write_text(
        """
minimax:
  model: MiniMax-M3
ollama:
  embedding_model: qwen3-embedding:4b
""".strip(),
        encoding="utf-8",
    )
    prompts_path.write_text("paper_profile_system: test\n", encoding="utf-8")

    manifest = build_benchmark_manifest(
        input_path=input_path,
        mode="abstract",
        top_k=5,
        app_config_path=app_config_path,
        prompts_path=prompts_path,
        clean_benchmark=True,
        profile_snapshot_reused=False,
        timestamp="20260601_120000",
    )

    assert manifest == {
        "timestamp": "20260601_120000",
        "input_path": str(input_path),
        "mode": "abstract",
        "top_k": 5,
        "app_config_hash": hash_file(app_config_path),
        "prompt_hash": hash_file(prompts_path),
        "minimax_model": "MiniMax-M3",
        "embedding_model": "qwen3-embedding:4b",
        "clean_benchmark": True,
        "profile_snapshot_reused": False,
    }


def test_hash_file_raises_for_missing_path(tmp_path):
    from src.evaluation.benchmark_manifest import hash_file

    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "missing.yaml")


def test_save_results_writes_benchmark_manifest(tmp_path):
    result = EvaluationResult(
        total_count=0,
        mode="abstract",
        top_k=5,
        hit_at_1=0,
        hit_at_3=0,
        hit_at_5=0,
        hit_at_10=0,
        area_match_count=0,
        area_subject_tag_match_count=0,
        level_match_count=0,
        by_area={},
        by_level={},
        paper_results=[],
    )
    manifest = {
        "timestamp": "20260601_120000",
        "input_path": "data/evaluation/papers_metadata_light_30.jsonl",
        "mode": "abstract",
        "top_k": 5,
        "app_config_hash": "abc",
        "prompt_hash": "def",
        "minimax_model": "MiniMax-M3",
        "embedding_model": "qwen3-embedding:4b",
        "clean_benchmark": True,
        "profile_snapshot_reused": True,
    }

    saved_path = save_results(
        result,
        output_dir=str(tmp_path),
        benchmark_manifest=manifest,
    )

    data = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert data["benchmark_manifest"] == manifest


def test_build_baseline_record_extracts_manifest_and_metrics(tmp_path):
    from scripts.register_baseline_result import build_baseline_record

    result_path = tmp_path / "eval.json"
    result_path.write_text(
        json.dumps(
            {
                "benchmark_manifest": {
                    "input_path": "data/evaluation/papers_metadata_light_30.jsonl",
                    "app_config_hash": "app-hash",
                    "prompt_hash": "prompt-hash",
                    "minimax_model": "MiniMax-M2.7",
                },
                "metrics": {
                    "hit_at_5": 14,
                    "mrr": 0.2589,
                    "coarse_hit_count": 28,
                    "coarse_hit_in_rule_top20_count": 24,
                    "acceptable_journal_hit_at_5": 24,
                },
            }
        ),
        encoding="utf-8",
    )

    record = build_baseline_record(result_path, label="light30_m27_default")

    assert record == {
        "label": "light30_m27_default",
        "result_path": str(result_path),
        "input_path": "data/evaluation/papers_metadata_light_30.jsonl",
        "hit_at_5": 14,
        "mrr": 0.2589,
        "coarse_hit_count": 28,
        "coarse_hit_in_rule_top20_count": 24,
        "acceptable_journal_hit_at_5": 24,
        "app_config_hash": "app-hash",
        "prompt_hash": "prompt-hash",
        "minimax_model": "MiniMax-M2.7",
    }


def test_register_baseline_result_rejects_duplicate_label(tmp_path):
    from scripts.register_baseline_result import register_baseline_record

    registry_path = tmp_path / "baseline_registry.json"
    existing = {
        "baselines": [
            {
                "label": "light30_m27_default",
                "result_path": "old.json",
            }
        ]
    }
    registry_path.write_text(json.dumps(existing), encoding="utf-8")
    record = {
        "label": "light30_m27_default",
        "result_path": "new.json",
    }

    with pytest.raises(ValueError, match="already exists"):
        register_baseline_record(record, registry_path=registry_path, replace=False)


def test_register_baseline_result_can_replace_duplicate_label(tmp_path):
    from scripts.register_baseline_result import register_baseline_record

    registry_path = tmp_path / "baseline_registry.json"
    registry_path.write_text(
        json.dumps({"baselines": [{"label": "light30_m27_default", "result_path": "old.json"}]}),
        encoding="utf-8",
    )
    record = {
        "label": "light30_m27_default",
        "result_path": "new.json",
    }

    register_baseline_record(record, registry_path=registry_path, replace=True)

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    assert data["baselines"] == [record]

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

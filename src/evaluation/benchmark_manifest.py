"""Reproducibility metadata for evaluation runs."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def hash_file(path: str | Path) -> str:
    """Return a SHA256 hash for a file."""
    file_path = Path(path)
    with file_path.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_benchmark_manifest(
    *,
    input_path: str | Path,
    mode: str,
    top_k: int,
    app_config_path: str | Path = "configs/app.yaml",
    prompts_path: str | Path = "configs/prompts.yaml",
    clean_benchmark: bool = False,
    profile_snapshot_reused: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a compact manifest that makes an evaluation result reproducible."""
    app_config_path = Path(app_config_path)
    prompts_path = Path(prompts_path)

    with app_config_path.open("r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f) or {}

    return {
        "timestamp": timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_path": str(input_path),
        "mode": mode,
        "top_k": top_k,
        "app_config_hash": hash_file(app_config_path),
        "prompt_hash": hash_file(prompts_path),
        "minimax_model": app_config.get("minimax", {}).get("model"),
        "embedding_model": app_config.get("ollama", {}).get("embedding_model"),
        "clean_benchmark": clean_benchmark,
        "profile_snapshot_reused": profile_snapshot_reused,
    }

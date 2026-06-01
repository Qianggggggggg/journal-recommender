"""Helpers for constructing configured LLM clients."""

from __future__ import annotations

import os
from typing import Any

from .llm import MiniMaxLLM


def build_minimax_llm(app_config: dict[str, Any], api_key: str | None = None) -> MiniMaxLLM:
    """Create a MiniMax client from configs/app.yaml-style settings."""
    minimax_config = app_config.get("minimax", {})
    configured_key = api_key or minimax_config.get("api_key")
    resolved_key = _resolve_api_key(configured_key)
    if not resolved_key:
        raise RuntimeError("MINIMAX_API_KEY 未配置")

    return MiniMaxLLM(
        api_key=resolved_key,
        base_url=minimax_config.get("base_url", "https://api.minimax.chat"),
        model=minimax_config.get("model", "MiniMax-M2.7"),
        temperature=float(minimax_config.get("temperature", 0.1)),
        max_tokens=int(minimax_config.get("max_tokens", 8192)),
    )


def _resolve_api_key(value: str | None) -> str | None:
    if value and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1])
    return value or os.getenv("MINIMAX_API_KEY")

"""Tests for scripts/precompute_evidence.py CLI.

P0.4 (2026-06-16): the precompute script accepts ``--prompt-version v1|v2``
and routes the selected prompt into ``LLMEvidenceExtractor``. The behavior
is locked here so future refactors cannot silently drop the v2 path.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "precompute_evidence.py"


def _load_precompute_module():
    """Import the precompute_evidence module without running main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "precompute_evidence_under_test", str(SCRIPT_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_prompt_version_is_v1():
    """Backward-compat default: omitting --prompt-version must keep v1."""
    module = _load_precompute_module()
    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "precompute_evidence.py",
            "--baseline-eval", "/tmp/x.json",
        ]
        args = module.parse_args()
        assert args.prompt_version == "v1"
    finally:
        sys.argv = saved_argv


def test_prompt_version_v2_accepted():
    """P0.4 wiring: --prompt-version v2 must be parseable."""
    module = _load_precompute_module()
    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "precompute_evidence.py",
            "--baseline-eval", "/tmp/x.json",
            "--prompt-version", "v2",
        ]
        args = module.parse_args()
        assert args.prompt_version == "v2"
    finally:
        sys.argv = saved_argv


def test_invalid_prompt_version_rejected():
    """An unknown version string must fail argparse, not silently fall back.

    argparse's ``choices=["v1", "v2"]`` enforces this; lock the behavior.
    """
    module = _load_precompute_module()
    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "precompute_evidence.py",
            "--baseline-eval", "/tmp/x.json",
            "--prompt-version", "v9",
        ]
        with pytest.raises(SystemExit) as exc_info:
            module.parse_args()
        assert exc_info.value.code == 2  # argparse usage error
    finally:
        sys.argv = saved_argv


def test_module_uses_select_evidence_prompts_helper():
    """P0.4 invariant: precompute_evidence must use select_evidence_prompts.

    This guards against future edits that re-introduce the hardcoded
    ``prompts["llm_evidence_extractor_system"]`` lookups. We scan the
    source for the helper import + use of the helper.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "select_evidence_prompts" in src, (
        "precompute_evidence.py must import select_evidence_prompts; "
        "hardcoded 'llm_evidence_extractor_system' is no longer the v2-safe path"
    )
    # The hardcoded v1 key lookup should no longer be the construction path
    assert 'prompts["llm_evidence_extractor_system"]' not in src, (
        "precompute_evidence.py still hardcodes the v1 prompt key; "
        "replace with select_evidence_prompts(...) call"
    )

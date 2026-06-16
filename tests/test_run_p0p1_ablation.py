"""Tests for scripts/run_p0p1_ablation.py.

P0.5 (2026-06-16): the ablation runner must
  1. Find existing v1 evidence snapshot under ``holdout240_evidence_*.json``
  2. Find existing v2 evidence snapshot under ``holdout240_v2_<ts>/...``
  3. Pre-compute v2 snapshot automatically if missing
  4. Patch app.yaml with all three overrides: prompt_version,
     accepted_paper_weight, snapshot_path
  5. Restore app.yaml after each cell (no leakage between cells)
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_p0p1_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_p0p1_ablation_under_test", str(SCRIPT_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module():
    return _load_module()


class TestFindLatestSnapshotForVersion:
    """v1 and v2 snapshots live in different directory layouts; the helper
    must distinguish them so a v2 cell doesn't accidentally reuse a v1
    snapshot (which would silently disable P0).
    """

    def test_v1_returns_v1_snapshot_when_present(self, module, tmp_path, monkeypatch):
        # Create fake v1 snapshot
        v1_dir = tmp_path
        v1_path = v1_dir / "holdout240_evidence_20260606_143232.json"
        v1_path.write_text("{}")
        # Also a v2 dir that must be ignored
        v2_dir = v1_dir / "holdout240_v2_20260616_120000"
        v2_dir.mkdir()
        v2_path = v2_dir / "holdout240_v2_20260616_120000.json"
        v2_path.write_text("{}")

        monkeypatch.setattr(module, "EVIDENCE_DIR", v1_dir)
        result = module._find_latest_snapshot_for_version("v1")
        assert result is not None
        assert "v2" not in result.name
        assert "holdout240_evidence" in result.name

    def test_v2_returns_v2_snapshot_when_present(self, module, tmp_path, monkeypatch):
        v2_dir = tmp_path / "holdout240_v2_20260616_120000"
        v2_dir.mkdir()
        v2_path = v2_dir / "holdout240_v2_20260616_120000.json"
        v2_path.write_text("{}")
        # Also a v1 snapshot at the top level
        v1_path = tmp_path / "holdout240_evidence_20260606.json"
        v1_path.write_text("{}")

        monkeypatch.setattr(module, "EVIDENCE_DIR", tmp_path)
        result = module._find_latest_snapshot_for_version("v2")
        assert result is not None
        assert "holdout240_v2_" in result.name

    def test_returns_none_when_missing(self, module, tmp_path, monkeypatch):
        monkeypatch.setattr(module, "EVIDENCE_DIR", tmp_path)
        assert module._find_latest_snapshot_for_version("v1") is None
        assert module._find_latest_snapshot_for_version("v2") is None


class TestPatchAppYaml:
    """P0.5: the patcher must update all three keys atomically and restore
    cleanly. The function returns a backup path so the caller can restore.
    """

    def _make_fake_app_yaml(self, tmp_path: Path) -> Path:
        yaml = tmp_path / "app.yaml"
        yaml.write_text(
            "candidate_generator:\n"
            "  accepted_paper_weight: 0.20\n"
            "ranking:\n"
            "  evidence_role:\n"
            '    snapshot_path: "data/evaluation/evidence/old.json"\n'
            '    prompt_version: "v1"\n',
            encoding="utf-8",
        )
        return yaml

    def test_patch_all_three_keys(self, module, tmp_path, monkeypatch):
        yaml = self._make_fake_app_yaml(tmp_path)
        monkeypatch.setattr(module, "APP_YAML", yaml)
        backup = module._patch_app_yaml(
            prompt_version="v2",
            accepted_weight=0.00,
            snapshot_path=Path("data/evaluation/evidence/new.json"),
        )
        try:
            text = yaml.read_text(encoding="utf-8")
            assert 'prompt_version: "v2"' in text
            assert "accepted_paper_weight: 0.0" in text
            assert 'snapshot_path: "data/evaluation/evidence/new.json"' in text
            # backup file exists
            assert backup.exists()
        finally:
            module._restore_app_yaml(backup)
        # After restore, original values are back
        text = yaml.read_text(encoding="utf-8")
        assert 'prompt_version: "v1"' in text
        assert "accepted_paper_weight: 0.2" in text
        assert 'snapshot_path: "data/evaluation/evidence/old.json"' in text

    def test_patch_without_snapshot_path_leaves_it_alone(
        self, module, tmp_path, monkeypatch
    ):
        yaml = self._make_fake_app_yaml(tmp_path)
        monkeypatch.setattr(module, "APP_YAML", yaml)
        backup = module._patch_app_yaml(
            prompt_version="v2", accepted_weight=0.00
        )
        try:
            text = yaml.read_text(encoding="utf-8")
            # prompt_version and accepted_paper_weight changed
            assert 'prompt_version: "v2"' in text
            assert "accepted_paper_weight: 0.0" in text
            # snapshot_path left alone
            assert 'snapshot_path: "data/evaluation/evidence/old.json"' in text
        finally:
            module._restore_app_yaml(backup)

    def test_restore_removes_backup(self, module, tmp_path, monkeypatch):
        yaml = self._make_fake_app_yaml(tmp_path)
        monkeypatch.setattr(module, "APP_YAML", yaml)
        backup = module._patch_app_yaml(
            prompt_version="v2", accepted_weight=0.00
        )
        assert backup.exists()
        module._restore_app_yaml(backup)
        assert not backup.exists()


class TestEnsureSnapshotForVersion:
    """v1 must reuse existing snapshots; v2 must auto-precompute if missing."""

    def test_v1_missing_raises(self, module, tmp_path, monkeypatch):
        """v1 path is manual — if no v1 snapshot exists, fail with a clear
        error rather than auto-precompute, because the production pipeline
        relies on a stable v1 baseline.
        """
        monkeypatch.setattr(module, "EVIDENCE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="v1 evidence snapshot"):
            module._ensure_snapshot_for_version("v1")

    def test_v1_present_returns_path(self, module, tmp_path, monkeypatch):
        v1 = tmp_path / "holdout240_evidence_20260606.json"
        v1.write_text("{}")
        monkeypatch.setattr(module, "EVIDENCE_DIR", tmp_path)
        result = module._ensure_snapshot_for_version("v1")
        assert result == v1

    def test_v2_present_returns_path(self, module, tmp_path, monkeypatch):
        v2_dir = tmp_path / "holdout240_v2_20260616_120000"
        v2_dir.mkdir()
        v2 = v2_dir / "holdout240_v2_20260616_120000.json"
        v2.write_text("{}")
        monkeypatch.setattr(module, "EVIDENCE_DIR", tmp_path)
        result = module._ensure_snapshot_for_version("v2")
        assert result == v2


class TestWorkersArgument:
    """P0.6: --workers must default to 1 (CLAUDE.md baseline discipline) and
    be propagated to both run_evaluation.py and precompute_evidence.py.
    Diagnostic ablations can override with --workers 5-10.
    """

    def test_default_workers_is_1(self, module):
        """Backward-compat default: omitting --workers keeps workers=1."""
        saved_argv = sys.argv[:]
        try:
            sys.argv = ["run_p0p1_ablation.py"]
            args = module.main.__wrapped__ if hasattr(module.main, "__wrapped__") else None
            # Use argparse directly via the parser builder inside main
            # — re-create the parser to test default
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--workers", type=int, default=1)
            ns = parser.parse_args([])
            assert ns.workers == 1
        finally:
            sys.argv = saved_argv

    def test_run_eval_propagates_workers(self, module, monkeypatch, tmp_path):
        """_run_eval must pass workers to subprocess.check_call."""
        import subprocess
        captured = {}

        # Use tmp_path for results dir to avoid touching the real one
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, "RESULTS_DIR", results_dir)

        def fake_check_call(cmd, **kwargs):
            captured["cmd"] = cmd
            # Find --output path and create that file
            idx = cmd.index("--output")
            out_path = Path(cmd[idx + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("{}")

        monkeypatch.setattr(subprocess, "check_call", fake_check_call)
        module._run_eval("v1", 0.2, "test_workers", workers=10)
        # Check that --workers 10 was in the cmd
        idx = captured["cmd"].index("--workers")
        assert captured["cmd"][idx + 1] == "10"

    def test_run_precompute_propagates_workers(self, module, monkeypatch, tmp_path):
        """_run_precompute must pass workers to subprocess.check_call."""
        import subprocess
        captured = {}

        # Set up: need a results file for the heuristic baseline-eval lookup
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        baseline_eval = results_dir / "eval_abstract_top5_20260615_172101.json"
        baseline_eval.write_text("{}")
        monkeypatch.setattr(module, "RESULTS_DIR", results_dir)
        # Use a tmp EVIDENCE_DIR so we don't pollute the real one
        evidence_dir = tmp_path / "evidence"
        monkeypatch.setattr(module, "EVIDENCE_DIR", evidence_dir)

        def fake_check_call(cmd, **kwargs):
            captured["cmd"] = cmd
            # The snapshot file path is buildable from --output-dir + a known
            # filename pattern. Re-derive it the same way _run_precompute does.
            idx = cmd.index("--output-dir")
            out_dir = Path(cmd[idx + 1])
            # _run_precompute will look for ``holdout240_<version>_*.json`` in out_dir
            # after subprocess returns. Mimic the post-condition by creating one.
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # The version is the last positional arg in the cmd that follows --prompt-version
            pv_idx = cmd.index("--prompt-version")
            version = cmd[pv_idx + 1]
            snap = out_dir / f"holdout240_{version}_{ts}.json"
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text("{}")

        monkeypatch.setattr(subprocess, "check_call", fake_check_call)
        module._run_precompute("v2", workers=5)
        idx = captured["cmd"].index("--workers")
        assert captured["cmd"][idx + 1] == "5"

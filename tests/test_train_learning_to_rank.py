"""Tests for scripts/train_learning_to_rank.py (Task 5.2).

训练脚本契约:
- CLI: --train <jsonl> --output <json> --seed <int> [--max-iter N] [--use-standardization/--no-standardization]
- 输出 JSON schema:
    {
      "schema_version": 1,
      "model_type": "logistic_regression",
      "feature_names": [...],   # 长度 == feature_dim
      "coef": [...],            # 长度 == feature_dim
      "intercept": float,
      "use_standardization": bool,
      "scaler_mean": [...] | null,  # 当 use_standardization=True 时有
      "scaler_scale": [...] | null,
      "metrics": {
        "n_train": int,
        "n_positive": int,
        "n_negative": int,
        "pairwise_accuracy": float,        # 同 paper 中 gold score > hard_neg score 的比例
        "positive_mean_score": float,
        "hard_negative_mean_score": float,
      },
      "convergence_info": {
        "converged": bool,
        "n_iter": int | None,
        "max_iter": int,
        "warning_message": str | None,  # raw text of ConvergenceWarning
      },
      "seed": int,
    }
- 加载产物后能 predict (走 LearningToRanker.load)。
- 不能在训练数据缺失时静默成功。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _make_train_jsonl(path: Path, n_pos: int = 30, n_neg: int = 90) -> None:
    """写一个 19 维的小型训练 jsonl,符合 build_ranking_training_data.py 输出。

    2026-06-26: paper_strength removed (was 20-dim, now 19-dim).
    """
    feature_names = [
        "retrieval_rank", "rule_rank", "rule_score",
        "scope_bm25_rank", "scope_vector_rank",
        "typical_bm25_rank", "typical_vector_rank",
        "accepted_bm25_rank", "accepted_vector_rank",
        "route_count", "has_scope_route", "has_typical_route",
        "has_accepted_route", "has_identity_anchor",
        "same_gold_area", "same_parsed_ccf_area", "same_ccf_level",
        "journal_ccf_numeric", "candidate_in_accepted_corpus",
    ]
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_pos):
            row = {
                "paper_id": f"paper_{i // 3}",
                "journal_id": f"gold_{i}",
                "label": 1,
                "features": [1.0] * 19,
                "feature_names": feature_names,
                "negative_type": "gold",
                "variant": "full_hybrid",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        for i in range(n_neg):
            row = {
                "paper_id": f"paper_{i // 5}",
                "journal_id": f"neg_{i}",
                "label": 0,
                "features": [0.0] * 19,
                "feature_names": feature_names,
                "negative_type": "hard_rule_top20",
                "variant": "full_hybrid",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_train(train_path: Path, output_path: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "train_learning_to_rank.py"),
        "--train", str(train_path),
        "--output", str(output_path),
        "--seed", "42",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_train_script_writes_expected_schema(tmp_path: Path):
    """训练脚本输出必须包含 plan 5.2 要求的字段。"""
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "ltr.json"
    _make_train_jsonl(train_path)
    result = _run_train(train_path, out_path, "--max-iter", "1000")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model_type"] == "logistic_regression"
    assert isinstance(payload["feature_names"], list)
    assert len(payload["feature_names"]) == 19
    assert len(payload["coef"]) == 19
    assert isinstance(payload["intercept"], float)
    assert payload["seed"] == 42
    # metrics
    m = payload["metrics"]
    assert m["n_train"] == 120
    assert m["n_positive"] == 30
    assert m["n_negative"] == 90
    assert 0.0 <= m["pairwise_accuracy"] <= 1.0
    assert -1.0 <= m["positive_mean_score"] <= 1.0
    # convergence_info
    ci = payload["convergence_info"]
    assert "converged" in ci
    assert ci["max_iter"] == 1000


def test_train_script_with_real_data_runs_without_crashing(tmp_path: Path):
    """5.2.d 真实数据 sanity check:训练脚本必须能跑通,不能因为收敛问题崩。
    这条测试用的是仓库自带的 ranker_train_full_v2.jsonl(若存在)。
    """
    repo_train = Path(__file__).resolve().parent.parent / "data" / "training" / "ranker_train_full_v2.jsonl"
    if not repo_train.exists():
        pytest.skip("real training data not present")
    out_path = tmp_path / "ltr_real.json"
    # plan 5.2 重点:开 standardization + 高 max_iter,记录收敛状态
    result = _run_train(repo_train, out_path, "--max-iter", "5000", "--use-standardization")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    # pairwise accuracy 在合理 LTR baseline 上应 >= 0.5(显著好于随机 0.5)
    assert payload["metrics"]["pairwise_accuracy"] >= 0.5, payload["metrics"]
    # 真实数据在标准化 + 5000 iter 下应该收敛
    assert payload["convergence_info"]["converged"] is True, payload["convergence_info"]


def test_train_script_fails_on_missing_input(tmp_path: Path):
    """输入文件不存在时必须以非零码退出,不能静默成功。"""
    out_path = tmp_path / "ltr.json"
    missing = tmp_path / "does_not_exist.jsonl"
    result = _run_train(missing, out_path)
    assert result.returncode != 0


def test_train_script_fails_on_empty_input(tmp_path: Path):
    """空 jsonl 必须报错,不能训练零样本。"""
    train_path = tmp_path / "empty.jsonl"
    train_path.write_text("", encoding="utf-8")
    out_path = tmp_path / "ltr.json"
    result = _run_train(train_path, out_path)
    assert result.returncode != 0


def test_train_script_without_standardization_also_works(tmp_path: Path):
    """`--no-standardization` 必须保留为可用选项(向后兼容,plan 5.1 行为)。"""
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "ltr.json"
    _make_train_jsonl(train_path)
    result = _run_train(train_path, out_path, "--no-standardization")
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["use_standardization"] is False
    assert payload["scaler_mean"] is None
    assert payload["scaler_scale"] is None


def test_train_script_saves_scaler_when_standardized(tmp_path: Path):
    """`--use-standardization` 时产物必须包含 scaler_mean / scaler_scale,
    这样加载模型后能反推回原特征空间。
    """
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "ltr.json"
    _make_train_jsonl(train_path)
    result = _run_train(train_path, out_path, "--use-standardization")
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["use_standardization"] is True
    assert isinstance(payload["scaler_mean"], list)
    assert isinstance(payload["scaler_scale"], list)
    assert len(payload["scaler_mean"]) == 19
    assert len(payload["scaler_scale"]) == 19


def test_train_script_artifact_is_loadable_and_predictable(tmp_path: Path):
    """训练产物必须能被 LearningToRanker.load 加载并产出与训练时一致的预测。
    plan 5.2 要求: 产物 JSON 是下游推理的唯一入口,不能有 schema drift。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.ranker.learning_to_rank import LearningToRanker

    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "ltr.json"
    _make_train_jsonl(train_path)
    result = _run_train(train_path, out_path, "--use-standardization")
    assert result.returncode == 0, f"stderr={result.stderr}"

    loaded = LearningToRanker.load(str(out_path))
    assert loaded._feature_dim == 19
    assert loaded.use_standardization is True
    assert loaded._scaler is not None
    assert loaded.convergence_info is not None
    assert "converged" in loaded.convergence_info

    # 在训练数据上 predict 必须能跑通且不抛异常
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = loaded.predict_scores(rows)
    assert len(scores) == len(rows)
    # 全 1 行的 score 应高于全 0 行的 score
    pos_scores = [s for r, s in zip(rows, scores) if r["label"] == 1]
    neg_scores = [s for r, s in zip(rows, scores) if r["label"] == 0]
    assert min(pos_scores) > max(neg_scores)


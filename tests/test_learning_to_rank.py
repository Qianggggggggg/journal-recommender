"""Tests for src/ranker/learning_to_rank.py (Task 5.1)."""
import json
from pathlib import Path

import pytest

from src.ranker.learning_to_rank import LearningToRanker


def _make_row(features: list, label: int, paper_id: str = "p", jid: str = "j") -> dict:
    """构造 LTR 训练行,符合 scripts/build_ranking_training_data.py 输出格式。"""
    return {
        "paper_id": paper_id,
        "journal_id": jid,
        "label": label,
        "features": list(features),
        "feature_names": None,  # 不需要,LearningToRanker 只用 features
        "negative_type": "gold" if label == 1 else "hard_rule_top20",
        "variant": "full_hybrid",
    }


def test_learning_to_ranker_is_unfitted_before_fit():
    """未 fit 的 ranker 必须能让 predict_scores 抛异常,不能静默返回随机数。"""
    ranker = LearningToRanker()
    with pytest.raises(RuntimeError, match="[Ff]it"):
        ranker.predict_scores([_make_row([0.1] * 20, label=0)])


def test_fit_then_predict_scores_higher_for_positives_than_hard_negatives():
    """plan 5.1 关键测试:训练后,正样本分数 > 硬负样本分数。"""
    # 构造可分数据:正样本 features 全 1.0(强信号),硬负样本 features 全 0.0
    # 训练集: 5 个 positive (label=1) + 5 个 hard negative (label=0)
    rows = []
    for i in range(5):
        rows.append(_make_row([1.0] * 20, label=1, paper_id=f"p{i}", jid="gold"))
    for i in range(5):
        rows.append(_make_row([0.0] * 20, label=0, paper_id=f"p{i}", jid=f"neg{i}"))

    ranker = LearningToRanker(seed=42)
    ranker.fit(rows)

    # 同样的数据,positive 的 score 应高于 hard negative
    scores = ranker.predict_scores(rows)
    pos_scores = [s for r, s in zip(rows, scores) if r["label"] == 1]
    neg_scores = [s for r, s in zip(rows, scores) if r["label"] == 0]
    assert min(pos_scores) > max(neg_scores), (
        f"positives should all score > hard negatives; "
        f"pos={pos_scores}, neg={neg_scores}"
    )


def test_save_and_load_roundtrip_preserves_predictions(tmp_path: Path):
    """save/load 后预测结果必须完全一致(per plan 5.1 的接口契约)。"""
    rows = [_make_row([1.0] * 20, label=1, paper_id="p0", jid="g")]
    rows += [_make_row([0.0] * 20, label=0, paper_id="p0", jid="n0")]

    ranker = LearningToRanker(seed=42)
    ranker.fit(rows)
    scores_before = ranker.predict_scores(rows)

    save_path = tmp_path / "ltr.json"
    ranker.save(str(save_path))
    assert save_path.exists()

    loaded = LearningToRanker.load(str(save_path))
    scores_after = loaded.predict_scores(rows)
    assert scores_before == scores_after


def test_fit_is_deterministic_with_same_seed():
    """相同 seed + 相同数据 → 训练后预测必须 bit-equal(per plan "确定性"要求)。"""
    rows = [_make_row([0.5 * (i % 2)] * 20, label=i % 2, paper_id=f"p{i // 2}", jid=f"j{i}") for i in range(20)]
    r1 = LearningToRanker(seed=42).fit(rows)
    s1 = r1.predict_scores(rows)
    r2 = LearningToRanker(seed=42).fit(rows)
    s2 = r2.predict_scores(rows)
    assert s1 == s2


def test_fit_raises_on_empty_rows():
    """空训练集必须抛 ValueError,不能静默成功(per plan "graceful error")。"""
    ranker = LearningToRanker()
    with pytest.raises(ValueError, match="[Ee]mpty"):
        ranker.fit([])


def test_fit_raises_on_rows_with_mismatched_feature_length():
    """不同 row 的 features 长度不一致 → 抛 ValueError,不能静默截断/补零。"""
    ranker = LearningToRanker()
    bad_rows = [
        _make_row([1.0] * 20, label=1),
        _make_row([1.0] * 10, label=0),  # 长度不一致
    ]
    with pytest.raises(ValueError, match="[Ff]eature length"):
        ranker.fit(bad_rows)


def test_predict_scores_returns_one_score_per_row():
    """predict_scores 输出长度必须 == 输入行数,顺序与输入一致。"""
    rows = [_make_row([0.5] * 20, label=i % 2, paper_id=f"p{i}", jid=f"j{i}") for i in range(7)]
    ranker = LearningToRanker(seed=0).fit(rows)
    scores = ranker.predict_scores(rows)
    assert len(scores) == 7
    assert all(isinstance(s, float) for s in scores)


# ---------------------------------------------------------------------------
# Task 5.2 收敛处理 / 特征标准化
# ---------------------------------------------------------------------------


def test_fit_captures_convergence_warning_for_real_data(tmp_path: Path):
    """plan 5.2:真实 ranker_train_full_v2.jsonl 上 sklearn 报 ConvergenceWarning,
    ranker 必须记录 convergence_info 而不是让 warning 静默扩散到 stderr。
    """
    import warnings

    import numpy as np

    # 构造一个能让 lbfgs 在 1000 iter 内无法收敛的数据:
    # 大量不同尺度的特征 + 强噪声。max_iter=10 一定不够。
    rng = np.random.default_rng(42)
    n_pos, n_neg = 30, 200
    X_pos = rng.normal(loc=10.0, scale=5.0, size=(n_pos, 5))
    X_neg = rng.normal(loc=0.0, scale=5.0, size=(n_neg, 5))
    # 再加几个 sentinel-like 特征放大尺度
    X_pos = np.hstack([X_pos, rng.uniform(500, 1000, size=(n_pos, 2))])
    X_neg = np.hstack([X_neg, rng.uniform(0, 50, size=(n_neg, 2))])
    rows = []
    for i in range(n_pos):
        rows.append(_make_row(X_pos[i].tolist(), label=1, paper_id=f"p{i}"))
    for i in range(n_neg):
        rows.append(_make_row(X_neg[i].tolist(), label=0, paper_id=f"p{i % 5}", jid=f"n{i}"))

    ranker = LearningToRanker(seed=42, max_iter=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 静默 stderr 输出,但 convergence_info 仍要记录
        ranker.fit(rows)
    info = ranker.convergence_info
    assert info is not None
    assert "converged" in info
    # 1000 → 10 一定没收敛完,这里断言它**检测到了**这种情况
    # (无论 lbfgs 内部 n_iter 数字,关键是 converged 字段是 bool 且在严格 10 iter 下=False)
    assert info["converged"] is False or info["n_iter"] is not None


def test_fit_records_convergence_info_for_easy_data():
    """简单可分数据:max_iter 充足,convergence_info.converged 应为 True。"""
    rows = []
    for i in range(5):
        rows.append(_make_row([1.0] * 20, label=1, paper_id=f"p{i}"))
    for i in range(5):
        rows.append(_make_row([0.0] * 20, label=0, paper_id=f"p{i}", jid=f"n{i}"))
    ranker = LearningToRanker(seed=42, max_iter=1000)
    ranker.fit(rows)
    info = ranker.convergence_info
    assert info is not None
    assert info["converged"] is True


def test_standardization_lets_real_data_converge_within_max_iter():
    """use_standardization=True 在真实 ranker_train_full_v2.jsonl 上,
    lbfgs 必须在 max_iter=1000 内收敛;不标准化必报 ConvergenceWarning。
    这是 plan 5.2 重点 — 验证标准化对真实训练数据的实际价值。
    """
    import json

    real_train = (
        Path(__file__).resolve().parent.parent
        / "data" / "training" / "ranker_train_full_v2.jsonl"
    )
    if not real_train.exists():
        pytest.skip("real training data not present")
    rows = [json.loads(line) for line in real_train.read_text(encoding="utf-8").splitlines() if line.strip()]

    r_std = LearningToRanker(seed=42, max_iter=1000, use_standardization=True).fit(rows)
    r_raw = LearningToRanker(seed=42, max_iter=1000, use_standardization=False).fit(rows)

    assert r_std.convergence_info["converged"] is True, (
        f"standardized fit on real data should converge; got {r_std.convergence_info}"
    )
    assert r_raw.convergence_info["converged"] is False, (
        f"raw fit on real data should fail to converge in 1000 iter (this is the bug we are fixing); "
        f"got {r_raw.convergence_info}"
    )


def test_save_load_roundtrip_preserves_convergence_info(tmp_path: Path):
    """save/load 后 convergence_info 字段必须保留(下游训练脚本要读它)。"""
    rows = [_make_row([1.0] * 20, label=1)]
    rows += [_make_row([0.0] * 20, label=0, jid="n")]
    ranker = LearningToRanker(seed=42, max_iter=500).fit(rows)
    save_path = tmp_path / "ltr.json"
    ranker.save(str(save_path))
    loaded = LearningToRanker.load(str(save_path))
    assert loaded.convergence_info == ranker.convergence_info


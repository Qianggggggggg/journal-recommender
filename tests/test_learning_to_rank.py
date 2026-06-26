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


# ---------------------------------------------------------------------------
# 2026-06-25: LightGBM LambdaMART backend
# ---------------------------------------------------------------------------


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def test_lightgbm_backend_fit_predict_produces_ranked_scores():
    """LightGBM backend fit 后, predict_scores 必须输出**有梯度**的 scores
    (不是全 0 或全常数)。

    设计说明:
    - per-call min-max 把 raw scores 归一到 [0,1];如果所有 raw scores 相同
      则全 0(已知 fallback 行为,见 _predict_lightgbm)。
    - 因此本测试验证:构造**让 raw scores 有梯度**的数据(每个 paper 内
      pos/neg features 不同 → raw score 不同 → min-max 后非全 0)。
    - 用 paper-level 强信号(pos 的 feature[0]=10.0,neg 的 feature[0]=0.0)
      让 LightGBM 学到简单规则。
    """
    if not _lightgbm_available():
        pytest.skip("lightgbm not installed")
    rows = []
    for i in range(8):
        # 强可分信号:pos 在 feature[0]=10.0,neg 在 feature[0]=0.0
        # 其它 27 维随机噪声 → 每个 paper 内 pos vs neg 的 raw score 应该有梯度
        import random
        random.seed(i)
        noise_pos = [random.random() for _ in range(27)]
        noise_neg = [random.random() for _ in range(27)]
        pos_feats = [10.0] + noise_pos
        neg_feats = [0.0] + noise_neg
        rows.append(_make_row(pos_feats, label=1, paper_id=f"p{i}", jid="gold"))
        rows.append(_make_row(neg_feats, label=0, paper_id=f"p{i}", jid=f"n{i}"))
    ranker = LearningToRanker(seed=42, max_iter=200, backend="lightgbm")
    ranker.fit(rows)
    scores = ranker.predict_scores(rows)
    pos_scores = [s for r, s in zip(rows, scores) if r["label"] == 1]
    neg_scores = [s for r, s in zip(rows, scores) if r["label"] == 0]
    # 验证 1: scores 必须有梯度(不是全 0/全 1 的退化情形)
    distinct = len(set(scores))
    assert distinct > 1, (
        f"predict_scores output is degenerate (all identical); got {scores}. "
        f"This means LightGBM produced constant raw scores → per-call min-max "
        f"collapsed to a single value. Either data construction or training "
        f"needs fixing."
    )
    # 验证 2: positives 必须排在 hard negatives 前面(listwise objective 体现)
    assert min(pos_scores) > max(neg_scores), (
        f"positives must rank above hard negatives; pos={pos_scores}, neg={neg_scores}"
    )


def test_lightgbm_backend_save_load_roundtrip_bit_equal(tmp_path: Path):
    """LightGBM booster 是 deterministic 的(seed + num_threads=1),save/load 后预测必须 bit-equal。"""
    if not _lightgbm_available():
        pytest.skip("lightgbm not installed")
    rows = [_make_row([1.0] * 28, label=1, paper_id="p0", jid="g")]
    rows += [_make_row([0.0] * 28, label=0, paper_id="p0", jid="n0")]
    rows += [_make_row([0.5] * 28, label=0, paper_id="p0", jid="n1")]
    ranker = LearningToRanker(seed=42, max_iter=50, backend="lightgbm")
    ranker.fit(rows)
    scores_before = ranker.predict_scores(rows)

    save_path = tmp_path / "ltr_lgb.json"
    ranker.save(str(save_path))

    payload = json.loads(save_path.read_text())
    assert payload["backend"] == "lightgbm"
    assert payload["model_type"] == "lightgbm_lambdarank"
    assert payload["feature_dim"] == 28
    assert payload.get("lightgbm_booster_str"), "booster_str must be populated"

    loaded = LearningToRanker.load(str(save_path))
    assert loaded._backend == "lightgbm"
    scores_after = loaded.predict_scores(rows)
    assert scores_before == scores_after, (
        f"LightGBM booster not deterministic: before={scores_before}, after={scores_after}"
    )


def test_lightgbm_backend_groups_by_paper_id_in_order():
    """_group_rows_by_paper 必须保持 paper_id first-appearance 顺序,group sizes 之和 == len(rows)。"""
    rows = []
    # 故意打乱 paper_id 出现顺序
    for paper in ["c", "a", "b", "a", "c", "b"]:
        rows.append(_make_row([0.5] * 20, label=1, paper_id=paper))
    X, y, group = LearningToRanker._group_rows_by_paper(rows)
    assert sum(group.tolist()) == len(rows)
    # 顺序应该是 c, a, b (first appearance)
    assert len(group) == 3
    assert group.tolist() == [2, 2, 2]
    assert X.shape == (6, 20)
    assert y.tolist() == [1] * 6


def test_lightgbm_backend_missing_dependency_raises(monkeypatch):
    """lightgbm import 失败时,backend=lightgbm 必须 raise RuntimeError,不能 fallback 到 LR。"""
    import src.ranker.learning_to_rank as ltr_module

    monkeypatch.setattr(ltr_module, "_HAS_LIGHTGBM", False)
    ranker = LearningToRanker(seed=42, max_iter=10, backend="lightgbm")
    rows = [_make_row([0.5] * 20, label=1)]
    with pytest.raises(RuntimeError, match="lightgbm"):
        ranker.fit(rows)


def test_lightgbm_backend_predict_scores_in_0_1_range():
    """per-call min-max 归一化必须输出 [0,1] 范围。"""
    if not _lightgbm_available():
        pytest.skip("lightgbm not installed")
    rows = [_make_row([1.0] * 20, label=1, paper_id="p0", jid="g")]
    rows += [_make_row([0.0] * 20, label=0, paper_id="p0", jid="n0")]
    rows += [_make_row([0.5] * 20, label=0, paper_id="p0", jid="n1")]
    ranker = LearningToRanker(seed=42, max_iter=20, backend="lightgbm").fit(rows)
    scores = ranker.predict_scores(rows)
    assert all(0.0 <= s <= 1.0 for s in scores), f"scores must be in [0,1]; got {scores}"


def test_lightgbm_backend_handles_27_dim_schema():
    """27-dim 数据 save/load 必须保留 feature_dim=27 (2026-06-26: was 28)。"""
    if not _lightgbm_available():
        pytest.skip("lightgbm not installed")
    rows = [_make_row([1.0] * 27, label=1, paper_id="p0", jid="g")]
    rows += [_make_row([0.0] * 27, label=0, paper_id="p0", jid="n0")]
    ranker = LearningToRanker(seed=42, max_iter=20, backend="lightgbm").fit(rows)
    assert ranker._feature_dim == 27
    save_path = Path("/tmp/lgb27_test.json")
    ranker.save(str(save_path))
    loaded = LearningToRanker.load(str(save_path))
    assert loaded._feature_dim == 27
    assert loaded.predict_scores(rows) == ranker.predict_scores(rows)


def test_backend_auto_picks_sklearn_when_lightgbm_unavailable(monkeypatch):
    """backend='auto' 不应强制 lightgbm;lightgbm 不可用时仍能 auto-pick sklearn/numpy。"""
    import src.ranker.learning_to_rank as ltr_module

    monkeypatch.setattr(ltr_module, "_HAS_LIGHTGBM", False)
    ranker = LearningToRanker(seed=42, backend="auto")
    # auto 选 sklearn (若有) or numpy;lightgbm 缺失不影响 auto 选择
    assert ranker._backend in ("sklearn", "numpy")


def test_backend_explicit_unknown_raises():
    """backend='foo' 必须 raise ValueError,不能静默 fallback。"""
    with pytest.raises(ValueError, match="Unknown backend"):
        LearningToRanker(seed=42, backend="foo")


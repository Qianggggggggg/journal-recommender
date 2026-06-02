"""Learning-to-Rank reranker (Task 5.1 baseline).

第一版用 sklearn LogisticRegression(优先,per plan 5.1)。
sklearn 不可用时 fallback 到 numpy 实现的小型 logistic regression。

纪律:
- **确定性**:`random_state=seed` 锁定,同 seed + 同数据 → bit-equal 预测。
- **无新增依赖**:sklearn 已是项目隐式依赖(没在 pyproject.toml 显式列出,但
  conda 环境里有);numpy 是必需的(已用)。
- **可序列化**:save/load 用 JSON,人类可读,便于 git diff 与回归测试。
- **不接受坏数据**:空训练集 / 特征长度不一致必须抛 ValueError。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# sklearn 是首选(更稳、有 calibration),不可用时降级到 numpy
try:  # pragma: no cover
    from sklearn.linear_model import LogisticRegression as _SklearnLR

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


class LearningToRanker:
    """LTR baseline:logistic regression over paper-candidate features。

    训练目标:预测给定 (paper_features, journal_features) → label ∈ {0, 1}。
    推理:`predict_scores` 返回每行 score(类 1 的概率,越大越像 gold)。
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model: Any = None
        self._feature_dim: Optional[int] = None
        self._backend: str = "sklearn" if _HAS_SKLEARN else "numpy"

    # ---- 训练 / 推理 ----

    def _extract_xy(self, rows: List[dict]) -> tuple:
        """把 [{features, label}, ...] 拆成 (X: np.ndarray, y: np.ndarray)。

        验证:行非空、特征长度一致;否则抛 ValueError。
        """
        if not rows:
            raise ValueError("Empty training rows; cannot fit LearningToRanker.")
        feature_lens = {len(r["features"]) for r in rows if "features" in r and r["features"]}
        if len(feature_lens) != 1:
            raise ValueError(
                f"Inconsistent feature lengths in training rows: {feature_lens}; "
                f"all rows must have the same feature vector length."
            )
        X = np.asarray([r["features"] for r in rows], dtype=np.float64)
        y = np.asarray([r["label"] for r in rows], dtype=np.int64)
        return X, y

    def fit(self, rows: List[dict]) -> "LearningToRanker":
        X, y = self._extract_xy(rows)
        self._feature_dim = X.shape[1]

        if self._backend == "sklearn":
            self._model = _SklearnLR(
                random_state=self.seed,
                max_iter=1000,
                solver="lbfgs",
            )
            self._model.fit(X, y)
        else:
            # numpy fallback:小批量梯度下降的 logistic regression
            self._model = _NumpyLogisticRegression(
                n_features=X.shape[1],
                learning_rate=0.1,
                n_iterations=2000,
                seed=self.seed,
            )
            self._model.fit(X, y)
        return self

    def predict_scores(self, rows: List[dict]) -> List[float]:
        """返回每行 score(类 1 的概率)。未 fit 时抛 RuntimeError。"""
        if self._model is None:
            raise RuntimeError(
                "LearningToRanker must be fit before predict_scores; call .fit(rows) first."
            )
        if not rows:
            return []
        X = np.asarray([r["features"] for r in rows], dtype=np.float64)
        if self._feature_dim is not None and X.shape[1] != self._feature_dim:
            raise ValueError(
                f"Feature dim mismatch: trained on {self._feature_dim}, "
                f"got {X.shape[1]}"
            )
        if self._backend == "sklearn":
            # predict_proba 返回 (n, 2),取第 2 列(类 1)
            probs = self._model.predict_proba(X)[:, 1]
        else:
            probs = self._model.predict_proba(X)
        return [float(p) for p in probs]

    # ---- 序列化 ----

    def save(self, path: str) -> None:
        """序列化模型到 JSON 路径(weights + bias + 元信息)。"""
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted LearningToRanker.")
        if self._backend == "sklearn":
            coef = self._model.coef_[0].tolist()
            intercept = float(self._model.intercept_[0])
        else:
            coef = self._model.weights.tolist()
            intercept = float(self._model.bias)
        payload = {
            "schema_version": 1,
            "backend": self._backend,
            "seed": self.seed,
            "feature_dim": self._feature_dim,
            "coef": coef,
            "intercept": intercept,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LearningToRanker":
        """从 JSON 路径反序列化模型。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ranker = cls(seed=int(payload.get("seed", 42)))
        ranker._feature_dim = int(payload["feature_dim"])
        ranker._backend = str(payload["backend"])
        coef = np.asarray(payload["coef"], dtype=np.float64)
        intercept = float(payload["intercept"])
        # 重新构造 backend 模型,塞入权重;推理时直接走 numpy 路径,避免依赖 sklearn
        ranker._model = _NumpyLogisticRegression.from_weights(
            n_features=len(coef), weights=coef, bias=intercept
        )
        # 切到 numpy backend(即使原 backend 是 sklearn,推理时也用 numpy 实现避免循环依赖)
        ranker._backend = "numpy"
        return ranker


class _NumpyLogisticRegression:
    """numpy 实现的 logistic regression(确定性,seed 锁定)。

    sklearn 不可用时 fallback;load() 时也走这条路径(避免引入 sklearn 推理依赖)。
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        n_iterations: int = 2000,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.seed = seed
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # 数值稳定:clip 避免 exp 溢出
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyLogisticRegression":
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=self.n_features)
        self.bias = 0.0
        n = X.shape[0]
        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)
            grad_w = (X.T @ (p - y)) / n
            grad_b = float(np.mean(p - y))
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    @classmethod
    def from_weights(cls, n_features: int, weights: np.ndarray, bias: float) -> "_NumpyLogisticRegression":
        obj = cls(n_features=n_features, seed=0)
        obj.weights = np.asarray(weights, dtype=np.float64)
        obj.bias = float(bias)
        return obj

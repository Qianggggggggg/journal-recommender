"""Learning-to-Rank reranker (Task 5.1 baseline + Task 5.2 convergence handling).

第一版用 sklearn LogisticRegression(优先,per plan 5.1)。
sklearn 不可用时 fallback 到 numpy 实现的小型 logistic regression。

纪律:
- **确定性**:`random_state=seed` 锁定,同 seed + 同数据 → bit-equal 预测。
- **无新增依赖**:sklearn 已是项目隐式依赖(没在 pyproject.toml 显式列出,但
  conda 环境里有);numpy 是必需的(已用)。
- **可序列化**:save/load 用 JSON,人类可读,便于 git diff 与回归测试。
- **不接受坏数据**:空训练集 / 特征长度不一致必须抛 ValueError。
- **Task 5.2 收敛处理**:plan 5.2 重点 — 真实 ranker_train_full_v2.jsonl 上
  lbfgs 在默认 1000 iter 内无法收敛(lbfgs 对未标准化的 rank 特征梯度爆炸)。
  ranker 暴露 `max_iter` 和 `use_standardization` 开关,并把收敛状态写入
  `convergence_info`,不静默吞 ConvergenceWarning。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# sklearn 是首选(更稳、有 calibration),不可用时降级到 numpy
try:  # pragma: no cover
    from sklearn.exceptions import ConvergenceWarning as _SklearnConvergenceWarning
    from sklearn.linear_model import LogisticRegression as _SklearnLR
    from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False
    _SklearnConvergenceWarning = Warning  # type: ignore[assignment,misc]


# sklearn 在 1.3 之前是 sklearn.utils.ConvergenceWarning,1.4 之后是
# sklearn.exceptions.ConvergenceWarning;两边都是 Warning 子类,fallback 即可。
class _NumpyStandardScaler:
    """numpy 实现的 StandardScaler(只用于 save/load 反序列化,fit 用 sklearn 原生)。

    训练时优先用 sklearn.preprocessing.StandardScaler;它处理方差为 0 的特征
    不会崩(np.nan 行为),与 numpy 行为一致。推理端为了避免循环依赖,这里
    提供纯 numpy 等价物。
    """

    def __init__(self, mean: Optional[np.ndarray] = None, scale: Optional[np.ndarray] = None) -> None:
        self.mean = mean
        self.scale = scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler not fitted.")
        # scale 为 0 的维度保持 0(与 sklearn 行为一致)
        safe_scale = np.where(self.scale == 0, 1.0, self.scale)
        return (X - self.mean) / safe_scale

    def to_dict(self) -> dict:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_NumpyStandardScaler":
        mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=np.float64)
        scale = None if d.get("scale") is None else np.asarray(d["scale"], dtype=np.float64)
        return cls(mean=mean, scale=scale)


class LearningToRanker:
    """LTR baseline:logistic regression over paper-candidate features。

    训练目标:预测给定 (paper_features, journal_features) → label ∈ {0, 1}。
    推理:`predict_scores` 返回每行 score(类 1 的概率,越大越像 gold)。

    Task 5.2 收敛处理:
    - `max_iter`: lbfgs 最大迭代次数,默认 1000(plan 5.1)。
      真实数据需要 ≥ 5000 才能收敛。
    - `use_standardization`: 是否在 fit/predict 前对特征做 z-score 标准化。
      真实数据特征尺度差异巨大(0/1 布尔 + 0~999 rank),不标准化会让
      lbfgs 永远跑不到最优。建议默认 True。
    - `convergence_info`: 训练结束后暴露的收敛状态,结构:
        {"converged": bool, "n_iter": int|None, "max_iter": int, "warning_message": str|None}
      下游训练脚本必须把它写入产物 JSON,便于回归检测。
    """

    def __init__(
        self,
        seed: int = 42,
        max_iter: int = 1000,
        use_standardization: bool = False,
    ) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.use_standardization = use_standardization
        self._model: Any = None
        self._feature_dim: Optional[int] = None
        self._scaler: Optional[_NumpyStandardScaler] = None
        self._backend: str = "sklearn" if _HAS_SKLEARN else "numpy"
        self.convergence_info: Optional[Dict[str, Any]] = None

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

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return X
        return self._scaler.transform(X)

    def fit(self, rows: List[dict]) -> "LearningToRanker":
        X, y = self._extract_xy(rows)
        self._feature_dim = X.shape[1]

        # 特征标准化(Task 5.2 收敛处理)
        if self.use_standardization and _HAS_SKLEARN:
            sklearn_scaler = _SklearnStandardScaler()
            X = sklearn_scaler.fit_transform(X)
            self._scaler = _NumpyStandardScaler(
                mean=sklearn_scaler.mean_,
                scale=sklearn_scaler.scale_,
            )
        else:
            self._scaler = None

        if self._backend == "sklearn":
            self.convergence_info = {
                "converged": True,
                "n_iter": None,
                "max_iter": self.max_iter,
                "warning_message": None,
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", _SklearnConvergenceWarning)
                self._model = _SklearnLR(
                    random_state=self.seed,
                    max_iter=self.max_iter,
                    solver="lbfgs",
                )
                self._model.fit(X, y)
            # 检查 ConvergenceWarning
            conv_warnings = [
                str(w.message) for w in caught
                if issubclass(w.category, _SklearnConvergenceWarning)
            ]
            if conv_warnings:
                self.convergence_info["converged"] = False
                self.convergence_info["warning_message"] = conv_warnings[0]
            # sklearn 把实际迭代数藏在 n_iter_[0]
            n_iter = getattr(self._model, "n_iter_", [None])[0]
            self.convergence_info["n_iter"] = int(n_iter) if n_iter is not None else None
            # 已知：n_iter 达到 max_iter 但仍报 warning → 一律标记为未收敛
            if self.convergence_info["n_iter"] is not None and self.convergence_info["n_iter"] >= self.max_iter:
                self.convergence_info["converged"] = False
        else:
            # numpy fallback:小批量梯度下降的 logistic regression
            self._model = _NumpyLogisticRegression(
                n_features=X.shape[1],
                learning_rate=0.1,
                n_iterations=2000,
                seed=self.seed,
            )
            self._model.fit(X, y)
            self.convergence_info = {
                "converged": True,
                "n_iter": 2000,
                "max_iter": self.max_iter,
                "warning_message": None,
            }
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
        X = self._transform(X)
        if self._backend == "sklearn":
            # predict_proba 返回 (n, 2),取第 2 列(类 1)
            probs = self._model.predict_proba(X)[:, 1]
        else:
            probs = self._model.predict_proba(X)
        return [float(p) for p in probs]

    # ---- 序列化 ----

    def save(self, path: str) -> None:
        """序列化模型到 JSON 路径(weights + bias + scaler + 元信息)。"""
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted LearningToRanker.")
        if self._backend == "sklearn":
            coef = self._model.coef_[0].tolist()
            intercept = float(self._model.intercept_[0])
        else:
            coef = self._model.weights.tolist()
            intercept = float(self._model.bias)
        scaler_dict = self._scaler.to_dict() if self._scaler is not None else {"mean": None, "scale": None}
        payload = {
            "schema_version": 1,
            "backend": self._backend,
            "seed": self.seed,
            "max_iter": self.max_iter,
            "use_standardization": self.use_standardization,
            "feature_dim": self._feature_dim,
            "coef": coef,
            "intercept": intercept,
            "scaler_mean": scaler_dict["mean"],
            "scaler_scale": scaler_dict["scale"],
            "convergence_info": self.convergence_info,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LearningToRanker":
        """从 JSON 路径反序列化模型。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ranker = cls(
            seed=int(payload.get("seed", 42)),
            max_iter=int(payload.get("max_iter", 1000)),
            use_standardization=bool(payload.get("use_standardization", False)),
        )
        ranker._feature_dim = int(payload["feature_dim"])
        ranker._backend = str(payload["backend"])
        coef = np.asarray(payload["coef"], dtype=np.float64)
        intercept = float(payload["intercept"])
        ranker.convergence_info = payload.get("convergence_info")
        # 恢复 scaler
        if payload.get("scaler_mean") is not None and payload.get("scaler_scale") is not None:
            ranker._scaler = _NumpyStandardScaler(
                mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
                scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
            )
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


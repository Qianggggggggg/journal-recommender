"""动态门控网络：根据 PaperProfile 预测三路召回权重。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..papers.paper_model import PaperProfile


ROUTE_NAMES = ("bm25", "vector", "text")


class PaperProfileFeaturizer:
    """轻量 profile featurizer，避免训练脚本强依赖外部 embedding 服务。"""

    def __init__(self, hash_dim: int = 128):
        self.hash_dim = hash_dim
        self.research_areas = [
            "计算机体系结构/并行与分布计算/存储系统",
            "计算机网络",
            "网络与信息安全",
            "软件工程/系统软件/程序设计语言",
            "数据库/数据挖掘/内容检索",
            "计算机科学理论",
            "计算机图形学与多媒体",
            "人工智能",
            "人机交互与普适计算",
            "交叉/综合/新兴",
        ]
        self.method_types = ["method", "system", "experiment", "survey", "theory", "application"]
        self.novelty_types = ["new_method", "new_application", "benchmark", "performance", "efficiency", "theory", "survey", "system"]

    @property
    def output_dim(self) -> int:
        return self.hash_dim + len(self.research_areas) + len(self.method_types) + len(self.novelty_types) + 5

    def transform(self, profile: PaperProfile) -> np.ndarray:
        vec = np.zeros(self.output_dim, dtype=np.float32)
        offset = 0

        text = " ".join([
            profile.title or "",
            profile.abstract or "",
            " ".join(profile.keywords),
            " ".join(profile.techniques),
            " ".join(profile.datasets),
            " ".join(profile.evaluation_metrics),
        ])
        for token in _tokens(text):
            idx = _stable_hash(token) % self.hash_dim
            vec[idx] += 1.0
        if vec[:self.hash_dim].sum() > 0:
            vec[:self.hash_dim] /= max(np.linalg.norm(vec[:self.hash_dim]), 1e-6)
        offset += self.hash_dim

        for area in profile.research_area:
            if area in self.research_areas:
                vec[offset + self.research_areas.index(area)] = 1.0
        offset += len(self.research_areas)

        if profile.method_type in self.method_types:
            vec[offset + self.method_types.index(profile.method_type)] = 1.0
        offset += len(self.method_types)

        if profile.novelty_type in self.novelty_types:
            vec[offset + self.novelty_types.index(profile.novelty_type)] = 1.0
        offset += len(self.novelty_types)

        vec[offset:offset + 5] = np.array([
            min(len(profile.title or "") / 200.0, 1.0),
            min(len(profile.abstract or "") / 2000.0, 1.0),
            min(len(profile.keywords) / 10.0, 1.0),
            min(len(profile.techniques) / 10.0, 1.0),
            min(len(profile.datasets) / 10.0, 1.0),
        ], dtype=np.float32)
        return vec

    def to_config(self) -> dict:
        return {"hash_dim": self.hash_dim}

    @classmethod
    def from_config(cls, config: dict) -> "PaperProfileFeaturizer":
        return cls(hash_dim=int(config.get("hash_dim", 128)))


class GatingNetwork(nn.Module):
    """全连接 + softmax 的三路权重预测网络。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ROUTE_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(features), dim=-1)


class DynamicWeightGater:
    """CandidateGenerator 使用的推理包装器。"""

    def __init__(self, model: GatingNetwork, featurizer: PaperProfileFeaturizer):
        self.model = model
        self.featurizer = featurizer
        self.model.eval()

    def predict_weights(self, profile: PaperProfile) -> Dict[str, float]:
        features = torch.tensor(self.featurizer.transform(profile), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            weights = self.model(features).squeeze(0).cpu().numpy()
        return {name: float(weights[idx]) for idx, name in enumerate(ROUTE_NAMES)}

    @classmethod
    def load(cls, path: str) -> "DynamicWeightGater":
        payload = torch.load(path, map_location="cpu")
        featurizer = PaperProfileFeaturizer.from_config(payload.get("featurizer", {}))
        model = GatingNetwork(featurizer.output_dim, hidden_dim=payload.get("hidden_dim", 64))
        model.load_state_dict(payload["state_dict"])
        return cls(model, featurizer)

    def save(self, path: str, hidden_dim: int = 64) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "featurizer": self.featurizer.to_config(),
            "hidden_dim": hidden_dim,
        }, path)


def best_weight_label(
    route_rankings: Dict[str, Sequence[str]],
    positive_journal_id: str,
    grid_step: float = 0.1,
) -> Tuple[float, float, float]:
    """网格搜索三路权重，最大化正例融合排名。"""
    route_scores = {}
    for route, ranking in route_rankings.items():
        route_scores[route] = {
            jid: 1.0 / (rank + 1)
            for rank, jid in enumerate(ranking)
        }

    candidate_ids = set()
    for scores in route_scores.values():
        candidate_ids.update(scores.keys())

    best_weights = (1.0, 0.0, 0.0)
    best_rank = float("inf")
    steps = int(round(1.0 / grid_step))
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            weights = (i / steps, j / steps, k / steps)
            fused = []
            for jid in candidate_ids:
                score = (
                    weights[0] * route_scores.get("bm25", {}).get(jid, 0.0)
                    + weights[1] * route_scores.get("vector", {}).get(jid, 0.0)
                    + weights[2] * route_scores.get("text", {}).get(jid, 0.0)
                )
                fused.append((jid, score))
            fused.sort(key=lambda x: x[1], reverse=True)
            rank = next((idx for idx, (jid, _) in enumerate(fused) if jid == positive_journal_id), float("inf"))
            if rank < best_rank:
                best_rank = rank
                best_weights = weights
    return best_weights


def train_gating_network(
    profiles: Sequence[PaperProfile],
    labels: Sequence[Sequence[float]],
    epochs: int = 100,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    hash_dim: int = 128,
) -> DynamicWeightGater:
    featurizer = PaperProfileFeaturizer(hash_dim=hash_dim)
    model = GatingNetwork(featurizer.output_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    x = torch.tensor(np.stack([featurizer.transform(p) for p in profiles]), dtype=torch.float32)
    y = torch.tensor(np.array(labels, dtype=np.float32), dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x).clamp_min(1e-8)
        loss = loss_fn(pred.log(), y)
        loss.backward()
        optimizer.step()

    return DynamicWeightGater(model, featurizer)


def _tokens(text: str) -> List[str]:
    return [tok.lower() for tok in text.replace("-", " ").split() if len(tok) > 2]


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)

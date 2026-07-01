"""交叉编码器精排与 LambdaRank pairwise 损失。"""
from __future__ import annotations

import hashlib
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class HashingPairEncoder:
    """无 DeBERTa 权重时的轻量 pair encoder。"""

    def __init__(self, output_dim: int = 512):
        self.output_dim = output_dim

    def encode(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        vectors = torch.zeros(len(pairs), self.output_dim, dtype=torch.float32)
        half = self.output_dim // 2
        for row, (paper_text, journal_text) in enumerate(pairs):
            for token in _tokens(paper_text):
                vectors[row, _stable_hash("p:" + token) % half] += 1.0
            for token in _tokens(journal_text):
                vectors[row, half + (_stable_hash("j:" + token) % (self.output_dim - half))] += 1.0
        return F.normalize(vectors, dim=-1)


class TransformerPairEncoder:
    """DeBERTa-tiny / BERT-mini 交叉编码器 backbone。"""

    def __init__(self, model_name: str = "microsoft/deberta-v3-small", local_files_only: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()
        self.output_dim = int(self.model.config.hidden_size)

    def encode(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        with torch.no_grad():
            batch = self.tokenizer(
                [p for p, _ in pairs],
                [j for _, j in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = self.model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return F.normalize(pooled, dim=-1)


def make_pair_encoder(
    model_name: str = "microsoft/deberta-v3-small",
    fallback_dim: int = 512,
    local_files_only: bool = True,
):
    try:
        return TransformerPairEncoder(model_name=model_name, local_files_only=local_files_only)
    except Exception:
        return HashingPairEncoder(output_dim=fallback_dim)


class CrossEncoderRanker(nn.Module):
    """轻量交叉编码器打分头。"""

    def __init__(self, pair_encoder=None, encoder_dim: int | None = None, hidden_dim: int = 128):
        super().__init__()
        self.pair_encoder = pair_encoder or HashingPairEncoder()
        encoder_dim = encoder_dim or self.pair_encoder.output_dim
        self.scorer = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        encoded = self.pair_encoder.encode(pairs)
        return self.scorer(encoded).squeeze(-1)


def lambda_rank_pairwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """LambdaRank 风格 pairwise logistic loss。

    labels 越大表示相关性越高；只对 label_i > label_j 的 pair 计算损失。
    """
    losses = []
    order = torch.argsort(labels, descending=True)
    ideal_dcg = _dcg(labels[order]).clamp_min(1e-6)
    for i in range(len(scores)):
        for j in range(len(scores)):
            if labels[i] <= labels[j]:
                continue
            delta_ndcg = torch.abs(labels[i] - labels[j]) / ideal_dcg
            losses.append(delta_ndcg * F.softplus(-(scores[i] - scores[j])))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def build_hard_negative_pairs(
    paper_text: str,
    positive_journal_text: str,
    negative_journal_texts: Sequence[str],
) -> tuple[list[tuple[str, str]], torch.Tensor]:
    pairs = [(paper_text, positive_journal_text)]
    labels = [1.0]
    for text in negative_journal_texts:
        pairs.append((paper_text, text))
        labels.append(0.0)
    return pairs, torch.tensor(labels, dtype=torch.float32)


def _dcg(labels: torch.Tensor) -> torch.Tensor:
    gains = torch.pow(2.0, labels) - 1.0
    discounts = torch.log2(torch.arange(len(labels), device=labels.device, dtype=torch.float32) + 2.0)
    return (gains / discounts).sum()


def _tokens(text: str) -> List[str]:
    return [tok.lower() for tok in text.replace("-", " ").split() if len(tok) > 2]


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)

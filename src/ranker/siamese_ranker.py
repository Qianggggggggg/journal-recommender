"""双塔粗排模型：论文塔 + 典型摘要期刊塔。"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class HashingTextEncoder:
    """无外部模型权重时使用的确定性文本编码器。"""

    def __init__(self, output_dim: int = 256):
        self.output_dim = output_dim

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = torch.zeros(len(texts), self.output_dim, dtype=torch.float32)
        for row, text in enumerate(texts):
            for token in _tokens(text):
                vectors[row, _stable_hash(token) % self.output_dim] += 1.0
        return F.normalize(vectors, dim=-1)


class TransformerTextEncoder:
    """SciBERT/SPECTER 文本编码器；本地无权重时由调用方 fallback。"""

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", local_files_only: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()
        self.output_dim = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            batch = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = self.model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return F.normalize(pooled, dim=-1)


def make_text_encoder(
    model_name: str = "allenai/scibert_scivocab_uncased",
    fallback_dim: int = 256,
    local_files_only: bool = True,
):
    """优先加载 SciBERT/SPECTER，本地不可用时回退到 hashing encoder。"""
    try:
        return TransformerTextEncoder(model_name=model_name, local_files_only=local_files_only)
    except Exception:
        return HashingTextEncoder(output_dim=fallback_dim)


class JournalAnchorTower(nn.Module):
    """4 篇典型摘要编码后的注意力池化期刊塔。"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, anchor_embeddings: torch.Tensor) -> torch.Tensor:
        # anchor_embeddings: [batch, anchors, dim]
        weights = torch.softmax(self.attention(anchor_embeddings).squeeze(-1), dim=-1)
        pooled = (anchor_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(pooled, dim=-1)


class TwoTowerRanker(nn.Module):
    """论文/期刊伪孪生粗排模型。"""

    def __init__(
        self,
        text_encoder=None,
        encoder_dim: Optional[int] = None,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.text_encoder = text_encoder or HashingTextEncoder()
        encoder_dim = encoder_dim or self.text_encoder.output_dim
        self.paper_projection = nn.Linear(encoder_dim, projection_dim)
        self.anchor_projection = nn.Linear(encoder_dim, projection_dim)
        self.journal_tower = JournalAnchorTower(projection_dim)

    def encode_papers(self, paper_texts: Sequence[str]) -> torch.Tensor:
        encoded = self.text_encoder.encode(paper_texts)
        return F.normalize(self.paper_projection(encoded), dim=-1)

    def encode_journals(self, journal_anchor_texts: Sequence[Sequence[str]]) -> torch.Tensor:
        flat = [text for group in journal_anchor_texts for text in group]
        group_sizes = [len(group) for group in journal_anchor_texts]
        if not flat:
            raise ValueError("journal_anchor_texts cannot be empty")
        encoded = F.normalize(self.anchor_projection(self.text_encoder.encode(flat)), dim=-1)
        groups = []
        cursor = 0
        max_size = max(group_sizes)
        for size in group_sizes:
            group = encoded[cursor:cursor + size]
            cursor += size
            if size < max_size:
                pad = group[-1:].repeat(max_size - size, 1)
                group = torch.cat([group, pad], dim=0)
            groups.append(group)
        return self.journal_tower(torch.stack(groups, dim=0))

    def forward(self, paper_texts: Sequence[str], journal_anchor_texts: Sequence[Sequence[str]]) -> torch.Tensor:
        paper_embeddings = self.encode_papers(paper_texts)
        journal_embeddings = self.encode_journals(journal_anchor_texts)
        return paper_embeddings @ journal_embeddings.T


def info_nce_loss(paper_embeddings: torch.Tensor, journal_embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """批内负样本 InfoNCE，对角线为正例。"""
    logits = paper_embeddings @ journal_embeddings.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def _tokens(text: str) -> List[str]:
    return [tok.lower() for tok in text.replace("-", " ").split() if len(tok) > 2]


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)

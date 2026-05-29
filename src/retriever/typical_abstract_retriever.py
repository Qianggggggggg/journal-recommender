"""基于典型摘要库的 BM25 / 向量 / 文本召回。"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rank_bm25

from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore
from ..journals.typical_abstract_store import (
    TypicalAbstractStore,
    aggregate_anchor_scores,
)
from ..utils.embedding import OllamaEmbedding


class TypicalAbstractBM25Retriever:
    """以典型摘要为文档，检索后聚合到期刊。"""

    def __init__(
        self,
        abstract_store: TypicalAbstractStore,
        journal_store: JournalStore,
        aggregate_mode: str = "max",
    ):
        self.abstract_store = abstract_store
        self.journal_store = journal_store
        self.aggregate_mode = aggregate_mode
        self._bm25_index: Optional[rank_bm25.BM25Okapi] = None
        self._built = False

    def build_index(self) -> None:
        docs = [r.search_text.lower().split() for r in self.abstract_store.records]
        if not docs:
            return
        self._bm25_index = rank_bm25.BM25Okapi(docs)
        self._built = True

    def retrieve(self, query: str, top_k: int = 30, anchor_k: Optional[int] = None) -> List[Tuple[Journal, float]]:
        if not self._built:
            self.build_index()
        if self._bm25_index is None:
            return []

        anchor_k = anchor_k or max(top_k * 4, top_k)
        scores = self._bm25_index.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:anchor_k]
        scored_records = [
            (self.abstract_store.records[idx], float(scores[idx]))
            for idx in top_indices
            if idx < len(self.abstract_store.records)
        ]
        return self._journals_from_anchor_scores(scored_records, top_k=top_k)

    def _journals_from_anchor_scores(
        self,
        scored_records,
        top_k: int,
    ) -> List[Tuple[Journal, float]]:
        score_map = aggregate_anchor_scores(scored_records, self.aggregate_mode)
        sorted_ids = sorted(score_map, key=lambda jid: score_map[jid], reverse=True)[:top_k]
        results = []
        for jid in sorted_ids:
            journal = self.journal_store.get_journal(jid)
            if journal:
                results.append((journal, score_map[jid]))
        return results


class TypicalAbstractTextRetriever:
    """典型摘要关键词交集召回。"""

    def __init__(
        self,
        abstract_store: TypicalAbstractStore,
        journal_store: JournalStore,
        aggregate_mode: str = "max",
    ):
        self.abstract_store = abstract_store
        self.journal_store = journal_store
        self.aggregate_mode = aggregate_mode

    def retrieve(self, query: str, top_k: int = 30, anchor_k: Optional[int] = None) -> List[Tuple[Journal, float]]:
        anchor_k = anchor_k or max(top_k * 4, top_k)
        records = self.abstract_store.search_by_text(query, top_k=anchor_k)
        score_map = aggregate_anchor_scores(records, self.aggregate_mode)
        sorted_ids = sorted(score_map, key=lambda jid: score_map[jid], reverse=True)[:top_k]
        results = []
        for jid in sorted_ids:
            journal = self.journal_store.get_journal(jid)
            if journal:
                results.append((journal, score_map[jid]))
        return results


class TypicalAbstractEmbeddingRetriever:
    """从典型摘要 FAISS 索引检索，按 journal_id 聚合。"""

    def __init__(
        self,
        abstract_store: TypicalAbstractStore,
        journal_store: JournalStore,
        embedding_client: OllamaEmbedding,
        faiss_path: str = "data/processed/typical_abstracts_index.faiss",
        metadata_path: str = "data/processed/typical_abstracts_metadata.parquet",
        aggregate_mode: str = "max",
    ):
        self.abstract_store = abstract_store
        self.journal_store = journal_store
        self.embedding_client = embedding_client
        self.faiss_path = faiss_path
        self.metadata_path = metadata_path
        self.aggregate_mode = aggregate_mode
        self._index = None
        self._metadata: Optional[pd.DataFrame] = None
        self.load()

    def load(self) -> None:
        if not Path(self.faiss_path).exists() or not Path(self.metadata_path).exists():
            return
        import faiss

        self._index = faiss.read_index(self.faiss_path)
        self._metadata = pd.read_parquet(self.metadata_path)

    @property
    def is_available(self) -> bool:
        return self._index is not None and self._metadata is not None

    def retrieve(self, query_text: str, top_k: int = 30, anchor_k: Optional[int] = None) -> List[Tuple[Journal, float]]:
        if not self.is_available:
            return []

        anchor_k = anchor_k or max(top_k * 4, top_k)
        query_embedding = self.embedding_client.embed(query_text).reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_embedding, min(anchor_k, self._index.ntotal))

        score_map: Dict[str, float] = {}
        count_map: Dict[str, int] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            row = self._metadata.iloc[int(idx)]
            jid = row["journal_id"]
            score = -float(dist)
            if self.aggregate_mode == "sum":
                score_map[jid] = score_map.get(jid, 0.0) + score
            elif self.aggregate_mode == "mean":
                score_map[jid] = score_map.get(jid, 0.0) + score
                count_map[jid] = count_map.get(jid, 0) + 1
            else:
                score_map[jid] = max(score_map.get(jid, float("-inf")), score)

        if self.aggregate_mode == "mean":
            score_map = {jid: score / max(count_map.get(jid, 1), 1) for jid, score in score_map.items()}

        sorted_ids = sorted(score_map, key=lambda jid: score_map[jid], reverse=True)[:top_k]
        results = []
        for jid in sorted_ids:
            journal = self.journal_store.get_journal(jid)
            if journal:
                results.append((journal, score_map[jid]))
        return results

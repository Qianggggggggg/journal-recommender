"""基于真实已发表论文画像的 BM25 / 向量召回。

任务 3.1 落地 ``AcceptedPaperBM25Retriever``:把 ``data/accepted_papers/*.json``
里每一篇真实论文当作 BM25 文档建索引,检索时按期刊聚合,得分 = ``max(per-paper
score) + min(0.05 * matching_paper_count, bonus_cap)``。

任务 3.2 落地 ``AcceptedPaperEmbeddingRetriever``:从预先构建好的 FAISS
索引 + parquet metadata 中读取,用 OllamaEmbedding 编码 query 后做相似度
搜索,然后用与 BM25 一致的"max + capped count bonus"聚合规则把 paper-level
score 聚合到 journal-level。

两个 retriever 都暴露 ``last_route_details``,字段集合相同,便于上层
CandidateGenerator 写 retrieval trace 用于实验诊断。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rank_bm25

from ..journals.accepted_paper_store import AcceptedPaperStore
from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore


class AcceptedPaperBM25Retriever:
    """以真实已发表论文为文档,按期刊聚合 BM25 召回。"""

    def __init__(
        self,
        accepted_store: AcceptedPaperStore,
        journal_store,  # 实际为 JournalStore,test 中允许 fake 兼容
        paper_count_bonus: float = 0.05,
        bonus_cap: float = 0.3,
    ):
        self.accepted_store = accepted_store
        self.journal_store = journal_store
        self.paper_count_bonus = paper_count_bonus
        self.bonus_cap = bonus_cap
        self._bm25_index: Optional[rank_bm25.BM25Okapi] = None
        self._built = False
        # 每次 retrieve 后填充,供 CandidateGenerator 取详情写 trace
        self.last_route_details: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """对每条 AcceptedPaperRecord 的 title+abstract 建 BM25 索引。"""
        docs = [r.search_text.lower().split() for r in self.accepted_store.records]
        if not docs:
            return
        self._bm25_index = rank_bm25.BM25Okapi(docs)
        self._built = True

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        anchor_k: Optional[int] = None,
    ) -> List[Tuple[Journal, float]]:
        """主检索接口。

        anchor_k 控制每次取多少 paper-level top 进入期刊级聚合;默认
        ``max(top_k * 4, top_k)``,与 TypicalAbstract 保持一致。
        """
        if not self._built:
            self.build_index()
        if self._bm25_index is None:
            return []

        anchor_k = anchor_k or max(top_k * 4, top_k)
        scores = self._bm25_index.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:anchor_k]

        records = self.accepted_store.records

        per_top_score: Dict[str, float] = {}
        per_top_title: Dict[str, str] = {}
        per_count: Dict[str, int] = {}
        for idx in top_indices:
            if idx >= len(records):
                continue
            score = float(scores[idx])
            record = records[idx]
            jid = record.journal_id
            if jid not in per_top_score or score > per_top_score[jid]:
                per_top_score[jid] = score
                per_top_title[jid] = record.title
            per_count[jid] = per_count.get(jid, 0) + 1

        # 聚合 + 写 route_details
        self.last_route_details = {}
        final_score_map: Dict[str, float] = {}
        for jid, top_score in per_top_score.items():
            count = per_count[jid]
            raw_bonus = self.paper_count_bonus * count
            bonus = min(raw_bonus, self.bonus_cap)
            final_score = top_score + bonus
            final_score_map[jid] = final_score
            self.last_route_details[jid] = {
                "top_paper_title": per_top_title[jid],
                "top_paper_score": top_score,
                "matching_paper_count": count,
                "bonus": bonus,
                "final_score": final_score,
            }

        sorted_ids = sorted(
            final_score_map, key=lambda j: final_score_map[j], reverse=True
        )[:top_k]

        results: List[Tuple[Journal, float]] = []
        for jid in sorted_ids:
            journal = self.journal_store.get_journal(jid)
            if journal is None:
                continue
            results.append((journal, final_score_map[jid]))
        return results


class AcceptedPaperEmbeddingRetriever:
    """从预构建的 accepted-paper FAISS 索引检索,按期刊聚合。

    与 ``AcceptedPaperBM25Retriever`` 共用聚合规则:max(paper similarity) +
    min(0.05 * matching_paper_count, bonus_cap)。``last_route_details`` 字段集
    与 BM25 retriever 完全一致,CandidateGenerator 可以统一写 trace。
    """

    def __init__(
        self,
        accepted_store: AcceptedPaperStore,
        journal_store,
        embedding_client,
        faiss_path: str = "data/processed/accepted_papers_index.faiss",
        metadata_path: str = "data/processed/accepted_papers_metadata.parquet",
        paper_count_bonus: float = 0.05,
        bonus_cap: float = 0.3,
    ):
        self.accepted_store = accepted_store
        self.journal_store = journal_store
        self.embedding_client = embedding_client
        self.faiss_path = faiss_path
        self.metadata_path = metadata_path
        self.paper_count_bonus = paper_count_bonus
        self.bonus_cap = bonus_cap
        self._index = None
        self._metadata: Optional[pd.DataFrame] = None
        self.last_route_details: Dict[str, Dict] = {}
        self.load()

    def load(self) -> None:
        """加载 FAISS 索引和 parquet metadata。文件缺失时静默跳过,
        ``is_available`` 后续返回 False。"""
        if not Path(self.faiss_path).exists() or not Path(self.metadata_path).exists():
            return
        import faiss

        self._index = faiss.read_index(self.faiss_path)
        self._metadata = pd.read_parquet(self.metadata_path)

    @property
    def is_available(self) -> bool:
        return self._index is not None and self._metadata is not None

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        anchor_k: Optional[int] = None,
    ) -> List[Tuple[Journal, float]]:
        if not self.is_available:
            return []

        anchor_k = anchor_k or max(top_k * 4, top_k)
        anchor_k = min(anchor_k, self._index.ntotal)
        if anchor_k <= 0:
            return []

        query_embedding = (
            self.embedding_client.embed(query).reshape(1, -1).astype(np.float32)
        )
        distances, indices = self._index.search(query_embedding, anchor_k)

        per_top_score: Dict[str, float] = {}
        per_top_title: Dict[str, str] = {}
        per_count: Dict[str, int] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            row = self._metadata.iloc[int(idx)]
            jid = str(row["journal_id"])
            # FAISS L2 距离,转成 similarity (取负)。BM25 用绝对分数,
            # embedding 用 -distance,两边都是"越大越相关"。
            score = -float(dist)
            title = str(row.get("title", ""))
            if jid not in per_top_score or score > per_top_score[jid]:
                per_top_score[jid] = score
                per_top_title[jid] = title
            per_count[jid] = per_count.get(jid, 0) + 1

        self.last_route_details = {}
        final_score_map: Dict[str, float] = {}
        for jid, top_score in per_top_score.items():
            count = per_count[jid]
            bonus = min(self.paper_count_bonus * count, self.bonus_cap)
            final_score = top_score + bonus
            final_score_map[jid] = final_score
            self.last_route_details[jid] = {
                "top_paper_title": per_top_title[jid],
                "top_paper_score": top_score,
                "matching_paper_count": count,
                "bonus": bonus,
                "final_score": final_score,
            }

        sorted_ids = sorted(
            final_score_map, key=lambda j: final_score_map[j], reverse=True
        )[:top_k]

        results: List[Tuple[Journal, float]] = []
        for jid in sorted_ids:
            journal = self.journal_store.get_journal(jid)
            if journal is None:
                continue
            results.append((journal, final_score_map[jid]))
        return results

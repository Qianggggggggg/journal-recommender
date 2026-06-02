"""基于真实已发表论文画像的 BM25 召回。

任务 3.1 落地:把 ``data/accepted_papers/*.json`` 里每一篇真实论文当作 BM25
文档建索引,检索时按期刊聚合,得分 = ``max(per-paper score) + min(0.05 *
matching_paper_count, bonus_cap)``。这样既保留了"最相关那一篇"的强信号,
又给"画像里多篇命中"的期刊一点小奖励,但通过 cap 控制 runaway。

与 ``TypicalAbstractBM25Retriever`` 的差异:
- 文档来源不是 LLM 生成的"典型摘要",而是真实发表的 title + abstract,
  天然包含期刊的研究风格、术语和方法偏好。
- 聚合用 ``max + bonus``,不再用 ``aggregate_anchor_scores`` 中的 max/sum/mean
  三选一——因为画像论文数差异极大 (有的期刊 1 篇有的 50 篇),max 比 sum
  更稳,但加一点 count bonus 鼓励"画像丰富"的期刊。
- 每次检索后 ``last_route_details`` 暴露 (top matching paper title, score,
  matching paper count, bonus, final score),供 CandidateGenerator 写入
  retrieval trace 用于实验诊断。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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

"""期刊存储与检索"""
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import faiss
import numpy as np
import pandas as pd

from .journal_model import Journal

if TYPE_CHECKING:
    from .vector_searcher import VectorSearcher


class JournalStore:
    """期刊存储与检索（分离存储与搜索）"""

    def __init__(
        self,
        store_path: str = "data/processed/journals.jsonl",
    ):
        self.store_path = store_path
        self._journals: List[Journal] = []
        self._vector_searcher: Optional["VectorSearcher"] = None

    def set_vector_searcher(self, searcher: "VectorSearcher") -> None:
        """注入向量搜索器"""
        self._vector_searcher = searcher

    def has_vector_search(self) -> bool:
        """检查向量搜索是否可用"""
        return self._vector_searcher is not None and self._vector_searcher.is_available

    def load(self) -> None:
        """加载期刊数据"""
        if not os.path.exists(self.store_path):
            return

        self._journals = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                journal = Journal(**data)
                journal.build_profile()
                self._journals.append(journal)

    def save(self) -> None:
        """保存期刊数据"""
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.store_path, "w", encoding="utf-8") as f:
            for journal in self._journals:
                f.write(json.dumps(journal.model_dump(), ensure_ascii=False) + "\n")

    def add_journal(self, journal: Journal) -> None:
        """添加期刊"""
        self._journals.append(journal)

    def add_journals(self, journals: List[Journal]) -> None:
        """批量添加期刊"""
        for journal in journals:
            self.add_journal(journal)

    def search_by_vector(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[Journal, float]]:
        """向量检索（委托给 VectorSearcher）"""
        if self._vector_searcher is None:
            raise ValueError("Vector search not configured")
        return self._vector_searcher.search(query_embedding, self._journals, top_k)

    def search_by_text(
        self, query_text: str, top_k: int = 10
    ) -> List[Tuple[Journal, float]]:
        """文本搜索（基于 profile 的简单匹配）"""
        query_keywords = set(query_text.lower().split())
        scores = []
        for journal in self._journals:
            profile_keywords = set(journal.journal_profile.lower().split())
            intersection = query_keywords & profile_keywords
            if intersection:
                score = len(intersection) / max(len(query_keywords), 1)
                scores.append((journal, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_journal(self, journal_id: str) -> Optional[Journal]:
        """根据 ID 获取期刊"""
        for journal in self._journals:
            if journal.journal_id == journal_id:
                return journal
        return None

    def list_journals(
        self,
        subject_tag: Optional[str] = None,
        oa_type: Optional[str] = None,
        quartile: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Journal]:
        """列出期刊（支持过滤）"""
        results = self._journals
        if subject_tag:
            results = [j for j in results if subject_tag in j.subject_tags]
        if oa_type:
            results = [j for j in results if j.oa_type == oa_type]
        if quartile:
            results = [j for j in results if j.quartile == quartile]
        return results[offset:offset + limit]

    @property
    def count(self) -> int:
        """期刊数量"""
        return len(self._journals)

    @property
    def journals(self) -> List[Journal]:
        """获取所有期刊（只读视图）"""
        return self._journals

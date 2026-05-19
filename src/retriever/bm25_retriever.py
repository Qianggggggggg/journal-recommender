"""BM25 召回"""
from typing import List, Tuple

import rank_bm25

from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore


class BM25Retriever:
    """BM25 召回器"""

    def __init__(self, store: JournalStore):
        self.store = store
        self._tokenized_profiles: List[List[str]] = []
        self._bm25_index: rank_bm25.BM25Okapi = None
        self._built = False

    def build_index(self) -> None:
        """构建 BM25 索引"""
        if self.store.count == 0:
            return

        self._tokenized_profiles = []
        for journal in self.store._journals:
            profile = journal.journal_profile or journal.scope_text
            tokens = profile.lower().split()
            self._tokenized_profiles.append(tokens)

        self._bm25_index = rank_bm25.BM25Okapi(self._tokenized_profiles)
        self._built = True

    def retrieve(self, query: str, top_k: int = 30) -> List[Tuple[Journal, float]]:
        """BM25 检索"""
        if not self._built:
            self.build_index()

        if self._bm25_index is None:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25_index.get_scores(query_tokens)

        # 获取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.store._journals):
                results.append((self.store._journals[idx], float(scores[idx])))
        return results
"""向量检索召回"""
from typing import List, Tuple

import numpy as np

from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore
from ..utils.embedding import OllamaEmbedding


class EmbeddingRetriever:
    """向量检索召回器"""

    def __init__(self, store: JournalStore, embedding_client: OllamaEmbedding):
        self.store = store
        self.embedding_client = embedding_client

    def retrieve(
        self, query_text: str, top_k: int = 30
    ) -> List[Tuple[Journal, float]]:
        """向量检索"""
        # 获取查询向量
        query_embedding = self.embedding_client.embed(query_text)

        # FAISS 检索
        results = self.store.search_by_vector(query_embedding, top_k)

        # 转换为 (journal, score) 格式，score 取负距离（距离越小越相似）
        return [(journal, -score) for journal, score in results]

    def retrieve_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 30
    ) -> List[Tuple[Journal, float]]:
        """直接用向量检索"""
        results = self.store.search_by_vector(query_embedding, top_k)
        return [(journal, -score) for journal, score in results]
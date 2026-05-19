"""向量搜索接口"""
from typing import List, Tuple, Optional, Protocol
import numpy as np

import faiss
import pandas as pd

from .journal_model import Journal


class VectorIndex(Protocol):
    """向量索引协议（可注入不同实现）"""

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索返回距离和索引"""
        ...


class FaissIndex:
    """FAISS 向量索引适配器"""

    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """加载索引"""
        import os
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._index = faiss.read_index(self.index_path)
            try:
                self._metadata = pd.read_parquet(self.metadata_path)
            except Exception:
                # pyarrow 版本过低或 parquet 文件损坏，优雅降级
                self._metadata = None

    def save(self) -> None:
        """保存索引"""
        if self._index is not None:
            import os
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self._index, self.index_path)
        if self._metadata is not None:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            self._metadata.to_parquet(self.metadata_path)

    def build(self, embeddings: np.ndarray) -> None:
        """构建索引"""
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings.astype(np.float32))
        self._metadata = None  # 由调用者设置

    def set_metadata(self, metadata: pd.DataFrame) -> None:
        self._metadata = metadata

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索"""
        if self._index is None:
            raise ValueError("FAISS index not built")
        query = query.reshape(1, -1).astype(np.float32)
        return self._index.search(query, k)

    @property
    def is_loaded(self) -> bool:
        return self._index is not None


class VectorSearcher:
    """向量搜索器"""

    def __init__(self, index: FaissIndex):
        self._index = index

    @classmethod
    def from_files(
        cls, faiss_path: str = "data/processed/journals_index.faiss",
        metadata_path: str = "data/processed/journals_metadata.parquet"
    ) -> "VectorSearcher":
        """从文件创建"""
        idx = FaissIndex(faiss_path, metadata_path)
        idx.load()
        return cls(idx)

    def search(
        self, query_embedding: np.ndarray, journals: List[Journal], top_k: int = 10
    ) -> List[Tuple[Journal, float]]:
        """向量检索"""
        distances, indices = self._index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(journals):
                results.append((journals[int(idx)], float(dist)))
        return results

    @property
    def is_available(self) -> bool:
        return self._index.is_loaded
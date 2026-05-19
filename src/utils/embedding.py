"""Ollama Embedding 调用封装"""
import os
from typing import List

import httpx
import numpy as np


class OllamaEmbedding:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
    ):
        self.base_url = base_url
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        """获取单条文本的 embedding 向量"""
        url = f"{self.base_url}/api/embeddings"
        response = httpx.post(url, json={"model": self.model, "prompt": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取文本 embedding"""
        url = f"{self.base_url}/api/embeddings"
        results = []
        for text in texts:
            response = httpx.post(url, json={"model": self.model, "prompt": text}, timeout=30)
            response.raise_for_status()
            data = response.json()
            results.append(np.array(data["embedding"]))
        return results
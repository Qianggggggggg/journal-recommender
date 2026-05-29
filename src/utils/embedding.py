"""Ollama Embedding 调用封装"""
from typing import List

import numpy as np
import requests


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
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"])

    def embed_batch(self, texts: List[str], concurrency: int = 1, timeout: float = 60.0) -> List[np.ndarray]:
        """批量获取文本 embedding（串行请求，逐条嵌入）"""
        results = []
        for text in texts:
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    results.append(np.array(response.json()["embedding"]))
                    break
                except requests.HTTPError as e:
                    if e.response.status_code == 502 and attempt < 2:
                        import time
                        time.sleep(2 ** attempt)
                        continue
                    raise
        return results
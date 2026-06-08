"""Ollama Embedding 调用封装"""
from typing import List
from typing import Optional

import numpy as np
import requests


class OllamaEmbedding:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        """获取单条文本的 embedding 向量. Retries on 5xx and connection errors
        (Ollama occasionally 500s when the desktop app self-updates in the
        background, or when the model is reloaded after a long idle period)."""
        import time as _time
        for attempt in range(5):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return np.array(response.json()["embedding"])
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = (
                    status in (500, 502, 503, 504)
                    or isinstance(e, (requests.ConnectionError, requests.Timeout))
                )
                if retryable and attempt < 4:
                    wait = min(2 ** attempt, 30)
                    print(f"  [embed] {type(e).__name__} {status or ''} retry in {wait}s",
                          flush=True)
                    _time.sleep(wait)
                    continue
                raise

    def embed_batch(self, texts: List[str], concurrency: int = 1, timeout: Optional[float] = None) -> List[np.ndarray]:
        """批量获取文本 embedding（串行请求，逐条嵌入）"""
        request_timeout = self.timeout if timeout is None else timeout
        results = []
        for text in texts:
            for attempt in range(5):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=request_timeout,
                    )
                    response.raise_for_status()
                    results.append(np.array(response.json()["embedding"]))
                    break
                except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    retryable = (
                        status in (500, 502, 503, 504)
                        or isinstance(e, (requests.ConnectionError, requests.Timeout))
                    )
                    if retryable and attempt < 4:
                        import time as _time
                        wait = min(2 ** attempt, 30)
                        print(f"  [embed_batch] {type(e).__name__} {status or ''} retry in {wait}s",
                              flush=True)
                        _time.sleep(wait)
                        continue
                    raise
                    raise
        return results

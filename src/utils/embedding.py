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

    def embed_batch(self, texts: List[str], concurrency: int = 3, timeout: float = 60.0) -> List[np.ndarray]:
        """批量获取文本 embedding（带并发限制 + 重试 + 指数退避）"""
        import asyncio
        import httpx

        semaphore = asyncio.Semaphore(concurrency)

        async def fetchEmbedding(client: httpx.AsyncClient, text: str) -> np.ndarray:
            for attempt in range(3):
                try:
                    async with semaphore:
                        response = await client.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": self.model, "prompt": text},
                            timeout=httpx.Timeout(timeout),
                        )
                        response.raise_for_status()
                        return np.array(response.json()["embedding"])
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 502 and attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise

        async def main():
            async with httpx.AsyncClient() as client:
                tasks = [fetchEmbedding(client, text) for text in texts]
                return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, main())
                return future.result()
        else:
            return asyncio.run(main())
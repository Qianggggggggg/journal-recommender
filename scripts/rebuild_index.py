"""重建 BM25 和 FAISS 索引"""
import json
import sys
import os
import time

sys.path.insert(0, '/Users/qian/PycharmProjects/paper')

import numpy as np
import httpx
import faiss
import pandas as pd

from src.journals.journal_store import JournalStore
from src.journals.journal_model import Journal
from src.retriever.bm25_retriever import BM25Retriever

def embed_texts(texts: list, model: str = "qwen3-embedding:4b") -> np.ndarray:
    """使用 Ollama 生成 embedding"""
    embeddings = []
    url = "http://localhost:11434/api/embeddings"

    for i, text in enumerate(texts):
        try:
            resp = httpx.post(url, json={"model": model, "prompt": text}, timeout=60)
            resp.raise_for_status()
            embedding = np.array(resp.json()["embedding"])
            embeddings.append(embedding)
            if (i + 1) % 50 == 0:
                print(f"  Embedded {i+1}/{len(texts)}...")
        except Exception as e:
            print(f"  Error at {i}: {e}")
            embeddings.append(np.zeros(3584))  # qwen3-embedding 输出维度

    return np.array(embeddings)


def rebuild_bm25():
    """重建 BM25 索引"""
    print("=== 重建 BM25 索引 ===")
    store = JournalStore("data/processed/journals.jsonl")
    store.load()
    print(f"加载期刊: {store.count}")

    bm25 = BM25Retriever(store)
    bm25.build_index()
    print("BM25 索引构建完成\n")


def rebuild_faiss():
    """重建 FAISS 索引"""
    print("=== 重建 FAISS 索引 ===")

    # 加载期刊
    journals = []
    with open("data/processed/journals.jsonl", encoding="utf-8") as f:
        for line in f:
            journals.append(Journal(**json.loads(line)))

    print(f"加载期刊: {len(journals)}")

    # 生成 profile 文本
    profiles = []
    for j in journals:
        text = j.journal_name + ". " + (j.scope_text or "")
        profiles.append(text)

    print("生成 embedding (qwen3-embedding:4b)...")
    start = time.time()
    embeddings = embed_texts(profiles)
    print(f"Embedding shape: {embeddings.shape}, 耗时: {time.time()-start:.1f}s")

    # 构建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # 保存
    faiss_path = "data/processed/journals_index.faiss"
    metadata_path = "data/processed/journals_metadata.parquet"

    faiss.write_index(index, faiss_path)
    print(f"FAISS 索引保存: {faiss_path}")

    # 保存 metadata
    df = pd.DataFrame([{
        'journal_id': j.journal_id,
        'journal_name': j.journal_name,
        'subject_tags': ','.join(j.subject_tags),
        'ccf_rating': j.ccf_rating or '',
    } for j in journals])

    df.to_parquet(metadata_path)
    print(f"Metadata 保存: {metadata_path}")


if __name__ == "__main__":
    rebuild_bm25()
    print()
    rebuild_faiss()
    print("\n索引重建完成!")
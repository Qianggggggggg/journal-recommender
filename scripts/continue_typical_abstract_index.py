#!/usr/bin/env python3
"""增量构建FAISS索引：追加剩余的典型摘要"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import time

sys.path.insert(0, str(Path('.').absolute()))

from src.journals.typical_abstract_store import TypicalAbstractStore
from src.utils.embedding import OllamaEmbedding

BATCH_SIZE = 30  # 更小的批次
START_OFFSET = 300  # 从300开始（已有300条）

def main():
    store = TypicalAbstractStore()
    store.load()
    total = len(store.records)
    print(f"Total typical abstracts: {total}")

    # 检查已有索引
    existing_index = 'data/processed/typical_abstracts_index.faiss'
    existing_meta = 'data/processed/typical_abstracts_metadata.parquet'

    if Path(existing_index).exists():
        import faiss
        existing_idx = faiss.read_index(existing_index)
        print(f"Existing index has {existing_idx.ntotal} vectors")

    with open('configs/app.yaml', 'r') as f:
        config = yaml.safe_load(f)

    embedding_client = OllamaEmbedding(
        base_url=config['ollama']['base_url'],
        model=config['ollama']['embedding_model'],
    )

    remaining = total - START_OFFSET
    print(f"Embedding remaining {remaining} texts in batches of {BATCH_SIZE}...")

    success_count = 0
    for i in range(START_OFFSET, total, BATCH_SIZE):
        batch = store.records[i:i+BATCH_SIZE]
        texts = [r.search_text for r in batch]
        print(f"  Batch {i//BATCH_SIZE + 1}: embedding {len(texts)} texts...", end='', flush=True)

        for attempt in range(3):
            try:
                embeddings = embedding_client.embed_batch(texts, concurrency=2)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt * 10
                    print(f" error, retry in {wait}s...", end='', flush=True)
                    time.sleep(wait)
                    continue
                else:
                    raise

        success_count += len(texts)
        print(f" done ({success_count}/{remaining})")

    print(f"Successfully embedded {success_count} texts")
    print("Note: Run build_typical_abstract_index_batch.py to rebuild the full index with all data")

if __name__ == "__main__":
    main()
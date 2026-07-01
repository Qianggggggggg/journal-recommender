#!/usr/bin/env python3
"""分批构建典型摘要FAISS索引，避免Ollama超时。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path('.').absolute()))

from src.journals.typical_abstract_store import TypicalAbstractStore
from src.utils.embedding import OllamaEmbedding

BATCH_SIZE = 50
TOTAL_LIMIT = 1180  # 全部 1180 篇典型摘要

def main():
    store = TypicalAbstractStore()
    store.load()
    print(f"Loaded {store.count} typical abstracts from {store.journal_count} journals")

    with open('configs/app.yaml', 'r') as f:
        config = yaml.safe_load(f)

    embedding_client = OllamaEmbedding(
        base_url=config['ollama']['base_url'],
        model=config['ollama']['embedding_model'],
    )

    all_embeddings = []
    total = min(len(store.records), TOTAL_LIMIT)

    print(f"Embedding {total} texts in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = store.records[i:i+BATCH_SIZE]
        texts = [r.search_text for r in batch]
        print(f"  Batch {i//BATCH_SIZE + 1}: embedding {len(texts)} texts...", end='', flush=True)
        embeddings = embedding_client.embed_batch(texts, concurrency=3)
        all_embeddings.extend(embeddings)
        print(f" done ({len(all_embeddings)}/{total})")

    matrix = np.array(all_embeddings, dtype=np.float32)
    print(f"Matrix shape: {matrix.shape}")

    import faiss
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, 'data/processed/typical_abstracts_index.faiss')

    metadata = pd.DataFrame([
        {
            'anchor_id': store.records[i].anchor_id,
            'journal_id': store.records[i].journal_id,
            'journal_name': store.records[i].journal_name,
            'method_type': store.records[i].method_type,
            'novelty_level': store.records[i].novelty_level,
            'ccf_rating': store.records[i].ccf_rating,
        }
        for i in range(total)
    ])
    metadata.to_parquet('data/processed/typical_abstracts_metadata.parquet')

    print(f"FAISS index: data/processed/typical_abstracts_index.faiss")
    print(f"Metadata: data/processed/typical_abstracts_metadata.parquet")
    print("Done!")

if __name__ == "__main__":
    main()
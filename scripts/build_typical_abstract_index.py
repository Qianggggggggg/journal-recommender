#!/usr/bin/env python3
"""构建典型摘要 FAISS 向量索引。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.typical_abstract_store import TypicalAbstractStore
from src.utils.embedding import OllamaEmbedding


def build_typical_abstract_index(
    abstracts_dir: str = "data/typical_abstracts",
    faiss_path: str = "data/processed/typical_abstracts_index.faiss",
    metadata_path: str = "data/processed/typical_abstracts_metadata.parquet",
) -> None:
    """把 1,180 篇典型摘要编码为 FAISS 文档索引。"""
    store = TypicalAbstractStore(abstracts_dir)
    store.load()
    if store.count == 0:
        raise RuntimeError(f"No typical abstracts found in {abstracts_dir}")

    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    embedding_client = OllamaEmbedding(
        base_url=config["ollama"]["base_url"],
        model=config["ollama"]["embedding_model"],
        timeout=config.get("ollama", {}).get("timeout_seconds", 60),
        show_progress=True,
        progress_desc="Typical-abstract embeddings",
    )

    texts = [record.search_text for record in store.records]
    print(f"Embedding {len(texts)} typical abstracts from {store.journal_count} journals...")
    embeddings = embedding_client.embed_batch(texts)
    matrix = np.array(embeddings, dtype=np.float32)

    import faiss

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, faiss_path)

    metadata = pd.DataFrame([
        {
            "anchor_id": record.anchor_id,
            "journal_id": record.journal_id,
            "journal_name": record.journal_name,
            "method_type": record.method_type,
            "novelty_level": record.novelty_level,
            "ccf_rating": record.ccf_rating,
        }
        for record in store.records
    ])
    metadata.to_parquet(metadata_path)

    print(f"FAISS index saved to: {faiss_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    build_typical_abstract_index()

"""构建向量索引脚本"""
import sys
from pathlib import Path
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.journal_store import JournalStore
from src.journals.vector_searcher import FaissIndex, VectorSearcher
from src.utils.embedding import OllamaEmbedding
import yaml


def build_index(
    store_path: str = "data/processed/journals.jsonl",
    faiss_path: str = "data/processed/journals_index.faiss",
    meta_path: str = "data/processed/journals_metadata.parquet",
):
    """构建 FAISS 向量索引"""
    print("Loading journals...")
    store = JournalStore(store_path=store_path)
    store.load()

    if store.count == 0:
        print("No journals to index")
        return

    print(f"Indexing {store.count} journals...")

    # 加载配置
    config_path = project_root / "configs" / "app.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    # 初始化 embedding client
    embedding_client = OllamaEmbedding(
        base_url=app_config["ollama"]["base_url"],
        model=app_config["ollama"]["embedding_model"],
        timeout=app_config.get("ollama", {}).get("timeout_seconds", 60),
        show_progress=True,
        progress_desc="Journal-profile embeddings",
    )

    # 构建 journal_profile 列表
    profiles = [j.journal_profile for j in store._journals]

    # 批量获取 embedding
    print("Computing embeddings (this may take a while)...")
    embeddings = embedding_client.embed_batch(profiles)

    import numpy as np
    embeddings_matrix = np.array(embeddings)

    # 构建 FAISS 索引
    print("Building FAISS index...")
    faiss_idx = FaissIndex(faiss_path, meta_path)
    faiss_idx.build(embeddings_matrix)

    # 构建 metadata (DataFrame with journal info)
    import pandas as pd
    metadata = pd.DataFrame([{
        "journal_id": j.journal_id,
        "journal_name": j.journal_name,
    } for j in store._journals])
    faiss_idx.set_metadata(metadata)
    faiss_idx.save()

    # 设置向量搜索器
    vector_searcher = VectorSearcher(faiss_idx)
    store.set_vector_searcher(vector_searcher)

    print(f"Index built successfully: {store.count} journals")
    print(f"FAISS index saved to: {faiss_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    build_index()

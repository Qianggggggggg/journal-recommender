"""将 CCF 推荐期刊添加到数据库"""
import json
import sys
sys.path.insert(0, "/Users/qian/PycharmProjects/paper")

from scripts.ccf_journals_data import build_journal_list
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.utils.embedding import OllamaEmbedding
import numpy as np


def add_ccf_journals():
    """添加 CCF 期刊到数据库"""
    # 构建 CCF 期刊列表
    ccf_journals = build_journal_list()
    print(f"CCF journals: {len(ccf_journals)}")

    # 加载现有期刊
    store = JournalStore()
    store.load()
    print(f"Existing journals: {store.count}")

    # 获取现有期刊 ID 集合
    existing_ids = {j.journal_id for j in store._journals}

    # 添加新的 CCF 期刊（去重）
    added_count = 0
    for j_data in ccf_journals:
        if j_data["journal_id"] in existing_ids:
            print(f"Skip existing: {j_data['journal_id']}")
            continue

        journal = Journal(
            journal_id=j_data["journal_id"],
            journal_name=j_data["journal_name"],
            publisher=j_data.get("publisher", ""),
            subject_tags=j_data.get("subject_tags", []),
            keywords=[],
            scope_text=j_data.get("scope_text", ""),
            oa_type="subscription",  # CCF期刊默认订阅型
            submission_url=j_data.get("url", ""),
            homepage_url=j_data.get("url", ""),
            quartile=j_data.get("quartile", "Q2"),
            impact_like_score=None,
            review_time=None,
            apc=None,
            target_paper_type=[],
            ccf_rank=j_data.get("ccf_rank", "C"),
        )
        journal.build_profile()
        store.add_journal(journal)
        added_count += 1
        print(f"Added: {j_data['journal_id']} - {j_data['journal_name'][:40]}")

    print(f"\nAdded {added_count} new journals")
    print(f"Total journals: {store.count}")

    # 保存
    store.save()
    print("Saved to journals.jsonl")

    return store, added_count


def rebuild_faiss_index(store: JournalStore):
    """重建 FAISS 索引"""
    print("\nRebuilding FAISS index...")

    embedding_client = OllamaEmbedding(
        base_url="http://localhost:11434",
        model="qwen3-embedding:4b",
    )

    profiles = [j.journal_profile for j in store._journals]
    print(f"Embedding {len(profiles)} journals...")

    embeddings = []
    batch_size = 16
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i + batch_size]
        try:
            # 使用单条循环调用（兼容 embed 接口）
            for text in batch:
                emb = embedding_client.embed(text)
                embeddings.append(emb)
        except Exception as e:
            print(f"Batch {i} error: {e}")
            # 使用零向量填充失败的批次
            dim = 2560  # qwen3-embedding:4b 的维度
            for _ in batch:
                embeddings.append(np.zeros(dim))

    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    store.build_faiss_index(embeddings)
    store.save()
    print("FAISS index rebuilt and saved")


if __name__ == "__main__":
    store, added = add_ccf_journals()

    if added > 0:
        rebuild_faiss_index(store)
    else:
        print("No new journals added, skipping FAISS rebuild")
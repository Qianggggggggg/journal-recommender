#!/usr/bin/env python3
"""
诊断粗排遗漏脚本

针对 coarse_hit 为 false 的论文，分析目标期刊未被召回的原因。
"""
import json
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Plus
from src.journals.journal_store import JournalStore
from src.journals.vector_searcher import VectorSearcher, FaissIndex
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.embedding_retriever import EmbeddingRetriever
from src.papers.paper_model import PaperProfile
import yaml


def load_journals() -> List[Dict[str, Any]]:
    """加载期刊库"""
    import yaml
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)
    journal_path = app_config["data"]["journal_store_path"]
    print(f"期刊库路径: {journal_path}")
    journals = []
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            journals.append(json.loads(line))
    return journals


def load_papers_metadata() -> Dict[str, Dict]:
    """加载论文元数据（包含完整信息）"""
    papers = {}
    path = "data/evaluation/papers_metadata.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            arxiv_id = p.get("external_ids", {}).get("arXiv", "")
            if arxiv_id:
                papers[arxiv_id] = p
    return papers


def find_journal_by_name(journals: List[Dict], target_name: str) -> Optional[Dict]:
    """通过名称精确或模糊匹配期刊"""
    target_lower = target_name.lower()

    # 精确匹配
    for j in journals:
        if j.get("journal_name", "").lower() == target_lower:
            return j

    # 模糊匹配（包含关系）
    candidates = []
    for j in journals:
        jn = j.get("journal_name", "").lower()
        # 去除常见前缀后匹配
        if target_lower in jn or jn in target_lower:
            candidates.append(j)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # 返回最相似的
        for c in candidates:
            if c.get("journal_name", "").lower() == target_lower:
                return c
        return candidates[0]

    return None


def build_bm25_index(journals: List[Dict]) -> tuple:
    """构建 BM25 索引"""
    scope_texts = [j.get("scope_text", "") or "" for j in journals]
    tokenized_corpus = [text.split() for text in scope_texts]
    bm25 = BM25Plus(tokenized_corpus)

    # 归一化基准
    dummy_query = "machine learning deep neural network".split()
    dummy_scores = bm25.get_scores(dummy_query)
    max_score = max(dummy_scores) if max(dummy_scores) > 0 else 1.0

    return bm25, scope_texts, max_score


def bm25_retrieve(
    bm25: BM25Plus,
    journals: List[Dict],
    query_text: str,
    max_score: float,
    top_k: int = 50
) -> List[tuple]:
    """BM25 检索"""
    tokenized_query = query_text.split()
    scores = bm25.get_scores(tokenized_query)

    # 归一化
    normalized = [s / max_score if max_score > 0 else 0.0 for s in scores]

    # 排序
    scored = list(zip(range(len(journals)), normalized))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def main():
    # 加载评估结果
    eval_path = "data/evaluation/results/eval_abstract_top5_20260526_204824.json"
    if not os.path.exists(eval_path):
        # 尝试找最新的
        results_dir = Path("data/evaluation/results")
        if results_dir.exists():
            files = list(results_dir.glob("eval_*.json"))
            if files:
                eval_path = str(sorted(files)[-1])

    print(f"加载评估结果: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 筛选 coarse_hit 为 false 的论文
    miss_papers = [p for p in eval_data["paper_results"] if not p.get("coarse_hit")]
    print(f"\n粗排遗漏论文数: {len(miss_papers)}")

    # 加载期刊库
    print("加载期刊库...")
    journals = load_journals()
    print(f"期刊库总数: {len(journals)}")

    # 加载论文完整信息
    print("加载论文完整信息...")
    papers_metadata = load_papers_metadata()
    print(f"论文元数据总数: {len(papers_metadata)}")

    # 构建 BM25 索引
    print("构建 BM25 索引...")
    bm25, scope_texts, max_bm25_score = build_bm25_index(journals)

    # 诊断每篇论文
    print("\n" + "=" * 80)
    print("详细诊断报告")
    print("=" * 80)

    miss_reasons = {
        "journal_not_in_corpus": [],
        "bm25_rank_too_low": [],
        "query_mismatch": [],
    }

    for i, paper in enumerate(miss_papers, 1):
        title = paper.get("title", "")
        venue = paper.get("venue", "")
        ccf_level = paper.get("ccf_level", "")
        research_area = paper.get("research_area", "")
        arxiv_id = paper.get("arxiv", "")

        # 获取论文完整信息
        full_info = papers_metadata.get(arxiv_id, {})
        abstract = full_info.get("abstract", "")
        keywords = full_info.get("keywords", [])
        techniques = full_info.get("techniques", [])
        datasets = full_info.get("datasets", [])
        evaluation_metrics = full_info.get("evaluation_metrics", [])
        application_domain = full_info.get("application_domain", [])
        novelty_type = full_info.get("novelty_type", "")

        print(f"\n{'='*60}")
        print(f"[{i}] 论文: {title[:50]}...")
        print(f"    目标期刊: {venue}")
        print(f"    CCF等级: {ccf_level} | 研究领域: {research_area}")

        # 1. 检查期刊是否在库中
        matched_journal = find_journal_by_name(journals, venue)

        if matched_journal is None:
            print(f"    ❌ 期刊缺失 - 未在期刊库中找到")
            miss_reasons["journal_not_in_corpus"].append({
                "title": title,
                "venue": venue,
                "ccf_level": ccf_level
            })
            continue

        print(f"    ✅ 期刊存在于库中")
        print(f"    期刊ID: {matched_journal.get('journal_id', 'N/A')}")

        # 2. 输出期刊信息
        print(f"\n    --- 期刊信息 ---")
        print(f"    scope_text (前200字): {matched_journal.get('scope_text', '')[:200]}...")
        print(f"    keywords: {matched_journal.get('keywords', [])}")
        print(f"    subject_tags: {matched_journal.get('subject_tags', [])}")

        # 3. 构建 rich_query（使用完整论文信息）
        rich_query = " ".join([
            title,
            abstract or "",
            " ".join(techniques),
            " ".join(keywords),
            " ".join(datasets),
            " ".join(evaluation_metrics),
            " ".join(application_domain),
            novelty_type or "",
        ])
        print(f"\n    rich_query (前200字): {rich_query[:200]}...")

        # 4. BM25 检索
        bm25_results = bm25_retrieve(bm25, journals, rich_query, max_bm25_score, top_k=50)

        # 找到目标期刊的位置
        target_idx = None
        for idx, j in enumerate(journals):
            if j.get("journal_name", "").lower() == venue.lower():
                target_idx = idx
                break

        target_rank = None
        target_score = None
        if target_idx is not None:
            for rank, (idx, score) in enumerate(bm25_results, 1):
                if idx == target_idx:
                    target_rank = rank
                    target_score = score
                    break

        if target_rank:
            print(f"\n    BM25 检索结果:")
            print(f"    目标期刊排名: 第 {target_rank} 位 (分数: {target_score:.4f})")
            if target_rank > 20:
                print(f"    ⚠️ 排名太低，未能进入前50候选")
                miss_reasons["bm25_rank_too_low"].append({
                    "title": title,
                    "venue": venue,
                    "bm25_rank": target_rank,
                    "score": target_score
                })
        else:
            print(f"\n    BM25 检索结果:")
            print(f"    ⚠️ 目标期刊未在 BM25 Top50 中 (可能分数为0)")

        # 5. 显示 BM25 Top10
        print(f"\n    BM25 Top10:")
        for rank, (idx, score) in enumerate(bm25_results[:10], 1):
            j = journals[idx]
            marker = " <-- 目标" if j.get("journal_name", "").lower() == venue.lower() else ""
            print(f"    {rank:2d}. {j.get('journal_name', '')[:40]:<40} (分数: {score:.4f}){marker}")

    # 6. 汇总
    print("\n" + "=" * 80)
    print("汇总诊断结果")
    print("=" * 80)

    total_miss = len(miss_papers)
    print(f"\n总遗漏论文数: {total_miss}")

    journal_not_found = len(miss_reasons["journal_not_in_corpus"])
    bm25_rank_low = len(miss_reasons["bm25_rank_too_low"])

    print(f"\n遗漏原因分布:")
    print(f"  1. 期刊不在库中: {journal_not_found}/{total_miss} ({journal_not_found*100/total_miss:.1f}%)")
    print(f"  2. BM25排名过低: {bm25_rank_low}/{total_miss} ({bm25_rank_low*100/total_miss:.1f}%)")

    if miss_reasons["journal_not_in_corpus"]:
        print(f"\n缺失期刊列表:")
        for p in miss_reasons["journal_not_in_corpus"]:
            print(f"  - {p['venue']} (CCF-{p['ccf_level']}, 论文: {p['title'][:40]}...)")

    if miss_reasons["bm25_rank_too_low"]:
        print(f"\nBM25 排名过低期刊:")
        for p in sorted(miss_reasons["bm25_rank_too_low"], key=lambda x: x["bm25_rank"], reverse=True):
            print(f"  - {p['venue']}: 第 {p['bm25_rank']} 位 (分数: {p['score']:.4f})")


if __name__ == "__main__":
    main()
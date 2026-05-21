#!/usr/bin/env python3
"""为爬取的论文匹配 CCF 期刊等级"""

import json
import sys
from pathlib import Path
from typing import Optional

try:
    from Levenshtein import ratio
except ImportError:
    print("请安装 python-Levenshtein: pip install python-Levenshtein")
    sys.exit(1)


def load_ccf_journals(ccf_path: str = "data/journals_ccf.jsonl") -> dict[str, dict]:
    """加载 CCF 期刊列表"""
    journals = {}
    with open(ccf_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            journals[j["journal_name"]] = j
    return journals


def load_raw_papers(raw_path: str) -> list[dict]:
    """加载原始爬取的论文"""
    papers = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def find_matching_journal(paper: dict, ccf_journals: dict, threshold: float = 0.6) -> Optional[dict]:
    """
    模糊匹配论文 venue 与 CCF 期刊

    Args:
        paper: 论文数据
        ccf_journals: CCF 期刊字典
        threshold: 匹配度阈值

    Returns:
        匹配的期刊数据
    """
    venue = paper.get("venue", "").lower()
    if not venue:
        return None

    best_match = None
    best_score = 0

    for name, journal in ccf_journals.items():
        # 计算相似度
        name_lower = name.lower()
        score = ratio(venue, name_lower)

        # 如果 venue 包含期刊名关键词
        if any(kw in venue for kw in name_lower.split()):
            score = max(score, 0.7)

        if score > best_score and score >= threshold:
            best_score = score
            best_match = journal

    return best_match


def match_and_save(raw_path: str, output_path: str, ccf_path: str = "data/journals_ccf.jsonl"):
    """匹配 CCF 期刊并保存"""
    ccf_journals = load_ccf_journals(ccf_path)
    papers = load_raw_papers(raw_path)

    matched_count = 0
    results = []

    for paper in papers:
        journal = find_matching_journal(paper, ccf_journals)

        if journal:
            matched_paper = {
                "title": paper["title"],
                "abstract": paper.get("abstract", ""),
                "research_area": paper.get("keyword", ""),
                "published_journal": journal["journal_name"],
                "ccf_rating": journal.get("ccf_rating", "N/A"),
                "quartile": journal.get("quartile", ""),
            }
            results.append(matched_paper)
            matched_count += 1
            print(f"匹配成功 ({matched_count}): {paper['title'][:40]}... -> {journal['journal_name']}")
        else:
            print(f"未匹配: {paper['title'][:40]}...")

    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for paper in results:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"\n完成！共匹配 {matched_count}/{len(papers)} 篇，保存到 {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="匹配 CCF 期刊")
    parser.add_argument("--input", "-i", default="data/evaluation/ground_truth_raw.jsonl",
                        help="原始论文数据路径")
    parser.add_argument("--output", "-o", default="data/evaluation/ground_truth.jsonl",
                        help="输出文件路径")

    args = parser.parse_args()

    match_and_save(args.input, args.output)
#!/usr/bin/env python3
"""DBLP 论文数据采集脚本 - 用于评估数据集构建"""

import json
import sys
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


def search_dblp_by_keyword(keyword: str, year: int = 2020, max_results: int = 50) -> list[dict]:
    """
    通过关键词搜索 DBLP 论文

    Args:
        keyword: 搜索关键词
        year: 发表年份下限
        max_results: 最大结果数

    Returns:
        论文列表
    """
    # DBLP API (faceted search)
    url = f"https://dblp.org/search/pubs?q={keyword}&f=year:{year}&h={max_results}&format=json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        papers = []
        for hit in data.get("result", {}).get("hits", {}).get("hit", []):
            info = hit.get("info", {})
            title = info.get("title", "")
            abstract = info.get("abstract", "")
            venue = info.get("venue", "")
            year = info.get("year", "")
            ee = info.get("ee", "")

            if not title:
                continue

            papers.append({
                "title": title,
                "abstract": abstract,
                "venue": venue,
                "year": year,
                "url": ee,
                "keyword": keyword,
            })

        return papers
    except Exception as e:
        print(f"Error searching DBLP for keyword '{keyword}': {e}", file=sys.stderr)
        return []


def crawl_evaluation_dataset(output_path: str, target_count: int = 200):
    """
    爬取评估数据集

    Args:
        output_path: 输出文件路径
        target_count: 目标论文数量
    """
    # 搜索关键词，覆盖不同领域
    keywords = [
        "deep learning",
        "neural network",
        "computer vision",
        "natural language processing",
        "machine learning",
        "reinforcement learning",
        "graph neural network",
        "transformer",
        "object detection",
        "semantic segmentation",
        "text classification",
        "question answering",
        "knowledge graph",
        "recommender system",
        "federated learning",
    ]

    all_papers = []
    seen_titles = set()

    print(f"开始爬取 DBLP 论文，目标 {target_count} 篇...")

    for keyword in keywords:
        if len(all_papers) >= target_count:
            break

        print(f"搜索关键词: {keyword}")

        papers = search_dblp_by_keyword(keyword, year=2019, max_results=50)

        for paper in papers:
            title = paper["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            all_papers.append(paper)
            print(f"  已获取 {len(all_papers)}/{target_count} 篇: {title[:50]}...")

            if len(all_papers) >= target_count:
                break

    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"\n完成！共获取 {len(all_papers)} 篇论文，保存到 {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DBLP 论文数据采集")
    parser.add_argument("--output", "-o", default="data/evaluation/ground_truth_raw.jsonl",
                        help="输出文件路径")
    parser.add_argument("--count", "-c", type=int, default=200,
                        help="目标论文数量")

    args = parser.parse_args()

    crawl_evaluation_dataset(args.output, args.count)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Scholar 论文爬虫 - 支持多 CCF 领域、多等级
输出每篇论文的完整信息（JSONL），确保 venue、等级、领域与 journals 数据一致。
"""

import json
import os
import time
import argparse
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import requests
import pandas as pd
from tqdm import tqdm

# ====================== 配置 ======================
API_KEY = os.environ.get("S2_API_KEY", "")
BASE_URL = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

REQUEST_DELAY = 0.3
MAX_RETRIES = 3

# ====================== 工具函数 ======================
def load_journals(journals_file: str) -> Dict[str, Tuple[str, List[str]]]:
    """加载期刊文件，返回 {journal_name: (ccf_rating, [subject_tags])}"""
    journal_map = {}
    if journals_file.endswith('.csv'):
        df = pd.read_csv(journals_file)
        for _, row in df.iterrows():
            name = row['journal_name']
            rating = row['ccf_rating']
            tags = eval(row['subject_tags']) if isinstance(row['subject_tags'], str) else row['subject_tags']
            journal_map[name] = (rating, tags)
    else:  # JSONL
        with open(journals_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    name = data['journal_name']
                    rating = data['ccf_rating']
                    tags = data['subject_tags']
                    journal_map[name] = (rating, tags)
    return journal_map

def search_papers_by_venue(venue: str, limit: int = 50, year_range: Tuple[int, int] = (2020, 2026), sort: str = "year:desc") -> List[Dict]:
    """搜索指定期刊的论文，返回原始 paper 对象列表"""
    query = f"venue:{venue} year:{year_range[0]}-{year_range[1]}"
    url = f"{BASE_URL}/paper/search"
    params = {"query": query, "limit": min(limit, 100), "fields": "title,abstract,venue,year,externalIds,authors", "sort": sort}
    papers = []
    offset = 0
    while True:
        params["offset"] = offset
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("data", [])
            if not batch:
                break
            papers.extend(batch)
            if len(batch) < params["limit"]:
                break
            offset += params["limit"]
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"Error searching venue {venue}: {e}")
            break
    return papers

def fetch_papers_for_combination(
    journal_map: Dict[str, Tuple[str, List[str]]],
    domain: str,
    ccf_level: str,
    count: int,
    max_per_journal: int = 8,
    min_abstract_len: int = 50
) -> List[Dict]:
    """针对单个 (domain, ccf_level) 组合爬取指定数量的论文"""
    target_journals = []
    for name, (rating, tags) in journal_map.items():
        if rating == ccf_level and domain in tags:
            target_journals.append(name)

    if not target_journals:
        print(f"警告：领域 [{domain}] 等级 {ccf_level} 无匹配期刊，跳过")
        return []

    print(f"领域 [{domain}] 等级 {ccf_level}：共 {len(target_journals)} 个期刊")
    collected = []
    seen_titles = set()

    for journal in tqdm(target_journals, desc=f"爬取 {domain} {ccf_level}", leave=False):
        if len(collected) >= count:
            break
        papers = search_papers_by_venue(journal, limit=max_per_journal)
        for paper in papers:
            if len(collected) >= count:
                break
            title = paper.get("title", "").strip()
            if not title or title in seen_titles:
                continue
            abstract = paper.get("abstract", "")
            if not abstract or len(abstract) < min_abstract_len:
                continue
            year = paper.get("year")
            if not year or year < 2019:
                continue
            # 宽松匹配 venue（可选）
            paper_venue = paper.get("venue", "")
            if journal.lower() not in paper_venue.lower() and paper_venue.lower() not in journal.lower():
                # 如果完全不匹配，尝试用 API 返回的 venue（但保留期刊名不变）
                pass
            ext_ids = paper.get("externalIds", {})
            doi = ext_ids.get("DOI", "")
            result = {
                "title": title,
                "abstract": abstract,
                "venue": journal,
                "year": year,
                "ccf_level": ccf_level,
                "research_area": [domain],
                "external_ids": {"doi": doi} if doi else {}
            }
            collected.append(result)
            seen_titles.add(title)
            time.sleep(REQUEST_DELAY / 2)

    # 如果不够，放宽年份范围重试（只对已用期刊的前半部分）
    if len(collected) < count:
        remaining = count - len(collected)
        for journal in target_journals[:len(target_journals)//2]:
            if remaining <= 0:
                break
            papers = search_papers_by_venue(journal, limit=10, year_range=(2015, 2026))
            for paper in papers:
                if remaining <= 0:
                    break
                title = paper.get("title", "").strip()
                if title in seen_titles:
                    continue
                abstract = paper.get("abstract", "")
                if not abstract or len(abstract) < min_abstract_len:
                    continue
                collected.append({
                    "title": title,
                    "abstract": abstract,
                    "venue": journal,
                    "year": paper.get("year", 0),
                    "ccf_level": ccf_level,
                    "research_area": [domain],
                    "external_ids": {"doi": paper.get("externalIds", {}).get("DOI", "")}
                })
                seen_titles.add(title)
                remaining -= 1
                time.sleep(REQUEST_DELAY)
    return collected[:count]

def main():
    parser = argparse.ArgumentParser(description="多领域多等级论文爬虫")
    parser.add_argument("--journals", help="期刊文件路径（CSV/JSONL）",default="/Users/qian/PycharmProjects/paper/data/processed/journals.jsonl")
    parser.add_argument("--domains", nargs="+", help="CCF 领域列表，例如 '计算机网络' '人工智能'",default=[
          "人机交互与普适计算",
          "数据库/数据挖掘/内容检索",
          "计算机图形学与多媒体"
      ])
    parser.add_argument("--ccf_levels", nargs="+", choices=["A","B","C"], help="CCF 等级列表，例如 A B",default=["A"])
    parser.add_argument("--counts", nargs="+", type=int, help="每个组合需要的论文数量，顺序与 --domains×--ccf_levels 匹配，也可只给一个数字（所有组合相同）",default=[3])
    parser.add_argument("--output", default="/Users/qian/PycharmProjects/paper/data/evaluation/papers.jsonl", help="输出文件")
    args = parser.parse_args()

    # 加载期刊
    journal_map = load_journals(args.journals)
    print(f"加载 {len(journal_map)} 个期刊")

    # 解析 counts
    total_combos = len(args.domains) * len(args.ccf_levels)
    if len(args.counts) == 1:
        per_combo_count = args.counts[0]
        counts = [per_combo_count] * total_combos
    elif len(args.counts) == total_combos:
        counts = args.counts
    else:
        raise ValueError(f"counts 参数数量应为 1 或 {total_combos}，当前 {len(args.counts)}")

    # 遍历所有组合
    all_papers = []
    idx = 0
    for domain in args.domains:
        for level in args.ccf_levels:
            needed = counts[idx]
            idx += 1
            if needed <= 0:
                continue
            print(f"\n=== 爬取组合: 领域={domain}, 等级={level}, 数量={needed} ===")
            papers = fetch_papers_for_combination(journal_map, domain, level, needed)
            all_papers.extend(papers)
            print(f"实际获得 {len(papers)} 篇")

    # 输出 JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"\n总计爬取 {len(all_papers)} 篇论文，已保存至 {args.output}")

if __name__ == "__main__":
    main()
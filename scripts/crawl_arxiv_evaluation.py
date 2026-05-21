#!/usr/bin/env python3
"""
从 DBLP + Semantic Scholar 采集 CCF 期刊论文用于评估

目标：10个领域 × 3个等级 × 2篇 = 60篇
只采集有 arXiv 版本且可直接下载的论文
"""

import json
import time
import re
import os
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

S2_API_KEY = "s2k-AeDASl4PzLGaxug6gQ9aNSlOQM10hmxa5IUVj4Ma"
S2_HEADERS = {"x-api-key": S2_API_KEY}

TARGET_AREAS = [
    "人工智能",
    "计算机网络",
    "计算机体系结构/并行与分布计算/存储系统",
    "数据库/数据挖掘/内容检索",
    "计算机图形学与多媒体",
    "软件工程/系统软件/程序设计语言",
    "计算机科学理论",
    "网络与信息安全",
    "人机交互与普适计算",
    "交叉/综合/新兴",
]

def load_journals():
    with open('data/journals_ccf.jsonl') as f:
        return [json.loads(line) for line in f]

def get_area_level_journals():
    """按领域和等级组织期刊"""
    journals = load_journals()
    area_level = {}
    for j in journals:
        for tag in j.get('subject_tags', []):
            if tag in TARGET_AREAS:
                level = j.get('ccf_rating', 'C')
                if tag not in area_level:
                    area_level[tag] = {}
                if level not in area_level[tag]:
                    area_level[tag][level] = []
                area_level[tag][level].append(j['journal_name'])
    return area_level

def search_semantic_scholar_by_journal(journal_name: str, area: str, level: str, limit: int = 10) -> list:
    """按期刊名搜索 Semantic Scholar，优先返回有 arXiv PDF 的论文"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": journal_name,
        "limit": limit,
        "fields": "title,abstract,year,venue,journal,externalIds,openAccessPdf,citations",
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=S2_HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = []
            for p in data.get('data', []):
                # 优先找有 arXiv 的
                ext = p.get('externalIds', {})
                arxiv_id = ext.get('ArXiv')
                oa = p.get('openAccessPdf', {})
                pdf_url = oa.get('url') if oa else None

                if not arxiv_id and not pdf_url:
                    # 没有 arXiv 且没有开放 PDF，跳过
                    continue

                papers.append({
                    'title': p.get('title', ''),
                    'abstract': p.get('abstract', '') or '',
                    'year': p.get('year'),
                    'venue': p.get('venue') or '',
                    'journal': journal_name,
                    'arxiv_id': arxiv_id,
                    'pdf_url': pdf_url or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None),
                    'area': area,
                    'level': level,
                })
            return papers
        except Exception as e:
            print(f"    S2 搜索失败: {e}")
            time.sleep(3)
    return []

def download_pdf(url: str, output_path: str) -> bool:
    """下载 PDF"""
    if not url:
        return False
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code in [403, 404, 429]:
                return False
            resp.raise_for_status()
            ct = resp.headers.get('content-type', '')
            if 'pdf' not in ct.lower() and 'octet-stream' not in ct.lower():
                return False
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
        except:
            time.sleep(2)
    return False

def main():
    area_level = get_area_level_journals()

    os.makedirs('data/evaluation/papers', exist_ok=True)
    os.makedirs('data/evaluation', exist_ok=True)

    all_papers = []
    downloaded = set()
    paper_idx = 0

    print("开始采集论文...")
    print(f"目标: 10领域 × 3等级 × 2篇 = 60篇\n")

    for area in TARGET_AREAS:
        if area not in area_level:
            print(f"[跳过] {area}: 无期刊")
            continue
        for level in ['A', 'B', 'C']:
            journals = area_level[area].get(level, [])
            if not journals:
                print(f"[跳过] {area}[{level}]: 无期刊")
                continue

            needed = 2  # 每等级需要 2 篇

            for j in journals:
                if needed <= 0:
                    break

                print(f"\n[{area}] [{level}] {j}")
                papers = search_semantic_scholar_by_journal(j, area, level, limit=15)

                if not papers:
                    print(f"  无结果")
                    continue

                # 去重已下载的
                papers = [p for p in papers if p.get('arxiv_id') and p['arxiv_id'] not in downloaded]

                for p in papers:
                    if needed <= 0:
                        break
                    arxiv_id = p.get('arxiv_id')
                    if not arxiv_id:
                        continue

                    pdf_path = f"data/evaluation/papers/{arxiv_id}.pdf"
                    print(f"  [{paper_idx+1}] 下载 {arxiv_id}: {p['title'][:45]}...")

                    # 优先用 arXiv PDF
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    success = download_pdf(pdf_url, pdf_path)

                    if not success and p.get('pdf_url'):
                        # 尝试 S2 返回的 PDF URL
                        success = download_pdf(p['pdf_url'], pdf_path)

                    if success:
                        downloaded.add(arxiv_id)
                        paper_idx += 1
                        needed -= 1

                        meta = {
                            'title': p['title'],
                            'abstract': p.get('abstract', ''),
                            'venue': j,
                            'year': p.get('year'),
                            'ccf_level': level,
                            'research_area': [area],
                            'external_ids': {'arXiv': arxiv_id},
                            'pdf_url': pdf_url,
                            'pdf_path': pdf_path,
                        }
                        all_papers.append(meta)
                        print(f"    成功 ({needed} 剩余)")
                    else:
                        print(f"    失败")

                    time.sleep(1)

    # 保存
    with open('data/evaluation/papers_metadata.jsonl', 'w', encoding='utf-8') as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\n\n=== 完成 ===")
    print(f"共 {len(all_papers)} 篇论文")

    from collections import Counter
    areas = Counter(p['research_area'][0] for p in all_papers)
    levels = Counter(p['ccf_level'] for p in all_papers)
    print(f"\n领域分布: {dict(areas)}")
    print(f"等级分布: {dict(levels)}")

if __name__ == "__main__":
    main()
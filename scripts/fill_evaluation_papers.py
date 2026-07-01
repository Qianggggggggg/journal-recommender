#!/usr/bin/env python3
"""
补充缺失类别 - 严格匹配CCF期刊venue
"""

import json
import time
import os
import requests

S2_API_KEY = os.environ.get("S2_API_KEY", "")
S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

TARGET_AREAS = [
    "人工智能", "计算机网络", "计算机体系结构/并行与分布计算/存储系统",
    "数据库/数据挖掘/内容检索", "计算机图形学与多媒体",
    "软件工程/系统软件/程序设计语言", "计算机科学理论",
    "网络与信息安全", "人机交互与普适计算", "交叉/综合/新兴"
]

def load_journals():
    with open('data/journals_ccf.jsonl') as f:
        return [json.loads(line) for line in f]

def get_current_papers():
    with open('data/evaluation/papers_metadata.jsonl') as f:
        papers = [json.loads(line) for line in f]
    result = {}
    for p in papers:
        area = p['research_area'][0]
        level = p['ccf_level']
        arxiv = p.get('external_ids', {}).get('arXiv', '')
        key = (area, level)
        if key not in result:
            result[key] = []
        if arxiv:
            result[key].append(arxiv)
    return result

def get_target_journals(area, level):
    journals = load_journals()
    result = []
    for j in journals:
        if area in j.get('subject_tags', []) and level == j.get('ccf_rating'):
            result.append(j['journal_name'])
    return result

def search_s2_by_journal_name(journal_name):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": journal_name,
        "limit": 30,
        "fields": "title,abstract,year,venue,journal,externalIds,openAccessPdf",
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
                ext = p.get('externalIds', {})
                arxiv_id = ext.get('ArXiv')
                if not arxiv_id:
                    continue
                venue = p.get('venue') or (p.get('journal', {}).get('name') if p.get('journal') else '')
                papers.append({
                    'title': p.get('title', ''),
                    'abstract': p.get('abstract', '') or '',
                    'year': p.get('year'),
                    'venue': venue,
                    'arxiv_id': arxiv_id,
                })
            return papers
        except Exception as e:
            print(f"    S2搜索失败: {e}")
            time.sleep(3)
    return []

def verify_and_download(arxiv_id, target_journal, current_arxiv):
    if arxiv_id in current_arxiv:
        return None

    try:
        # 查询S2确认venue
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        resp = requests.get(url, headers=S2_HEADERS, params={"fields": "venue,journal"}, timeout=15)
        if resp.status_code != 200:
            return None
        s2_data = resp.json()
        s2_venue = s2_data.get('venue') or (s2_data.get('journal', {}).get('name') if s2_data.get('journal') else '')

        # 严格验证venue匹配目标期刊
        target_keywords = [w for w in target_journal.split() if len(w) > 2]
        match_count = sum(1 for kw in target_keywords if kw.lower() in s2_venue.lower())

        if match_count < len(target_keywords) * 0.6:
            print(f"    venue不匹配: {s2_venue[:40]}")
            return None

        # 下载PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = f"data/evaluation/papers/{arxiv_id}.pdf"

        resp = requests.get(pdf_url, timeout=60, stream=True)
        if resp.status_code in [403, 404]:
            return None
        resp.raise_for_status()

        with open(pdf_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        return {
            'arxiv_id': arxiv_id,
            'venue': s2_venue,
            'pdf_path': pdf_path,
        }
    except Exception as e:
        print(f"    下载失败: {e}")
        return None

def main():
    current = get_current_papers()

    to_fill = [
        ("计算机网络", "B"),
        ("计算机体系结构/并行与分布计算/存储系统", "A"),
        ("数据库/数据挖掘/内容检索", "A"),
        ("计算机科学理论", "A"),
        ("计算机科学理论", "B"),
        ("网络与信息安全", "C"),
        ("交叉/综合/新兴", "A"),
    ]

    print(f"需要补充: {len(to_fill)} 个类别\n")

    os.makedirs('data/evaluation/papers', exist_ok=True)
    new_papers = []

    for area, level in to_fill:
        print(f"\n[{area}] [{level}]")
        journals = get_target_journals(area, level)

        found = False
        for j in journals:
            if found:
                break
            print(f"  搜索: {j}")

            papers = search_s2_by_journal_name(j)
            print(f"    找到 {len(papers)} 篇")
            time.sleep(1)

            for sp in papers:
                arxiv_id = sp['arxiv_id']
                result = verify_and_download(arxiv_id, j, current.get((area, level), []))

                if result:
                    print(f"    成功: {arxiv_id}")

                    try:
                        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
                        resp = requests.get(url, headers=S2_HEADERS, params={"fields": "title,abstract,year,venue"}, timeout=15)
                        if resp.status_code == 200:
                            s2_data = resp.json()
                            title = s2_data.get('title', sp['title'])
                            abstract = s2_data.get('abstract', '') or sp.get('abstract', '')
                            year = s2_data.get('year', sp.get('year'))
                            venue = s2_data.get('venue') or (s2_data.get('journal', {}).get('name') if s2_data.get('journal') else j)

                            new_papers.append({
                                'title': title,
                                'abstract': abstract,
                                'venue': venue,
                                'year': year,
                                'ccf_level': level,
                                'research_area': [area],
                                'external_ids': {'arXiv': arxiv_id},
                                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                                'pdf_path': result['pdf_path'],
                            })
                    except:
                        pass

                    found = True
                    break

                time.sleep(0.5)

            if not found:
                print(f"    未找到合适论文")

        if not found:
            print(f"  [警告] 无法找到")

    # 保存
    with open('data/evaluation/papers_metadata.jsonl') as f:
        existing = [json.loads(line) for line in f]

    all_papers = existing + new_papers

    with open('data/evaluation/papers_metadata.jsonl', 'w') as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\n\n=== 完成 ===")
    print(f"新增: {len(new_papers)} 篇")
    print(f"总计: {len(all_papers)} 篇")

    from collections import Counter
    areas = Counter(p['research_area'][0] for p in all_papers)
    levels = Counter(p['ccf_level'] for p in all_papers)
    print(f"\n领域: {dict(areas)}")
    print(f"等级: {dict(levels)}")

if __name__ == "__main__":
    main()
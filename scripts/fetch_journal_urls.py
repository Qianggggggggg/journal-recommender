#!/usr/bin/env python3
"""
从 DBLP 获取期刊的真实 submission_url 和 homepage_url
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup

INPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
SLEEP_INTERVAL = 2

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}


def get_urls_from_dblp(journal_id):
    """从 DBLP 页面获取外部链接"""
    url = f'https://dblp.org/db/journals/{journal_id}/'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            links = {
                'acm': [],
                'ieee': [],
                'springer': [],
                'elsevier': [],
                'wiley': [],
                'other': []
            }

            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                if 'dl.acm.org' in href:
                    links['acm'].append(href)
                elif 'ieeexplore.ieee.org' in href:
                    links['ieee'].append(href)
                elif 'link.springer.com' in href:
                    links['springer'].append(href)
                elif 'sciencedirect.com' in href:
                    links['elsevier'].append(href)
                elif 'wiley.com' in href:
                    links['wiley'].append(href)
                elif href.startswith('http'):
                    links['other'].append(href)

            return links
    except Exception as e:
        pass
    return {'acm': [], 'ieee': [], 'springer': [], 'elsevier': [], 'wiley': [], 'other': []}


def pick_best_url(links):
    """从多个链接中选择最佳 URL"""
    # 优先选择期刊主页，避免搜索页面
    priority_keywords = ['journal/', '/journal/', 'recentissue', 'terms.jsp']

    for key in ['acm', 'ieee', 'springer', 'elsevier', 'wiley']:
        urls = links.get(key, [])
        if not urls:
            continue

        # 优先找包含 'journal' 的 URL（更可能是期刊主页而不是搜索页）
        for url in urls:
            is_journal_page = any(kw in url.lower() for kw in ['journal/', '/journal', 'recentissue'])
            if is_journal_page:
                return url

        # 如果没有主页类型的URL，返回第一个
        return urls[0]

    # 返回其他链接
    if links.get('other'):
        return links['other'][0]

    return ''


def process_journals():
    """处理所有期刊，获取 URL"""
    # 读取所有期刊
    journals = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            journals.append(json.loads(line))

    print(f"Loaded {len(journals)} journals")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Sleep interval: {SLEEP_INTERVAL}s")
    print("=" * 60)

    # 处理每个期刊
    for i, journal in enumerate(journals, 1):
        journal_id = journal['journal_id']
        full_name = journal['journal_name']

        print(f"\n[{i}/{len(journals)}] {journal_id} - {full_name[:40]}")

        links = get_urls_from_dblp(journal_id)
        print(f"  Links found: ACM={len(links['acm'])}, IEEE={len(links['ieee'])}, Springer={len(links['springer'])}, Elsevier={len(links['elsevier'])}, Wiley={len(links['wiley'])}")

        url = pick_best_url(links)
        if url:
            print(f"  Selected: {url}")
            journal['submission_url'] = url
            journal['homepage_url'] = url
        else:
            print(f"  No URL found")

        # 延时
        time.sleep(SLEEP_INTERVAL)

    # 写入更新后的数据
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for journal in journals:
            f.write(json.dumps(journal, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"输出文件: {OUTPUT_PATH}")

    # 统计
    with_url = sum(1 for j in journals if j['submission_url'])
    print(f"有URL的期刊: {with_url}/{len(journals)}")


if __name__ == "__main__":
    process_journals()
#!/usr/bin/env python3
"""
从 iikx.com 抓取期刊 scope
通过查询 API 搜索期刊，然后抓取对应的 scope 页面
"""

import json
import re
import time
import requests
from bs4 import BeautifulSoup

INPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
SLEEP_INTERVAL = 2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.iikx.com/sci/journal.html",
}


def search_iikx_journal(keyword):
    """用 iikx 的搜索 API 查找期刊，返回可能的技术类期刊 ID 列表"""
    try:
        url = f"https://www.iikx.com/journal/query.json?keyword={requests.utils.quote(keyword)}&pageSize=10"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return []


def fetch_scope_from_page(page_url):
    """从 iikx 期刊页面抓取 scope"""
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code != 200 or len(resp.text) < 500:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 从页面标题提取期刊名
        title = soup.find('title')
        if not title:
            return None
        title_text = title.get_text(strip=True)
        journal_name = title_text.split('_')[0].replace('《', '').replace('》', '').strip()

        # 提取 scope - 在 article-content 里的长段落
        article = soup.find('article', class_='article-content')
        if not article:
            return None

        for p in article.find_all('p'):
            text = p.get_text(strip=True)
            # scope 通常 > 50 字符，以句子形式存在
            if len(text) > 50 and text[0].isupper() and '[' not in text[:5]:
                return text

        return None
    except:
        return None


def try_find_scope(journal_name, journal_id):
    """尝试找到期刊的 scope，先搜索，再找页面"""

    # 先用搜索 API 找
    results = search_iikx_journal(journal_name)
    if not results:
        return None

    # 在结果中找匹配度最高的
    best_match = None
    best_score = 0

    # 标准化期刊名用于比较
    norm_name = re.sub(r'[^a-zA-Z0-9\s]', '', journal_name).lower()
    norm_name_clean = re.sub(r'\s+', '', norm_name)

    for item in results:
        fullname = item.get('fullname', '').replace('<span style="color: #e73d4a;">', '').replace('</span>', '')
        abbr = item.get('abbreviation1', '')
        category = item.get('category', '')

        # 只有技术类的才考虑
        # 技术相关的 category 包括: Engineering, Computer Science 等
        if not any(kw in category.lower() for kw in ['engineer', 'computer', 'technol', 'electron', 'electrical', 'communication', 'information']):
            # 但如果期刊名非常匹配，还是可以试试
            item_norm = re.sub(r'[^a-zA-Z0-9\s]', '', fullname).lower().replace(' ', '')
            if item_norm[:15] == norm_name_clean[:15] or norm_name_clean[:10] in item_norm:
                pass  # 继续，但不跳过
            else:
                continue

        # 计算匹配度
        score = 0
        if norm_name_clean[:10] == re.sub(r'\s+', '', item_norm)[:10]:
            score = 100
        elif norm_name_clean[:6] in re.sub(r'\s+', '', item_norm) or re.sub(r'\s+', '', item_norm)[:6] in norm_name_clean:
            score = 80
        elif fullname.lower()[:20] == journal_name.lower()[:20]:
            score = 60

        if score > best_score:
            best_score = score

    # 实际上搜索 API 返回的结果不包含页面 URL
    # 我们只能返回 None，让调用方决定怎么处理
    return None


def main():
    print("从 iikx.com 搜索期刊 scope...")
    print("=" * 60)

    # 读取数据
    journals = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            journals.append(json.loads(line))

    print(f"Loaded {len(journals)} journals")

    # 先测试几个搜索
    test_names = [
        "IEEE Transactions on Dependable and Secure Computing",
        "IEEE Transactions on Information Forensics and Security",
        "Journal of Cryptology",
        "Computers & Security",
        "Designs, Codes and Cryptography",
    ]

    print("\n测试搜索:")
    for name in test_names:
        results = search_iikx_journal(name)
        print(f"\n'{name}':")
        for r in results[:3]:
            fullname = r.get('fullname', '').replace('<span style="color: #e73d4a;">', '').replace('</span>', '')
            print(f"  [{r.get('category','')}] {fullname}")

    # 统计
    with_scope = sum(1 for j in journals if j.get('scope_text') and j['scope_text'] not in ['', '暂无scope'])
    print(f"\n目前有 scope 的期刊: {with_scope}/{len(journals)}")


if __name__ == "__main__":
    main()
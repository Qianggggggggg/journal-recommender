#!/usr/bin/env python3
"""
期刊 Scope 爬虫脚本 V2
通过多种渠道获取每个期刊的真实 scope
"""

import csv
import json
import time
import re
import sys
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# 配置
CSV_PATH = "/Users/qian/Downloads/deepseek_csv_20260520_7a4972.csv"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
SLEEP_INTERVAL = 3  # 每次请求间隔秒数

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

DBLP_BASE = "https://dblp.org/db/journals/"


def load_csv_journals(csv_path):
    journals = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 6:
                seq, abbr, full_name, ccf_class, jtype, area = row[0], row[1], row[2], row[3], row[4], row[5]
                journals.append({
                    'seq': seq.strip(),
                    'abbr': abbr.strip() if abbr else '',
                    'full_name': full_name.strip(),
                    'ccf_class': ccf_class.strip(),
                    'area': area.strip(),
                })
    return journals


def generate_journal_id(journal):
    abbr = journal['abbr']
    full_name = journal['full_name']
    if abbr:
        return abbr.lower().replace(' ', '_').replace('/', '_')
    else:
        name = re.sub(r'[^a-zA-Z0-9]', '', full_name)
        return name.lower()


def fetch_from_dblp(journal_id, full_name):
    """从 DBLP 页面获取 scope 和官网链接"""
    url = f"{DBLP_BASE}{journal_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 查找官方期刊链接
            external_links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                if 'dl.acm.org' in href or 'ieeexplore' in href or 'springer.com' in href or 'wiley.com' in href:
                    external_links.append(href)

            # DBLP 的 meta description 通常只是 "Bibliographic content"，不太有用
            meta = soup.find('meta', attrs={'name': 'description'})
            desc = meta.get('content', '') if meta else ''

            return external_links, url
    except Exception as e:
        pass
    return [], url


def fetch_from_acm(journal_id):
    """从 ACM 获取 scope"""
    url = f"https://dl.acm.org/journal/{journal_id.lower()}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 尝试多种方式获取 scope

            # 1. meta description
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                desc = meta['content'].strip()
                if 'Bibliographic' not in desc and len(desc) > 30:
                    return desc, url

            # 2. og:description
            og = soup.find('meta', property='og:description')
            if og and og.get('content'):
                return og['content'].strip(), url

            # 3. 查找包含 "Scope" 或 "Aim" 的 section
            for section in soup.find_all(['div', 'section']):
                text = section.get_text()
                if 'aim' in text.lower() and 'scope' in text.lower():
                    # 提取相关段落
                    paragraphs = section.find_all('p')
                    for p in paragraphs:
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 100:
                            return p_text, url
    except Exception as e:
        pass
    return None, None


def fetch_from_ieee(journal_id):
    """从 IEEE 获取 scope"""
    # IEEE 期刊页面的 URL 格式比较复杂，尝试直接搜索
    url = f"https://ieeexplore.ieee.org/xplore/terms.jsp"

    # 获取搜索页面找期刊
    search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?searchWithin=Periodical+Name&value1={quote(journal_id)}&highlight=true&returnFacets=ALL"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                return meta['content'].strip(), search_url
    except:
        pass

    # fallback: IEEE Xplore 通用描述
    fallback_url = "https://ieeexplore.ieee.org/xplore/terms.jsp"
    return "暂无scope", fallback_url


def fetch_from_crossref(full_name):
    """从 Crossref API 获取期刊描述"""
    try:
        url = f"https://api.crossref.org/journals"
        params = {'query': full_name, 'rows': 1}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get('message', {}).get('items', [])
            if items:
                item = items[0]
                desc = item.get('description', '')
                if desc and len(desc) > 30:
                    return desc.strip(), "https://api.crossref.org"
    except:
        pass
    return None, None


def fetch_from_springer(journal_id, full_name):
    """从 Springer 获取 scope"""
    # Springer 的期刊页面
    springer_id = journal_id.replace('_', '-').lower()
    url = f"https://link.springer.com/journal/{springer_id}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 尝试 meta description
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                return meta['content'].strip(), url

            # 尝试 og:description
            og = soup.find('meta', property='og:description')
            if og and og.get('content'):
                return og['content'].strip(), url

            # 尝试从页面内容找 scope
            for text in soup.find_all(text=True):
                if 'aim' in text.lower() and 'scope' in text.lower():
                    parent = text.parent
                    if parent:
                        next_p = parent.find_next_sibling('p')
                        if next_p:
                            return next_p.get_text(strip=True), url
    except:
        pass
    return None, None


def fetch_from_wiley(journal_id, full_name):
    """从 Wiley 获取 scope"""
    url = f"https://onlinelibrary.wiley.com/journal/{journal_id.lower()}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                return meta['content'].strip(), url

            og = soup.find('meta', property='og:description')
            if og and og.get('content'):
                return og['content'].strip(), url
    except:
        pass
    return None, None


def get_scope_for_journal(journal):
    """获取单个期刊的 scope，尝试多种来源"""
    journal_id = generate_journal_id(journal)
    full_name = journal['full_name']

    print(f"  DBLP: {DBLP_BASE}{journal_id}/")
    external_links, dblp_url = fetch_from_dblp(journal_id, full_name)

    # 按优先级尝试各个来源

    # 1. 从 DBLP 找到的外部链接获取
    for link in external_links[:3]:  # 最多试3个
        if 'dl.acm.org' in link:
            print(f"  Trying ACM link: {link}")
            scope, src = fetch_from_acm(journal_id)
            if scope and scope != "暂无scope":
                return scope, link

    # 2. 尝试 ACM 直接
    print(f"  Trying ACM direct")
    scope, src = fetch_from_acm(journal_id)
    if scope and scope != "暂无scope":
        return scope, src

    # 3. 尝试 Crossref
    print(f"  Trying Crossref API")
    scope, src = fetch_from_crossref(full_name)
    if scope:
        return scope, src

    # 4. 尝试 Springer
    print(f"  Trying Springer")
    scope, src = fetch_from_springer(journal_id, full_name)
    if scope:
        return scope, src

    # 5. 尝试 Wiley
    print(f"  Trying Wiley")
    scope, src = fetch_from_wiley(journal_id, full_name)
    if scope:
        return scope, src

    # 6. 如果都失败了，尝试从 DBLP meta description 获取（虽然可能没用）
    if external_links:
        return "暂无scope（请访问官网）", external_links[0]

    return "暂无scope", ""


def process_journals():
    """处理所有期刊"""
    journals = load_csv_journals(CSV_PATH)
    print(f"Loaded {len(journals)} journals from CSV")
    print(f"Output will be written to: {OUTPUT_PATH}")
    print(f"Sleep interval: {SLEEP_INTERVAL}s between requests")
    print("=" * 60)

    # 清空输出文件
    open(OUTPUT_PATH, 'w').close()

    success_count = 0
    fail_count = 0

    for i, journal in enumerate(journals, 1):
        print(f"\n[{i}/{len(journals)}] {journal['full_name']} ({journal['ccf_class']})")

        scope, source_url = get_scope_for_journal(journal)

        if scope != "暂无scope" and not scope.endswith("（请访问官网）"):
            success_count += 1
            print(f"  -> SUCCESS: {scope[:60]}...")
        else:
            fail_count += 1
            print(f"  -> FAILED: {scope}")

        journal_id = generate_journal_id(journal)

        # 简化 scope
        if scope and len(scope) > 500:
            scope = scope[:500] + "..."

        # 确定出版社
        full_name = journal['full_name']
        if 'ACM' in full_name:
            publisher = 'ACM'
        elif 'IEEE' in full_name or 'Transactions' in full_name:
            publisher = 'IEEE'
        elif 'Elsevier' in full_name:
            publisher = 'Elsevier'
        elif 'Springer' in full_name:
            publisher = 'Springer'
        elif 'Wiley' in full_name:
            publisher = 'Wiley'
        else:
            publisher = 'unknown'

        record = {
            "journal_id": journal_id,
            "journal_name": full_name,
            "publisher": publisher,
            "subject_tags": [journal['area']],
            "ccf_rating": journal['ccf_class'],
            "scope_text": scope,
            "submission_url": source_url,
            "homepage_url": source_url,
            "keywords": [],
            "oa_type": "subscription",
            "impact_like_score": 0.0,
            "review_time": "",
            "apc": 0.0
        }

        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        time.sleep(SLEEP_INTERVAL)

    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"成功: {success_count}/{len(journals)}")
    print(f"失败: {fail_count}/{len(journals)}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_journals()
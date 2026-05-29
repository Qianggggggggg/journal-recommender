#!/usr/bin/env python3
"""
扫描 iikx.com 技术分类下的所有期刊页面，收集 scope
对 9600-20000 范围内的页面进行扫描，找出 CS 相关期刊并抓取 scope
"""

import json
import re
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
SCAN_START = 9600
SCAN_END = 20000
NUM_WORKERS = 3  # 减少并发，避免被封
REQUEST_DELAY = 0.5  # 每次请求后等待秒数

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.iikx.com/sci/technology/",
}

# CS 相关的关键词，用于过滤期刊
CS_KEYWORDS = [
    'computer', 'system', 'network', 'security', 'software', 'data', 'algorithm',
    'image', 'speech', 'signal', 'fuzzy', 'neural', 'learning', 'vision', 'graphic',
    'multimedia', 'web', 'database', 'internet', 'transaction', 'algorithm',
    'computing', 'computational', 'intelligent', 'knowledge', 'information',
    'distributed', 'parallel', 'performance', 'communication', 'wireless',
    'electronic', 'circuit', 'very large scale', 'vlsi', 'embedded',
    'artificial', 'intelligence', 'pattern', 'recognition', 'automation',
    'control', 'robot', 'sensor', 'visual', 'audio', 'video', 'image',
    'programming', 'language', 'modeling', 'simulation', 'graphics',
    'security', 'cryptography', 'cryptographic', 'privacy',
    'protocol', 'routing', 'topology', 'bandwidth', 'protocol',
]


def fetch_page(pid):
    """抓取单个页面，返回 (pid, journal_name, scope_text) 或 None"""
    url = f"https://www.iikx.com/sci/technology/{pid}.html"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code != 200 or len(resp.text) < 1500:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 检查是否 404 页面
        title = soup.find('title')
        if not title:
            return None
        title_text = title.get_text(strip=True)
        if '404' in title_text[:20] or '出错了' in title_text or '错误' in title_text[:10]:
            return None

        # 提取期刊名
        journal_name = title_text.split('_')[0].replace('《', '').replace('》', '').strip()
        if not journal_name or len(journal_name) < 5:
            return None

        # 检查是否是 CS 相关
        name_lower = journal_name.lower()
        is_cs = any(kw in name_lower for kw in CS_KEYWORDS)
        if not is_cs:
            return None

        # 提取 scope
        article = soup.find('article', class_='article-content')
        scope = None
        if article:
            for p in article.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50 and text[0].isupper() and '[' not in text[:5]:
                    scope = text
                    break

        if not scope:
            return None

        return (pid, journal_name, scope)

    except Exception as e:
        return None


def main():
    print(f"扫描 iikx.com 期刊范围: {SCAN_START}-{SCAN_END}")
    print(f"并发数: {NUM_WORKERS}, 延迟: {REQUEST_DELAY}s")
    print("=" * 60)

    # 先读取所有期刊用于匹配
    journals = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            journals.append(json.loads(line))

    # 构建标准化名称到期刊的映射
    name_to_journal = {}
    for j in journals:
        if j.get('scope_text') and j['scope_text'] not in ['', '暂无scope']:
            continue
        norm = re.sub(r'[^a-zA-Z0-9\s]', '', j['journal_name']).lower()
        name_to_journal[norm] = j

    print(f"待匹配期刊: {len(name_to_journal)} 个")

    # 扫描所有页面
    all_found = {}  # norm_name -> scope
    scanned = 0
    valid = 0

    pids_to_scan = list(range(SCAN_START, SCAN_END))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(fetch_page, pid): pid for pid in pids_to_scan}

        for future in as_completed(futures):
            pid = futures[future]
            scanned += 1

            if scanned % 500 == 0:
                print(f"  已扫描 {scanned}/{SCAN_END - SCAN_START}, 找到 {len(all_found)} 个 CS 期刊 scope")

            try:
                result = future.result()
            except Exception:
                continue

            if result:
                pid, journal_name, scope = result
                valid += 1
                norm = re.sub(r'[^a-zA-Z0-9\s]', '', journal_name).lower()
                all_found[norm] = scope

    print(f"\n扫描完成: 共扫描 {scanned} 个页面, 找到 {valid} 个 CS 期刊")
    print(f"准备匹配的 scope: {len(all_found)} 个")

    # 匹配到期刊
    updated = 0
    matched = []

    for norm_name, scope in all_found.items():
        # 精确匹配
        if norm_name in name_to_journal:
            j = name_to_journal[norm_name]
            if not j.get('scope_text') or j['scope_text'] in ['', '暂无scope']:
                j['scope_text'] = scope
                updated += 1
                matched.append((j['journal_id'], j['journal_name'][:40], scope[:60]))
                del name_to_journal[norm_name]
                continue

        # 模糊匹配
        name_clean = re.sub(r'\s+', '', norm_name)
        for key, j in list(name_to_journal.items()):
            key_clean = re.sub(r'\s+', '', key)
            if len(name_clean) > 5 and len(key_clean) > 5:
                # 检查开头或结尾是否匹配
                if name_clean[:12] == key_clean[:12] or name_clean[-12:] == key_clean[-12:]:
                    if not j.get('scope_text') or j['scope_text'] in ['', '暂无scope']:
                        j['scope_text'] = scope
                        updated += 1
                        matched.append((j['journal_id'], j['journal_name'][:40], scope[:60]))
                        del name_to_journal[key]
                        break

    print(f"\n匹配完成: 新增 {updated} 个 scope")

    # 写回文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for journal in journals:
            f.write(json.dumps(journal, ensure_ascii=False) + '\n')

    # 统计
    with_scope = sum(1 for j in journals if j.get('scope_text') and j['scope_text'] not in ['', '暂无scope'])
    print(f"有 scope 的期刊: {with_scope}/{len(journals)}")

    if matched:
        print(f"\n成功匹配的期刊 (前10个):")
        for jid, name, sc in matched[:10]:
            print(f"  {jid}: {name}")
            print(f"    scope: {sc}...")


if __name__ == "__main__":
    main()
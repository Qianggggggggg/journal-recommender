#!/usr/bin/env python3
"""
从CCF推荐目录PDF中提取期刊URL - 改进版
"""

import fitz
import re
import json

PDF_PATH = "/Users/qian/Downloads/第七版中国计算机学会推荐国际学术会议和期刊目录（正式版）.pdf"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"

# 期刊URL映射
journal_urls = {}


def extract_v4():
    """改进的提取方法，处理多行期刊名称和无缩写期刊"""
    doc = fitz.open(PDF_PATH)
    all_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        all_text += page.get_text() + "\n"

    doc.close()

    # 将连续空白替换为单个空格
    all_text = re.sub(r' +', ' ', all_text)

    # 查找所有 dblp 期刊链接（排除会议）
    # 模式: ... 出版社 URL
    pattern = r'(ACM|IEEE|Elsevier|Springer|Wiley|Science China|IOS Press)\s+(https?://dblp[^\s]+)'

    matches = re.findall(pattern, all_text)

    for publisher, url in matches:
        if '/conf/' in url:
            continue  # 跳过会议

        # 从URL提取期刊ID作为备用
        url_match = re.search(r'/journals/([^/]+)/', url)
        if url_match:
            journal_id_from_url = url_match.group(1)
            journal_urls[journal_id_from_url] = url

    return journal_urls


def fuzzy_match(journal_id, urls_dict):
    """模糊匹配期刊ID"""
    journal_id_lower = journal_id.lower().replace('_', '')

    # 直接匹配
    if journal_id in urls_dict:
        return urls_dict[journal_id]

    # 尝试匹配URL中的journal_id
    for url_id, url in urls_dict.items():
        if url_id.lower().replace('_', '') == journal_id_lower:
            return url

    return None


def main():
    print("从CCF PDF提取期刊URL...")

    urls = extract_v4()
    print(f"从PDF提取到 {len(urls)} 个URL")

    # 读取现有JSONL
    journals = []
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            journals.append(json.loads(line))

    print(f"JSONL中有 {len(journals)} 个期刊")

    # 更新URL
    updated = 0
    for j in journals:
        result = fuzzy_match(j['journal_id'], urls)
        if result:
            j['submission_url'] = result
            j['homepage_url'] = result
            updated += 1

    print(f"更新了 {updated} 个期刊的URL")

    # 写回文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for j in journals:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')

    print(f"已保存到 {OUTPUT_PATH}")

    # 显示未匹配的期刊
    no_url = [j['journal_id'] for j in journals if not j['submission_url']]
    if no_url:
        print(f"\n未找到URL的期刊 ({len(no_url)}):")
        for nid in no_url:
            # 查找对应的期刊全称
            jname = next((j['journal_name'] for j in journals if j['journal_id'] == nid), nid)
            print(f"  - {nid}: {jname}")


if __name__ == "__main__":
    main()
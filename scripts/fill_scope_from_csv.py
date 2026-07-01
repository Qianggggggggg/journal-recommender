#!/usr/bin/env python3
"""
将 deepseek_csv 中的 scope 填入 journals_output.jsonl
按 journal_id 匹配，填入前检查内容正确性
"""

import json
import csv

JSONL_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
CSV_PATH = "/Users/qian/Downloads/deepseek_csv_20260523_6f4e7c.csv"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"

# 读取 CSV 中的 scope
csv_scopes = {}  # journal_id -> scope_text
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 6:
            jid = row[1].strip()
            scope = row[5].strip() if row[5] else ''
            csv_scopes[jid] = scope

print(f"CSV 中有 {len(csv_scopes)} 条 scope 记录")

# 读取 JSONL
journals = []
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        journals.append(json.loads(line))

print(f"JSONL 中有 {len(journals)} 个期刊")

# 更新 scope
updated = 0
checked = 0
skipped = 0

for j in journals:
    jid = j['journal_id']
    if jid in csv_scopes:
        csv_scope = csv_scopes[jid]
        checked += 1

        if csv_scope and csv_scope.lower() not in ['暂无scope', 'unknown', '', 'n/a', 'none', '暂无']:
            # 检查 scope 长度是否合理（至少 20 字符）
            if len(csv_scope) >= 20:
                j['scope_text'] = csv_scope
                updated += 1
                print(f"  [OK] {jid}: {csv_scope[:60]}...")
            else:
                print(f"  [WARN] {jid}: scope 太短 ({len(csv_scope)} chars): {csv_scope[:40]}")
                j['scope_text'] = "unknown"
                skipped += 1
        else:
            j['scope_text'] = "unknown"
            skipped += 1
            print(f"  [EMPTY] {jid}: scope 为空，标注为 unknown")

# 写回文件
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for j in journals:
        f.write(json.dumps(j, ensure_ascii=False) + '\n')

print(f"\n完成!")
print(f"检查了 {checked} 个期刊的 scope")
print(f"更新了 {updated} 个")
print(f"标记为 unknown: {skipped} 个")

# 统计
with_scope = sum(1 for j in journals if j.get('scope_text') and j['scope_text'] not in ['', '暂无scope', 'unknown'])
with_unknown = sum(1 for j in journals if j.get('scope_text', '') == 'unknown')
print(f"\n当前状态:")
print(f"  有真实 scope: {with_scope}/{len(journals)}")
print(f"  标注 unknown: {with_unknown}/{len(journals)}")
print(f"  无 scope: {len(journals) - with_scope - with_unknown}/{len(journals)}")
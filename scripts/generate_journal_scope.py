"""从 OpenAlex API 获取真实 scope_text"""
import json
import time
import urllib.request
import urllib.parse

def get_openalex_info(journal_name: str) -> dict:
    """从 OpenAlex 获取期刊信息"""
    try:
        encoded = urllib.parse.quote(journal_name)
        url = f"https://api.openalex.org/journals?filter=display_name.search:{encoded}&per_page=1"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'JournalRecommender/1.0 (mailto:author@example.com)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        if data.get('results') and len(data['results']) > 0:
            journal = data['results'][0]
            topics = journal.get('topics', [])

            # 提取前8个topics作为scope
            topic_names = [t['display_name'] for t in topics[:8]]

            return {
                'found': True,
                'topics': topic_names,
                'topics_count': len(topics)
            }
    except Exception as e:
        pass

    return {'found': False, 'topics': [], 'topics_count': 0}


def build_scope_from_openalex(journal_name: str, openalex_info: dict) -> str:
    """基于 OpenAlex topics 构建 scope"""
    topics = openalex_info.get('topics', [])
    if not topics:
        return ""

    # 构建自然的scope描述
    if len(topics) == 1:
        scope = f"Covers all aspects of {topics[0]}."
    elif len(topics) == 2:
        scope = f"Focuses on {topics[0]} and {topics[1]}."
    else:
        scope = f"Covers {topics[0]}, {topics[1]}, {topics[2]}, and related topics including {topics[3]}, {topics[4]}, {topics[5]}."

    return scope


def main():
    journals = []
    with open("data/journals_ccf.jsonl") as f:
        for line in f:
            journals.append(json.loads(line))

    # 检查哪些需要更新（使用模板的）
    template_journals = [j for j in journals if "Key topics include" in j.get("scope_text", "")]
    print(f"使用模板的期刊: {len(template_journals)} 条")

    updated = 0
    failed = 0

    for i, j in enumerate(template_journals):
        name = j["journal_name"]
        area = j.get("subject_tags", [""])[0] if j.get("subject_tags") else ""
        ccf = j.get("ccf_rating", "")
        is_conference = "conference" in j.get("target_paper_type", [])

        info = get_openalex_info(name)
        source = ""

        if info['found'] and info['topics']:
            scope = build_scope_from_openalex(name, info)
            source = "OpenAlex"
        else:
            # OpenAlex 找不到，保持模板
            scope = j["scope_text"]
            source = "Template(keep)"
            failed += 1

        j["scope_text"] = scope

        print(f"{i+1:3d}. [{source}] {j['journal_id']}: {scope[:60]}..." if scope else f"{i+1:3d}. [Empty] {j['journal_id']}")

        updated += 1
        time.sleep(0.5)  # API 限流

        # 每50个暂停一下
        if (i + 1) % 50 == 0:
            print(f"\n--- 已处理 {i+1} 条，休息2秒 ---")
            time.sleep(2)

    print(f"\n完成！更新: {updated}, 失败: {failed}")

    # 保存
    with open("data/journals_ccf.jsonl", "w") as f:
        for j in journals:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # 统计
    still_template = sum(1 for j in journals if "Key topics include" in j.get("scope_text", ""))
    empty = sum(1 for j in journals if not j.get("scope_text"))
    print(f"剩余模板scope: {still_template}, 空scope: {empty}")

    # 示例
    print("\n示例:")
    with open("data/journals_ccf.jsonl") as f:
        for i, line in enumerate(f):
            if i > 190 and i < 205:
                j = json.loads(line)
                print(f"  [{j['ccf_rating']}] {j['journal_id']}: {j['scope_text'][:80]}...")


if __name__ == "__main__":
    main()
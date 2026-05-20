"""从 OpenAlex API 获取真实 scope_text"""
import json
import time
import urllib.request
import urllib.parse

def get_openalex_topics(journal_name: str) -> list:
    """从 OpenAlex 获取期刊的研究主题"""
    try:
        encoded = urllib.parse.quote(journal_name)
        url = f"https://api.openalex.org/journals?filter=display_name.search:{encoded}&per_page=1&fields=id,display_name,topics"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'JournalRecommender/1.0 (mailto:author@example.com)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        if data.get('results') and len(data['results']) > 0:
            journal = data['results'][0]
            topics = journal.get('topics', [])
            if topics:
                # 取前10个最重要的topics
                topic_names = [t['display_name'] for t in topics[:10]]
                return topic_names
    except Exception as e:
        pass
    return []


# 测试
test = get_openalex_topics("IEEE Transactions on Pattern Analysis and Machine Intelligence")
print(f"TPAMI topics: {test}")
time.sleep(0.5)

test = get_openalex_topics("Journal of Machine Learning Research")
print(f"JMLR topics: {test}")
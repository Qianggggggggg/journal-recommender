"""DOAJ 数据采集脚本"""
import json
import time
import httpx
from typing import List, Dict, Optional
from pathlib import Path

from src.journals.journal_model import Journal


def get_doaj_journals(subject: str = "computer science", max_results: int = 200) -> List[Dict]:
    """从 DOAJ 采集期刊数据"""
    journals = []
    page = 1
    page_size = 100

    while len(journals) < max_results:
        url = f"https://doaj.org/api/v2/search/journals/{subject}"
        params = {
            "pageSize": page_size,
            "page": page,
        }

        try:
            response = httpx.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                bibjson = item.get("bibjson", {})

                # 提取 APC 信息
                apc = 0
                apc_url = None
                if "apc" in bibjson:
                    apc_info = bibjson["apc"]
                    if isinstance(apc_info, dict):
                        apc = apc_info.get("max", [0])[0] if apc_info.get("max") else 0
                        apc_url = apc_info.get("url")

                # 提取审稿周期 (如果没有就留空)
                review_time = ""

                # 提取oa类型
                oa = bibjson.get("oa", False)
                if bibjson.get("boai"):  # Budapest Open Access Initiative
                    oa_type = "full_oa"
                elif oa:
                    oa_type = "full_oa"
                else:
                    oa_type = "subscription"

                journal_data = {
                    "journal_id": f"doaj-{item.get('id', '')}",
                    "journal_name": bibjson.get("title", ""),
                    "publisher": bibjson.get("publisher", {}).get("name", ""),
                    "subject_tags": _map_lcc_to_tags(bibjson.get("subject", [])),
                    "keywords": bibjson.get("keywords", [])[:10],
                    "scope_text": bibjson.get("description", ""),
                    "oa_type": oa_type,
                    "homepage_url": bibjson.get("website"),
                    "submission_url": bibjson.get("submission_url", ""),
                    "apc": apc,
                    "review_time": review_time,
                }

                journals.append(journal_data)

            print(f"  Page {page}: got {len(results)} journals, total: {len(journals)}")

            if len(results) < page_size:
                break

            page += 1
            time.sleep(0.5)  # 避免请求过快

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

    return journals[:max_results]


def _map_lcc_to_tags(subjects: List[Dict]) -> List[str]:
    """将 LCC 学科代码映射为我们的标签"""
    tag_mapping = {
        "QA1-939": "mathematics",
        "QA75-76": "computing",
        "QA76": "computer science",
        "Q1": "mathematics",
        "Q1-1": "mathematics",
        "T1": "technology",
        "TK1": "electrical engineering",
        "TR1": "photography",
    }

    tags = []
    for subj in subjects:
        code = subj.get("code", "")
        term = subj.get("term", "").lower()

        if "artificial intelligence" in term or "ai" in code.lower():
            tags.append("ai")
        if "machine learning" in term:
            tags.append("ai")
        if "computer vision" in term or "image processing" in term:
            tags.append("cv")
        if "natural language" in term or "computational linguistics" in term:
            tags.append("nlp")
        if "software" in term or "programming" in term:
            tags.append("se")
        if "network" in term or "internet" in term:
            tags.append("network")
        if "security" in term or "cryptography" in term:
            tags.append("security")
        if "database" in term or "data mining" in term:
            tags.append("db")
        if "theory" in term or "computation" in term:
            tags.append("theory")

    if not tags:
        tags = ["other"]

    return list(set(tags))[:5]


def crawl_specific_topics(topics: List[str], max_per_topic: int = 30) -> List[Dict]:
    """采集特定主题的期刊"""
    all_journals = {}
    topic_map = {
        "artificial intelligence": ["ai", "machine learning", "neural network"],
        "computer vision": ["computer vision", "image processing"],
        "natural language processing": ["nlp", "text mining", "computational linguistics"],
        "software engineering": ["software engineering", "software", "programming"],
        "network security": ["network security", "cybersecurity", "cryptography"],
        "database": ["database", "data mining", "information retrieval"],
        "theory": ["computational theory", "algorithm", "automata"],
    }

    for topic in topics:
        print(f"\nCrawling topic: {topic}")
        keywords = topic_map.get(topic.lower(), [topic])

        for kw in keywords:
            print(f"  Keyword: {kw}")
            try:
                response = httpx.get(
                    f"https://doaj.org/api/v2/search/journals/{kw.replace(' ', '%20')}",
                    params={"pageSize": max_per_topic},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("results", []):
                        jid = item.get("id")
                        if jid not in all_journals:
                            bibjson = item.get("bibjson", {})
                            all_journals[jid] = {
                                "journal_id": f"doaj-{jid}",
                                "journal_name": bibjson.get("title", ""),
                                "publisher": bibjson.get("publisher", {}).get("name", ""),
                                "subject_tags": _map_lcc_to_tags(bibjson.get("subject", [])),
                                "keywords": bibjson.get("keywords", [])[:10],
                                "scope_text": bibjson.get("description", ""),
                                "oa_type": "full_oa" if bibjson.get("oa") else "subscription",
                                "homepage_url": bibjson.get("website"),
                                "submission_url": bibjson.get("submission_url", ""),
                                "apc": 0,
                                "review_time": "",
                            }
                time.sleep(0.3)
            except Exception as e:
                print(f"    Error: {e}")

    return list(all_journals.values())


def save_journals(journals: List[Dict], output_path: str):
    """保存期刊数据到 JSONL 文件"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for j in journals:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"Saved {len(journals)} journals to {output_path}")


if __name__ == "__main__":
    print("=== DOAJ Journal Crawler ===")

    # 方法1: 采集计算机科学类期刊
    print("\n1. Crawling computer science journals...")
    journals = get_doaj_journals("computer science", max_results=200)
    print(f"Got {len(journals)} journals from DOAJ")

    # 方法2: 采集特定主题期刊
    print("\n2. Crawling AI/ML journals...")
    ai_journals = crawl_specific_topics(["artificial intelligence", "machine learning"], max_per_topic=50)
    print(f"Got {len(ai_journals)} AI/ML journals")

    # 合并去重
    all_journals = {j["journal_id"]: j for j in journals + ai_journals}
    print(f"\nTotal unique journals: {len(all_journals)}")

    # 保存
    output_file = "data/raw/doaj_journals.jsonl"
    save_journals(list(all_journals.values()), output_file)

    print("\nDone!")
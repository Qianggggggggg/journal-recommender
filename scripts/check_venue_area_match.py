#!/usr/bin/env python3
"""验证评测论文的 venue 与 research_area 是否匹配"""
import json
import sys
from pathlib import Path

def load_journals(path: str) -> dict:
    """加载期刊数据，建立 venue -> subject_tags 的映射"""
    journals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            # 使用 journal_name 作为 key（不区分大小写匹配）
            journals[j["journal_name"].lower()] = j
            # 也建立 venue 简称映射
            if "abbreviation" in j:
                journals[j["abbreviation"].lower()] = j
    return journals


def load_papers(path: str) -> list:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def normalize_venue(venue: str) -> str:
    """标准化 venue 名称用于匹配"""
    if not venue:
        return ""
    return venue.lower().strip()


def find_journal_by_venue(venue: str, journals: dict) -> tuple[dict | None, str]:
    """根据 venue 查找匹配的期刊"""
    if not venue:
        return None, ""

    v_lower = venue.lower().strip()

    # 直接匹配
    if v_lower in journals:
        return journals[v_lower], v_lower

    # 部分匹配 - venue 包含 journal_name
    for j_name, j in journals.items():
        if j_name in v_lower or v_lower in j_name:
            return j, j_name

    return None, ""


def check_mismatch(paper: dict, journals: dict) -> tuple[bool, str, str, str]:
    """检查论文的 venue 与 research_area 是否匹配

    Returns: (is_mismatch, reason, venue, subject_tags)
    """
    title = paper.get("title", "")[:50]
    venue = paper.get("venue", "")
    metadata_area = paper.get("research_area", [""])[0] if paper.get("research_area") else ""

    # 查找匹配的期刊
    matched_journal, matched_name = find_journal_by_venue(venue, journals)

    if not matched_journal:
        return False, f"venue 未找到匹配: {venue}", venue, ""

    journal_subject = matched_journal.get("subject_tags", [])

    # 检查 metadata_area 是否在期刊的 subject_tags 中
    if metadata_area in journal_subject:
        return False, "", venue, journal_subject

    # 不匹配
    return True, f"metadata_area '{metadata_area}' 不在期刊 subject_tags {journal_subject} 中", venue, journal_subject


def main():
    journal_path = "/Users/qian/PycharmProjects/paper/data/processed/journals.jsonl"
    paper_path = "data/evaluation/papers_metadata.jsonl"
    output_path = "data/evaluation/results/metadata_venue_check.json"

    journals = load_journals(journal_path)
    papers = load_papers(paper_path)

    print(f"加载 {len(journals)} 本期刊, {len(papers)} 篇论文\n")

    mismatches = []
    matched = []
    not_found = []

    for paper in papers:
        title = paper.get("title", "")[:50]
        venue = paper.get("venue", "")
        metadata_area = paper.get("research_area", [""])[0] if paper.get("research_area") else ""

        is_mismatch, reason, matched_venue, subject_tags = check_mismatch(paper, journals)

        entry = {
            "title": title,
            "venue": venue,
            "metadata_area": metadata_area,
            "matched_journal_venue": matched_venue,
            "journal_subject_tags": subject_tags,
            "reason": reason,
        }

        if not matched_venue:
            not_found.append(entry)
        elif is_mismatch:
            mismatches.append(entry)
        else:
            matched.append(entry)

    print(f"匹配: {len(matched)}")
    print(f"venue未找到: {len(not_found)}")
    print(f"领域不匹配: {len(mismatches)}")

    # 保存结果
    result = {
        "matched": matched,
        "not_found": not_found,
        "mismatches": mismatches,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存: {output_path}")

    # 打印不匹配的例子
    if mismatches:
        print(f"\n不匹配的例子 (前10个):")
        for m in mismatches[:10]:
            print(f"  论文: {m['title']}")
            print(f"  venue: {m['venue']} -> {m['matched_journal_venue']}")
            print(f"  metadata_area: {m['metadata_area']}")
            print(f"  journal subject_tags: {m['journal_subject_tags']}")
            print(f"  原因: {m['reason']}")
            print()


if __name__ == "__main__":
    main()
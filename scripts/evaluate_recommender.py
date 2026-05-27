#!/usr/bin/env python3
"""期刊推荐系统评估脚本"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationResult:
    """评估结果"""
    total_count: int
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int

    level_match_count: int
    strong_count: int
    strong_level_match: int
    medium_count: int
    medium_level_match: int
    weak_count: int
    weak_level_match: int

    by_area: dict[str, dict]


def load_ground_truth(path: str) -> list[dict]:
    """加载 ground truth 数据"""
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def load_ccf_journals(ccf_path: str = "data/journals_ccf.jsonl") -> dict[str, dict]:
    """加载 CCF 期刊列表（用于匹配）"""
    journals = {}
    with open(ccf_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            journals[j["journal_name"]] = j
    return journals


def get_paper_quality_level(strength: float) -> str:
    """根据 paper_strength 划分质量等级"""
    if strength >= 0.65:
        return "strong"
    elif strength >= 0.50:
        return "medium"
    else:
        return "weak"


def get_expected_ccf_levels(quality_level: str) -> list[str]:
    """根据论文质量等级，返回期望的 CCF 等级列表"""
    if quality_level == "strong":
        return ["A", "B"]
    elif quality_level == "medium":
        return ["B", "C"]
    else:
        return ["C", "N/A"]


def is_level_match(paper_strength: float, recommended_ccf: str) -> bool:
    """判断论文质量与推荐期刊 CCF 等级是否匹配"""
    quality_level = get_paper_quality_level(paper_strength)
    expected_levels = get_expected_ccf_levels(quality_level)
    return recommended_ccf in expected_levels


def evaluate_recommender(ground_truth_path: str, recommender_func, ccf_path: str = "data/journals_ccf.jsonl"):
    """
    评估推荐系统

    Args:
        ground_truth_path: ground truth 数据路径
        recommender_func: 推荐函数，输入 (title, abstract) 返回推荐期刊列表
        ccf_path: CCF 期刊数据路径
    """
    papers = load_ground_truth(ground_truth_path)
    ccf_journals = load_ccf_journals(ccf_path)

    result = EvaluationResult(
        total_count=len(papers),
        hit_at_1=0,
        hit_at_3=0,
        hit_at_5=0,
        level_match_count=0,
        strong_count=0,
        strong_level_match=0,
        medium_count=0,
        medium_level_match=0,
        weak_count=0,
        weak_level_match=0,
        by_area=defaultdict(lambda: {"total": 0, "hit": 0}),
    )

    print(f"开始评估，共 {len(papers)} 篇论文...")

    for paper in papers:
        title = paper["title"]
        abstract = paper.get("abstract", "")
        published_journal = paper.get("published_journal", "")
        expected_ccf = paper.get("ccf_rating", "N/A")
        research_area = paper.get("research_area", "")

        # 运行推荐
        try:
            recommendations = recommender_func(title, abstract)
        except Exception as e:
            print(f"推荐失败: {title[:30]}... - {e}", file=sys.stderr)
            continue

        # 计算 Hit@K
        recommended_journals = [rec.get("journal_name", "") for rec in recommendations[:5]]

        if published_journal in recommended_journals[:1]:
            result.hit_at_1 += 1
        if published_journal in recommended_journals[:3]:
            result.hit_at_3 += 1
        if published_journal in recommended_journals:
            result.hit_at_5 += 1

        # 计算 Level Match Rate
        paper_strength = estimate_paper_strength(abstract)

        quality_level = get_paper_quality_level(paper_strength)
        expected_levels = get_expected_ccf_levels(quality_level)

        if expected_ccf in expected_levels:
            result.level_match_count += 1

        # 按质量等级统计
        if quality_level == "strong":
            result.strong_count += 1
            if expected_ccf in expected_levels:
                result.strong_level_match += 1
        elif quality_level == "medium":
            result.medium_count += 1
            if expected_ccf in expected_levels:
                result.medium_level_match += 1
        else:
            result.weak_count += 1
            if expected_ccf in expected_levels:
                result.weak_level_match += 1

        # 按领域统计
        result.by_area[research_area]["total"] += 1
        if published_journal in recommended_journals:
            result.by_area[research_area]["hit"] += 1

        print(f"评估: {title[:40]}... (实际: {published_journal})")

    return result


def estimate_paper_strength(abstract: str) -> float:
    """
    估算论文质量强度（简化版本）

    实际应用中应调用 PaperQualityAssessor
    这里基于摘要长度和质量信号词估计
    """
    if not abstract:
        return 0.4

    # 质量信号词
    strong_signals = ["state-of-the-art", "novel", "significant", "improves", "achieves"]
    weak_signals = ["preliminary", "limited", "初步", "简单"]

    strength = 0.5  # 默认中等

    for signal in strong_signals:
        if signal.lower() in abstract.lower():
            strength += 0.1

    for signal in weak_signals:
        if signal.lower() in abstract.lower():
            strength -= 0.1

    return max(0.2, min(0.9, strength))


def print_report(result: EvaluationResult):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)

    print(f"\n总论文数：{result.total_count}")

    # Hit@K
    print(f"\n--- Hit@K ---")
    print(f"Top-1: {result.hit_at_1}/{result.total_count} ({result.hit_at_1*100/result.total_count:.1f}%)")
    print(f"Top-3: {result.hit_at_3}/{result.total_count} ({result.hit_at_3*100/result.total_count:.1f}%)")
    print(f"Top-5: {result.hit_at_5}/{result.total_count} ({result.hit_at_5*100/result.total_count:.1f}%)")

    # Level Match Rate
    print(f"\n--- Level Match Rate ---")
    total_match = result.level_match_count
    print(f"Overall: {total_match}/{result.total_count} ({total_match*100/result.total_count:.1f}%)")

    if result.strong_count > 0:
        print(f"强论文 (n={result.strong_count}): {result.strong_level_match}/{result.strong_count} ({result.strong_level_match*100/result.strong_count:.1f}%)")
    if result.medium_count > 0:
        print(f"中论文 (n={result.medium_count}): {result.medium_level_match}/{result.medium_count} ({result.medium_level_match*100/result.medium_count:.1f}%)")
    if result.weak_count > 0:
        print(f"弱论文 (n={result.weak_count}): {result.weak_level_match}/{result.weak_count} ({result.weak_level_match*100/result.weak_count:.1f}%)")

    # 按领域分布
    print(f"\n--- 按领域分布 ---")
    for area, stats in result.by_area.items():
        hit_rate = stats["hit"] * 100 / stats["total"] if stats["total"] > 0 else 0
        print(f"{area}: {stats['hit']}/{stats['total']} ({hit_rate:.1f}%)")

    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估推荐系统")
    parser.add_argument("--input", "-i", default="data/evaluation/ground_truth.jsonl",
                        help="ground truth 数据路径")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API 地址")

    args = parser.parse_args()

    # 推荐函数（调用本地 API）
    def recommender_func(title: str, abstract: str):
        import requests
        import json
        response = requests.post(
            f"{args.api_url}/api/recommend",
            json={"title": title, "abstract": abstract, "mode": "full", "top_k": 5},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=120,
        )
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("recommendations", [])

    result = evaluate_recommender(args.input, recommender_func)
    print_report(result)


if __name__ == "__main__":
    main()
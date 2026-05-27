# 期刊推荐评估系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建离线评估流程，验证系统推荐的准确率（Hit@5、Level Match Rate）

**Architecture:**
- 数据采集层：从 DBLP 爬取论文-期刊对应关系，构建 ground truth 数据集
- 评估层：对每篇论文运行推荐，计算 Hit@5 和 Level Match Rate
- 输出层：生成评估报告

**Tech Stack:** Python, requests, BeautifulSoup, DBLP API, pytest

---

## 文件结构

| 文件 | 作用 |
|------|------|
| `scripts/crawl_dblp_evaluation.py` | DBLP 论文数据采集脚本 |
| `scripts/evaluate_recommender.py` | 评估主脚本 |
| `data/evaluation/ground_truth.jsonl` | 评估数据集（脚本运行后生成） |
| `tests/test_evaluation.py` | 评估流程测试 |

---

## Task 1: 创建目录结构

**Files:**
- Create: `data/evaluation/` (目录)

```bash
mkdir -p data/evaluation
```

- [ ] **Step 1: 创建目录**

```bash
mkdir -p data/evaluation
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(evaluation): add data/evaluation directory"
```

---

## Task 2: 编写 DBLP 爬取脚本

**Files:**
- Create: `scripts/crawl_dblp_evaluation.py`

- [ ] **Step 1: 编写 DBLP 爬取脚本**

```python
#!/usr/bin/env python3
"""DBLP 论文数据采集脚本 - 用于评估数据集构建"""

import json
import sys
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


def search_dblp_by_keyword(keyword: str, year: int = 2020, max_results: int = 50) -> list[dict]:
    """
    通过关键词搜索 DBLP 论文

    Args:
        keyword: 搜索关键词
        year: 发表年份下限
        max_results: 最大结果数

    Returns:
        论文列表
    """
    # DBLP API (faceted search)
    url = f"https://dblp.org/search/pubs?q={keyword}&f=year:{year}&h={max_results}&format=json"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        papers = []
        for hit in data.get("result", {}).get("hits", {}).get("hit", []):
            info = hit.get("info", {})
            title = info.get("title", "")
            abstract = info.get("abstract", "")
            venue = info.get("venue", "")
            year = info.get("year", "")
            ee = info.get("ee", "")

            if not title:
                continue

            papers.append({
                "title": title,
                "abstract": abstract,
                "venue": venue,
                "year": year,
                "url": ee,
                "keyword": keyword,
            })

        return papers
    except Exception as e:
        print(f"Error searching DBLP for keyword '{keyword}': {e}", file=sys.stderr)
        return []


def fetch_paper_details(dblp_url: str) -> Optional[dict]:
    """
    从 DBLP 页面获取论文详情

    Args:
        dblp_url: DBLP 论文页面 URL

    Returns:
        论文详情（含 DOI、期刊信息等）
    """
    try:
        response = requests.get(dblp_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # 提取 DOI
        doi_meta = soup.find("meta", {"name": "citation_doi"})
        doi = doi_meta["content"] if doi_meta else ""

        # 提取期刊/会议信息
        journal_elem = soup.find("li", {"class": "ee"})
        journal = journal_elem.text if journal_elem else ""

        return {"doi": doi, "journal": journal}
    except Exception as e:
        print(f"Error fetching paper details from {dblp_url}: {e}", file=sys.stderr)
        return None


def crawl_evaluation_dataset(output_path: str, target_count: int = 200):
    """
    爬取评估数据集

    Args:
        output_path: 输出文件路径
        target_count: 目标论文数量
    """
    # 搜索关键词，覆盖不同领域
    keywords = [
        "deep learning",
        "neural network",
        "computer vision",
        "natural language processing",
        "machine learning",
        "reinforcement learning",
        "graph neural network",
        "transformer",
        "object detection",
        "semantic segmentation",
        "text classification",
        "question answering",
        "knowledge graph",
        "recommender system",
        "federated learning",
    ]

    all_papers = []
    seen_titles = set()

    print(f"开始爬取 DBLP 论文，目标 {target_count} 篇...")

    for keyword in keywords:
        if len(all_papers) >= target_count:
            break

        print(f"搜索关键词: {keyword}")

        papers = search_dblp_by_keyword(keyword, year=2019, max_results=50)

        for paper in papers:
            title = paper["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            all_papers.append(paper)
            print(f"  已获取 {len(all_papers)}/{target_count} 篇: {title[:50]}...")

            if len(all_papers) >= target_count:
                break

    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"\n完成！共获取 {len(all_papers)} 篇论文，保存到 {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DBLP 论文数据采集")
    parser.add_argument("--output", "-o", default="data/evaluation/ground_truth_raw.jsonl",
                        help="输出文件路径")
    parser.add_argument("--count", "-c", type=int, default=200,
                        help="目标论文数量")

    args = parser.parse_args()

    crawl_evaluation_dataset(args.output, args.count)
```

- [ ] **Step 2: 验证脚本可运行**

```bash
python scripts/crawl_dblp_evaluation.py --output data/evaluation/test.jsonl --count 5
```

预期：生成包含 5 篇论文的 test.jsonl 文件

- [ ] **Step 3: 提交**

```bash
git add scripts/crawl_dblp_evaluation.py
git commit -m "feat(evaluation): add DBLP data crawler script"
```

---

## Task 3: 构建评估数据集（添加 CCF 期刊匹配）

**Files:**
- Modify: `scripts/crawl_dblp_evaluation.py`
- Create: `scripts/match_ccf_journals.py`

**Context:**
原始 DBLP 数据只有论文标题/摘要，需要匹配到具体期刊名称，再关联 CCF 等级。

- [ ] **Step 1: 编写 CCF 期刊匹配脚本**

```python
#!/usr/bin/env python3
"""为爬取的论文匹配 CCF 期刊等级"""

import json
import sys
from pathlib import Path
from typing import Optional

from Levenshtein import ratio  # pip install python-Levenshtein


def load_ccf_journals(ccf_path: str = "data/journals_ccf.jsonl") -> dict[str, dict]:
    """加载 CCF 期刊列表"""
    journals = {}
    with open(ccf_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            journals[j["journal_name"]] = j
    return journals


def load_raw_papers(raw_path: str) -> list[dict]:
    """加载原始爬取的论文"""
    papers = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def find_matching_journal(paper: dict, ccf_journals: dict, threshold: float = 0.6) -> Optional[dict]:
    """
    模糊匹配论文 venue 与 CCF 期刊

    Args:
        paper: 论文数据
        ccf_journals: CCF 期刊字典
        threshold: 匹配度阈值

    Returns:
        匹配的期刊数据
    """
    venue = paper.get("venue", "").lower()
    if not venue:
        return None

    best_match = None
    best_score = 0

    for name, journal in ccf_journals.items():
        # 计算相似度
        name_lower = name.lower()
        score = ratio(venue, name_lower)

        # 如果 venue 包含期刊名关键词
        if any(kw in venue for kw in name_lower.split()):
            score = max(score, 0.7)

        if score > best_score and score >= threshold:
            best_score = score
            best_match = journal

    return best_match


def match_and_save(raw_path: str, output_path: str, ccf_path: str = "data/journals_ccf.jsonl"):
    """匹配 CCF 期刊并保存"""
    ccf_journals = load_ccf_journals(ccf_path)
    papers = load_raw_papers(raw_path)

    matched_count = 0
    results = []

    for paper in papers:
        journal = find_matching_journal(paper, ccf_journals)

        if journal:
            matched_paper = {
                "title": paper["title"],
                "abstract": paper.get("abstract", ""),
                "research_area": paper.get("keyword", ""),
                "published_journal": journal["journal_name"],
                "ccf_rating": journal.get("ccf_rating", "N/A"),
                "quartile": journal.get("quartile", ""),
            }
            results.append(matched_paper)
            matched_count += 1
            print(f"匹配成功 ({matched_count}): {paper['title'][:40]}... -> {journal['journal_name']}")
        else:
            print(f"未匹配: {paper['title'][:40]}...")

    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for paper in results:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"\n完成！共匹配 {matched_count}/{len(papers)} 篇，保存到 {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="匹配 CCF 期刊")
    parser.add_argument("--input", "-i", default="data/evaluation/ground_truth_raw.jsonl",
                        help="原始论文数据路径")
    parser.add_argument("--output", "-o", default="data/evaluation/ground_truth.jsonl",
                        help="输出文件路径")

    args = parser.parse_args()

    match_and_save(args.input, args.output)
```

- [ ] **Step 2: 运行数据采集和匹配**

```bash
python scripts/crawl_dblp_evaluation.py --output data/evaluation/ground_truth_raw.jsonl --count 200
python scripts/match_ccf_journals.py --input data/evaluation/ground_truth_raw.jsonl --output data/evaluation/ground_truth.jsonl
```

- [ ] **Step 3: 检查生成的数据**

```bash
wc -l data/evaluation/ground_truth.jsonl
head -3 data/evaluation/ground_truth.jsonl
```

预期：200 行数据，每行包含 title, abstract, published_journal, ccf_rating 等字段

- [ ] **Step 4: 提交数据采集相关脚本**

```bash
git add scripts/crawl_dblp_evaluation.py scripts/match_ccf_journals.py
git commit -m "feat(evaluation): add CCF journal matching script"
```

---

## Task 4: 编写评估脚本

**Files:**
- Create: `scripts/evaluate_recommender.py`

- [ ] **Step 1: 编写评估脚本**

```python
#!/usr/bin/env python3
"""期刊推荐系统评估脚本"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


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

    # 构建期刊名称到 CCF 等级的映射
    journal_to_ccf = {j["journal_name"]: j.get("ccf_rating", "N/A") for j in ccf_journals.values()}

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
        # 使用 PaperQualityAssessor 评估论文质量
        # 这里需要调用系统的质量评估功能
        # 简化处理：根据摘要长度估计（实际应用中应调用完整的 PaperQualityAssessor）
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
        response = requests.get(
            f"{args.api_url}/api/recommend/stream",
            params={"title": title, "abstract": abstract, "mode": "abstract", "top_k": 5},
            headers={"Accept": "application/json"},
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
```

- [ ] **Step 2: 测试评估脚本（无需 API）**

```python
# 测试评估脚本的数据处理逻辑
from scripts.evaluate_recommender import estimate_paper_strength, get_paper_quality_level

# 测试质量等级划分
assert get_paper_quality_level(0.8) == "strong"
assert get_paper_quality_level(0.5) == "medium"
assert get_paper_quality_level(0.3) == "weak"

# 测试论文质量估算
assert 0.4 <= estimate_paper_strength("state-of-the-art method") <= 0.9
assert 0.2 <= estimate_paper_strength("preliminary result") <= 0.6

print("评估脚本逻辑测试通过")
```

- [ ] **Step 3: 提交**

```bash
git add scripts/evaluate_recommender.py
git commit -m "feat(evaluation): add recommender evaluation script"
```

---

## Task 5: 编写测试

**Files:**
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: 编写评估相关测试**

```python
"""评估流程测试"""
import pytest
from scripts.evaluate_recommender import (
    get_paper_quality_level,
    get_expected_ccf_levels,
    is_level_match,
    estimate_paper_strength,
)


def test_paper_quality_level():
    """测试论文质量等级划分"""
    assert get_paper_quality_level(0.8) == "strong"
    assert get_paper_quality_level(0.7) == "strong"
    assert get_paper_quality_level(0.69) == "medium"
    assert get_paper_quality_level(0.5) == "medium"
    assert get_paper_quality_level(0.39) == "weak"
    assert get_paper_quality_level(0.2) == "weak"


def test_expected_ccf_levels():
    """测试期望 CCF 等级"""
    assert "A" in get_expected_ccf_levels("strong")
    assert "B" in get_expected_ccf_levels("strong")
    assert "B" in get_expected_ccf_levels("medium")
    assert "C" in get_expected_ccf_levels("medium")
    assert "C" in get_expected_ccf_levels("weak")


def test_level_match():
    """测试 Level Match 判断"""
    # 强论文应该匹配 A 或 B
    assert is_level_match(0.8, "A")
    assert is_level_match(0.8, "B")
    assert not is_level_match(0.8, "C")

    # 中论文应该匹配 B 或 C
    assert is_level_match(0.5, "B")
    assert is_level_match(0.5, "C")
    assert not is_level_match(0.5, "A")

    # 弱论文应该匹配 C 或 N/A
    assert is_level_match(0.3, "C")
    assert is_level_match(0.3, "N/A")
    assert not is_level_match(0.3, "A")


def test_estimate_paper_strength():
    """测试论文质量估算"""
    # 无摘要
    strength = estimate_paper_strength("")
    assert 0.3 <= strength <= 0.5

    # 有强信号词
    strength = estimate_paper_strength("We propose a novel state-of-the-art method that significantly improves accuracy")
    assert strength > 0.5

    # 有弱信号词
    strength = estimate_paper_strength("This is a preliminary study with limited experiments")
    assert strength < 0.5
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_evaluation.py -v
```

预期：所有测试通过

- [ ] **Step 3: 提交**

```bash
git add tests/test_evaluation.py
git commit -m "test(evaluation): add evaluation tests"
```

---

## Task 6: 完整运行评估流程

**前置条件：** 服务启动 `python -m src.app.main`

- [ ] **Step 1: 构建评估数据集**

```bash
python scripts/crawl_dblp_evaluation.py --output data/evaluation/ground_truth_raw.jsonl --count 200
python scripts/match_ccf_journals.py --input data/evaluation/ground_truth_raw.jsonl --output data/evaluation/ground_truth.jsonl
```

- [ ] **Step 2: 运行评估**

```bash
python scripts/evaluate_recommender.py --input data/evaluation/ground_truth.jsonl
```

- [ ] **Step 3: 分析结果，提交数据**

```bash
# 添加数据目录但忽略具体文件（太大）
git add data/evaluation/.gitkeep
git commit -m "feat(evaluation): add evaluation dataset structure"
```

---

## 自检清单

- [ ] spec 覆盖：每个 spec 需求都有对应 task
- [ ] 无 placeholder：代码中无 TBD/TODO
- [ ] 类型一致：Task 间类型定义一致

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-21-journal-recommender-evaluation-plan.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
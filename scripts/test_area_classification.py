#!/usr/bin/env python3
"""领域分类 & 等级匹配专项测试"""
import json
import sys
import os
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.papers.paper_model import PaperInput
from src.papers.paper_parser import PaperParser
from src.papers.quality_assessor import PaperQualityAssessor, PaperQuality
from src.utils.llm import MiniMaxLLM
import yaml


@dataclass
class PaperTestResult:
    title: str
    metadata_area: str = ""
    metadata_ccf_level: str = ""
    predicted_areas: list = field(default_factory=list)
    area_match: bool = False
    # 等级相关
    quality_level: str = ""
    level_match: bool = False
    paper_strength: float = 0.0
    novelty_score: float = 0.0
    rigor_score: float = 0.0
    reproducibility_score: float = 0.0
    significance_score: float = 0.0
    clarity_score: float = 0.0


def load_papers(path: str) -> list:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def normalize(text: str) -> str:
    return text.strip().lower() if text else ""


def evaluate_single_paper(
    paper: dict,
    llm: MiniMaxLLM,
    parser: PaperParser,
    quality_assessor: PaperQualityAssessor,
    prompts: dict,
) -> PaperTestResult:
    """测试单篇论文的领域分类和等级匹配"""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    metadata_area = paper.get("research_area", [""])[0] if paper.get("research_area") else ""
    metadata_ccf_level = paper.get("ccf_level", "").upper()

    result = PaperTestResult(
        title=title[:50],
        metadata_area=metadata_area,
        metadata_ccf_level=metadata_ccf_level,
    )

    paper_input = PaperInput(title=title, abstract=abstract or "", mode="abstract")

    # 解析论文
    try:
        profile = parser.parse(
            paper_input,
            prompts["paper_profile_system"],
            prompts["paper_profile_user"],
        )
    except Exception:
        return result

    # 质量评估
    try:
        quality = quality_assessor.assess(
            paper_input, profile,
            system_prompt=prompts.get("paper_quality_assessor_system", ""),
            user_prompt=prompts.get("paper_quality_assessor_user", ""),
        )
    except Exception:
        return result

    # 领域匹配
    predicted_areas = quality.ccf_research_area or []
    result.predicted_areas = predicted_areas
    result.area_match = normalize(metadata_area) in [normalize(a) for a in predicted_areas]

    # 等级匹配
    result.quality_level = quality.quality_level or ""
    result.level_match = result.quality_level.upper() == metadata_ccf_level if metadata_ccf_level else False

    # 详细分数
    result.paper_strength = quality.paper_strength
    result.novelty_score = quality.novelty_score
    result.rigor_score = quality.rigor_score
    result.reproducibility_score = quality.reproducibility_score
    result.significance_score = quality.significance_score
    result.clarity_score = quality.clarity_score

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="领域分类 & 等级匹配专项测试")
    parser.add_argument("--input", "-i", default="data/evaluation/papers_metadata.jsonl",
                        help="论文元数据路径")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="限制测试论文数量")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="并行线程数")
    args = parser.parse_args()

    # 初始化
    from dotenv import load_dotenv
    load_dotenv(override=True)

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY 未配置")

    llm = MiniMaxLLM(api_key=api_key)
    quality_assessor = PaperQualityAssessor(llm)
    parser = PaperParser(llm)

    # 加载论文
    papers = load_papers(args.input)
    if args.limit:
        papers = papers[:args.limit]

    print(f"加载 {len(papers)} 篇论文，并行 {args.workers} 线程\n")

    # 领域统计
    area_stats = {}
    level_stats = {"A": {"match": 0, "total": 0}, "B": {"match": 0, "total": 0}, "C": {"match": 0, "total": 0}}

    area_match_total = 0
    level_match_total = 0
    total = len(papers)
    all_results = []  # 所有论文的详细结果

    results_lock = threading.Lock()

    def update_stats(result: PaperTestResult):
        nonlocal area_match_total, level_match_total
        with results_lock:
            # 领域匹配
            if result.area_match:
                area_match_total += 1
            else:
                results.append({
                    "title": result.title,
                    "metadata_area": result.metadata_area,
                    "predicted_areas": result.predicted_areas,
                })

            # 收集所有结果用于详细报告
            all_results.append({
                "title": result.title,
                "metadata_area": result.metadata_area,
                "metadata_ccf_level": result.metadata_ccf_level,
                "predicted_areas": result.predicted_areas,
                "area_match": result.area_match,
                "quality_level": result.quality_level,
                "level_match": result.level_match,
                "paper_strength": result.paper_strength,
                "novelty_score": result.novelty_score,
                "rigor_score": result.rigor_score,
                "reproducibility_score": result.reproducibility_score,
                "significance_score": result.significance_score,
                "clarity_score": result.clarity_score,
            })

            # 按领域统计
            area = result.metadata_area or "未知"
            if area not in area_stats:
                area_stats[area] = {"match": 0, "total": 0}
            area_stats[area]["total"] += 1
            if result.area_match:
                area_stats[area]["match"] += 1

            # 等级匹配
            if result.level_match:
                level_match_total += 1

            # 按等级统计
            level = result.metadata_ccf_level
            if level in level_stats:
                level_stats[level]["total"] += 1
                if result.level_match:
                    level_stats[level]["match"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_paper, paper, llm, parser, quality_assessor, prompts
            ): paper
            for paper in papers
        }

        pbar = tqdm(as_completed(futures), total=len(papers), desc="测试", unit="篇")

        processed = 0

        for future in pbar:
            result = future.result()
            update_stats(result)
            processed += 1
            pbar.set_postfix({
                "area": f"{area_match_total}/{processed}({area_match_total*100//processed if processed else 0}%)",
                "level": f"{level_match_total}/{processed}({level_match_total*100//processed if processed else 0}%)",
            })

    # 汇总报告
    print("\n" + "=" * 70)
    print("                         测试报告                          ")
    print("=" * 70)

    print(f"\n📊 总体准确率")
    print(f"   领域匹配: {area_match_total}/{total} ({area_match_total*100/total:.1f}%)")
    print(f"   等级匹配: {level_match_total}/{total} ({level_match_total*100/total:.1f}%)")

    print(f"\n📋 按领域统计")
    print(f"   {'领域':<45} {'匹配':>8} {'总数':>6} {'匹配率':>8}")
    print(f"   {'-'*70}")
    for area, stats in sorted(area_stats.items(), key=lambda x: -x[1]["total"]):
        match_rate = stats["match"] * 100 / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(match_rate / 10) + "░" * (10 - int(match_rate / 10))
        print(f"   {area:<45} {stats['match']:>6}/{stats['total']:<6} {bar} {match_rate:>6.1f}%")

    print(f"\n📋 按CCF等级统计")
    print(f"   {'等级':>6} {'匹配':>8} {'总数':>6} {'匹配率':>8}")
    print(f"   {'-'*30}")
    for level in ["A", "B", "C"]:
        if level in level_stats and level_stats[level]["total"] > 0:
            stats = level_stats[level]
            match_rate = stats["match"] * 100 / stats["total"] if stats["total"] > 0 else 0
            bar = "█" * int(match_rate / 10) + "░" * (10 - int(match_rate / 10))
            print(f"   CCF-{level:<4} {stats['match']:>6}/{stats['total']:<6} {bar} {match_rate:>6.1f}%")

    # 保存报告
    output_dir = "data/evaluation/results/area_classification"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    mismatch_path = os.path.join(output_dir, f"mismatch_{timestamp}.json")

    # 生成文本报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("                         测试报告                          ")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"测试论文数: {total}")
    report_lines.append(f"并行线程数: {args.workers}")
    report_lines.append("")

    report_lines.append("📊 总体准确率")
    report_lines.append(f"   领域匹配: {area_match_total}/{total} ({area_match_total*100/total:.1f}%)")
    report_lines.append(f"   等级匹配: {level_match_total}/{total} ({level_match_total*100/total:.1f}%)")
    report_lines.append("")

    report_lines.append("📋 按领域统计")
    report_lines.append(f"   {'领域':<45} {'匹配':>8} {'总数':>6} {'匹配率':>8}")
    report_lines.append(f"   {'-'*70}")
    for area, stats in sorted(area_stats.items(), key=lambda x: -x[1]["total"]):
        match_rate = stats["match"] * 100 / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(match_rate / 10) + "░" * (10 - int(match_rate / 10))
        report_lines.append(f"   {area:<45} {stats['match']:>6}/{stats['total']:<6} {bar} {match_rate:>6.1f}%")

    report_lines.append("")
    report_lines.append("📋 按CCF等级统计")
    report_lines.append(f"   {'等级':>6} {'匹配':>8} {'总数':>6} {'匹配率':>8}")
    report_lines.append(f"   {'-'*30}")
    for level in ["A", "B", "C"]:
        if level in level_stats and level_stats[level]["total"] > 0:
            stats = level_stats[level]
            match_rate = stats["match"] * 100 / stats["total"] if stats["total"] > 0 else 0
            bar = "█" * int(match_rate / 10) + "░" * (10 - int(match_rate / 10))
            report_lines.append(f"   CCF-{level:<4} {stats['match']:>6}/{stats['total']:<6} {bar} {match_rate:>6.1f}%")

    report_lines.append("")

    # 保存详细结果JSON
    detail_path = os.path.join(output_dir, f"detail_{timestamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 生成详细报告（按等级分组显示每篇论文的评分）
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("                         详细评分报告                          ")
    report_lines.append("=" * 70)
    report_lines.append("")

    for level in ["A", "B", "C"]:
        level_papers = [r for r in all_results if r["metadata_ccf_level"] == level]
        if not level_papers:
            continue
        report_lines.append(f"\n## CCF-{level} 级论文 ({len(level_papers)} 篇)")
        report_lines.append(f"{'标题':<45} {'预测等级':<8} {'匹配':<4} {'strength':<8} {'N':<5} {'R':<5} {'P':<5} {'S':<5} {'C':<5}")
        report_lines.append("-" * 100)
        for p in sorted(level_papers, key=lambda x: -x["paper_strength"]):
            match_str = "✓" if p["level_match"] else "✗"
            report_lines.append(
                f"{p['title']:<45} {p['quality_level']:<8} {match_str:<4} "
                f"{p['paper_strength']:.3f}   "
                f"{p['novelty_score']:.1f}  {p['rigor_score']:.1f}  "
                f"{p['reproducibility_score']:.1f}  {p['significance_score']:.1f}  "
                f"{p['clarity_score']:.1f}"
            )

    report_lines.append("")
    report_lines.append(f"📄 详细评分: {detail_path}")

    # 保存文本报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 保存失败案例JSON（领域不匹配）
    mismatch_only = [r for r in all_results if not r["area_match"]]
    if mismatch_only:
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatch_only, f, ensure_ascii=False, indent=2)

    # 终端输出
    print("\n".join(report_lines))

    if mismatch_only:
        print(f"📄 领域不匹配案例: {mismatch_path}")
    print(f"📄 完整报告: {report_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

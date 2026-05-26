#!/usr/bin/env python3
"""
粗排候选数量 experiment - 寻找最优 top_k

测试不同粗排返回数量对最终推荐效果的影响。
其余参数保持不变（RuleScorer 逻辑、top_k=5 等）。
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.papers.paper_model import PaperInput, PaperProfile
from src.recommender.pipeline import RecommenderPipeline
from src.papers.paper_parser import PaperParser
from src.papers.quality_assessor import PaperQualityAssessor
from src.journals.journal_store import JournalStore
from src.journals.vector_searcher import VectorSearcher, FaissIndex
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.embedding_retriever import EmbeddingRetriever
from src.retriever.candidate_generator import CandidateGenerator
from src.ranker.rule_scorer import RuleScorer
from src.ranker.llm_ranker import LLMRanker
from src.utils.llm import MiniMaxLLM
from src.utils.embedding import OllamaEmbedding
import yaml


def load_papers_metadata(path: str) -> list:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def init_components():
    """初始化各组件"""
    import yaml
    from dotenv import load_dotenv
    load_dotenv(override=True)

    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    store = JournalStore()
    store.load()

    faiss_path = app_config["data"]["faiss_index_path"]
    metadata_path = app_config["data"]["metadata_path"]
    faiss_idx = FaissIndex(faiss_path, metadata_path)
    faiss_idx.load()
    if faiss_idx.is_loaded:
        vector_searcher = VectorSearcher(faiss_idx)
        store.set_vector_searcher(vector_searcher)

    bm25 = BM25Retriever(store)
    bm25.build_index()

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY 未配置")

    llm = MiniMaxLLM(
        api_key=api_key,
        base_url=app_config["minimax"]["base_url"],
        model=app_config["minimax"]["model"],
    )

    embedding_client = OllamaEmbedding(
        base_url=app_config["ollama"]["base_url"],
        model=app_config["ollama"]["embedding_model"],
    )

    embedding_retriever = None
    if store.has_vector_search():
        embedding_retriever = EmbeddingRetriever(store, embedding_client)

    retrieval_config = app_config.get("retrieval", {})
    merge_weights = retrieval_config.get("merge_weights", {"bm25": 0.4, "vector": 0.4, "tag": 0.2})

    generator = CandidateGenerator(store, bm25, embedding_retriever, merge_weights=merge_weights)
    scorer = RuleScorer()
    llm_ranker = LLMRanker(llm, prompts["llm_ranker_system"], prompts["llm_ranker_user"])
    quality_assessor = PaperQualityAssessor(llm)
    parser = PaperParser(llm)

    pipeline = RecommenderPipeline(
        candidate_generator=generator,
        rule_scorer=scorer,
        llm_ranker=llm_ranker,
        quality_assessor=quality_assessor,
    )
    pipeline.parser = parser

    return pipeline, prompts


def evaluate_single_paper(paper, pipeline, prompts, mode, coarse_top_k, final_top_k):
    """评估单篇论文（使用指定的粗排 top_k）"""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    venue = paper.get("venue", "")
    arxiv_id = paper.get("external_ids", {}).get("arXiv", "")

    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text="",
        mode=mode,
    )

    try:
        profile = pipeline.parser.parse(
            paper_input,
            prompts["paper_profile_system"],
            prompts["paper_profile_user"],
        )
    except Exception as e:
        print(f"\n解析失败: {title[:30]}... - {e}")
        return None

    # 直接调用 candidate_generator 以控制 coarse_top_k
    query_text = paper_input.title
    if paper_input.abstract:
        query_text += " " + paper_input.abstract

    candidates = pipeline.candidate_generator.generate(
        query_text, profile, top_k=coarse_top_k, mode=mode
    )
    candidate_journal_names = [j.journal_name for j in candidates] if candidates else []

    # RuleScorer 排序（使用不同的 top_k 选择）
    rule_ranked = pipeline.rule_scorer.rank(
        candidates, profile, oa_preference="any", top_k=min(coarse_top_k, 10)
    )
    rule_ranked_names = [j.journal_name for j, s, r in rule_ranked] if rule_ranked else []

    # 质量调整
    rule_ranked = pipeline._apply_quality_adjustment(rule_ranked, profile)

    # LLMRanker 精排（使用 top_k=final_top_k）
    try:
        llm_ranked, rank_method = pipeline.llm_ranker.rank(
            rule_ranked, profile, top_k=final_top_k
        )
    except Exception as e:
        # LLM 失败时降级
        llm_ranked = [(j, s, r, 0.5) for j, s, r in rule_ranked[:final_top_k]]

    recommendations = llm_ranked
    recommended_journals = [rec[0].journal_name for rec in recommendations]

    # 指标计算
    coarse_hit = venue in candidate_journal_names if venue else False
    coarse_hit_in_rule_top10 = venue in rule_ranked_names[:10] if venue else False
    hit_5 = venue in recommended_journals[:5] if venue else False

    return {
        "title": title[:40],
        "venue": venue,
        "coarse_hit": coarse_hit,
        "coarse_hit_in_rule_top10": coarse_hit_in_rule_top10,
        "hit_5": hit_5,
        "candidate_count": len(candidate_journal_names),
        "rule_ranked_names": rule_ranked_names[:10],
    }


def run_experiment(papers, pipeline, prompts, mode, coarse_top_k, final_top_k, workers):
    """运行实验"""
    coarse_hit_count = 0
    coarse_hit_in_rule_top10_count = 0
    hit_5_count = 0
    paper_results = []

    results_lock = threading.Lock()

    def update(result):
        nonlocal coarse_hit_count, coarse_hit_in_rule_top10_count, hit_5_count
        if result is None:
            return
        with results_lock:
            if result["coarse_hit"]:
                coarse_hit_count += 1
            if result["coarse_hit_in_rule_top10"]:
                coarse_hit_in_rule_top10_count += 1
            if result["hit_5"]:
                hit_5_count += 1
            paper_results.append(result)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_paper,
                paper, pipeline, prompts, mode, coarse_top_k, final_top_k
            ): paper
            for paper in papers
        }

        for future in as_completed(futures):
            result = future.result()
            update(result)

    total = len(papers)
    return {
        "coarse_top_k": coarse_top_k,
        "total": total,
        "coarse_hit": f"{coarse_hit_count}/{total} ({coarse_hit_count*100/total:.1f}%)",
        "coarse_hit_in_rule_top10": f"{coarse_hit_in_rule_top10_count}/{total} ({coarse_hit_in_rule_top10_count*100/total:.1f}%)",
        "hit_5": f"{hit_5_count}/{total} ({hit_5_count*100/total:.1f}%)",
        "coarse_hit_count": coarse_hit_count,
        "coarse_hit_in_rule_top10_count": coarse_hit_in_rule_top10_count,
        "hit_5_count": hit_5_count,
        "paper_results": paper_results,
    }


def print_detailed_analysis(all_results):
    """打印详细分析"""
    print("\n" + "=" * 80)
    print("各论文详细分析")
    print("=" * 80)

    for result in all_results:
        coarse_k = result["coarse_top_k"]
        print(f"\n--- coarse_top_k = {coarse_k} ---")
        for pr in result["paper_results"]:
            status = "✓" if pr["hit_5"] else "✗"
            coarse_status = "✓" if pr["coarse_hit"] else "✗"
            rule_status = "✓" if pr["coarse_hit_in_rule_top10"] else "✗"
            print(f"  {status} {pr['title'][:35]:<35} | venue={pr['venue'][:20]:<20} | coarse={coarse_status} rule={rule_status}")


def generate_markdown_table(all_results):
    """生成 Markdown 表格"""
    header = """| K (粗排) | 粗排命中率 | RuleScorer Top10 命中率 | 最终 Hit@5 |
|:-------:|:---------:|:-------------------:|:---------:|
"""
    rows = ""
    best_k = None
    best_hit5 = -1

    for result in all_results:
        k = result["coarse_top_k"]
        coarse_rate = result["coarse_hit"]
        rule_rate = result["coarse_hit_in_rule_top10"]
        hit5 = result["hit_5"]

        hit5_count = result["hit_5_count"]
        if hit5_count > best_hit5:
            best_hit5 = hit5_count
            best_k = k

        rows += f"| {k} | {coarse_rate} | {rule_rate} | {hit5} |\n"

    table = header + rows

    # 建议
    suggestion = ""
    # 检查是否有 K 值在粗排命中率仍为 100% 的前提下，RuleScorer Top10 命中率高于 50%
    candidates = []
    for result in all_results:
        if result["coarse_hit_count"] == result["total"]:  # 粗排命中率 100%
            candidates.append((result["coarse_top_k"], result["coarse_hit_in_rule_top10_count"]))

    if candidates:
        # 找 RuleScorer Top10 命中率最高的 K
        best_candidate = max(candidates, key=lambda x: x[1])
        if best_candidate[1] > 4:  # 超过 50%（8篇论文中超过4篇）
            suggestion = f"\n## 建议\n\n**最佳 K 值: {best_candidate[0]}**\n\n在粗排命中率保持 100% 的前提下，RuleScorer Top10 命中率最高 ({best_candidate[1]}/8 = {best_candidate[1]*100/8:.1f}%)。\n\n"
            if best_candidate[0] != 50:
                suggestion += f"将 `pipeline.py` 中的 `top_k=50` 改为 `top_k={best_candidate[0]}`。\n"
            else:
                suggestion += "保持当前 `top_k=50` 不变。\n"
        else:
            suggestion = "\n## 建议\n\n**保持 `top_k=50` 不变。**\n\nRuleScorer Top10 命中率最高也只有 {}/8 = {:.1f}%，说明问题不在粗排数量，而在 RuleScorer 本身的排序能力。\n".format(
                best_candidate[1], best_candidate[1]*100/8
            )
    else:
        suggestion = "\n## 建议\n\n粗排命中率未达到 100%，需要增加粗排数量或优化召回策略。\n"

    return table, suggestion, best_k


def main():
    parser = argparse.ArgumentParser(description="粗排候选数量实验")
    parser.add_argument("--input", "-i", default="data/evaluation/papers_metadata.jsonl",
                        help="论文元数据路径")
    parser.add_argument("--papers", "-n", type=int, default=8,
                        help="评估论文数量（默认8）")
    parser.add_argument("--mode", "-m", choices=["title", "abstract", "full"],
                        default="abstract", help="推荐模式")
    parser.add_argument("--coarse-k", "-k", type=int, nargs="+",
                        default=[20, 30, 40, 50], help="粗排候选数量")
    parser.add_argument("--final-top-k", "-f", type=int, default=5,
                        help="最终推荐数量（默认5）")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="并行线程数（默认4）")
    parser.add_argument("--no-analysis", action="store_true",
                        help="不打印详细分析")

    args = parser.parse_args()

    # 加载论文
    print(f"加载论文数据: {args.input}")
    papers = load_papers_metadata(args.input)[:args.papers]
    print(f"共 {len(papers)} 篇论文")

    # 初始化
    print("\n初始化推荐系统...")
    pipeline, prompts = init_components()

    # 运行实验
    all_results = []
    for coarse_k in args.coarse_k:
        print(f"\n{'='*60}")
        print(f"测试 coarse_top_k = {coarse_k}")
        print(f"{'='*60}")

        result = run_experiment(
            papers, pipeline, prompts,
            mode=args.mode,
            coarse_top_k=coarse_k,
            final_top_k=args.final_top_k,
            workers=args.workers
        )
        all_results.append(result)

        print(f"  粗排命中率: {result['coarse_hit']}")
        print(f"  RuleScorer Top10 命中率: {result['coarse_hit_in_rule_top10']}")
        print(f"  最终 Hit@5: {result['hit_5']}")

    # 生成表格
    table, suggestion, best_k = generate_markdown_table(all_results)

    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    print(table)
    print(suggestion)

    # 详细分析
    if not args.no_analysis:
        print_detailed_analysis(all_results)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/evaluation/results/candidate_k_experiment_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "mode": args.mode,
            "final_top_k": args.final_top_k,
            "results": [
                {k: v for k, v in r.items() if k != "paper_results"}
                for r in all_results
            ],
            "best_k": best_k,
            "suggestion": suggestion,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
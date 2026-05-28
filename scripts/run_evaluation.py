#!/usr/bin/env python3
"""
期刊推荐系统评估脚本 - 交互式实验

支持模式：
- 标题模式 (title)
- 摘要模式 (abstract)
- 全文模式 (full) - 从PDF提取文本

评估指标：
- Hit@K (K=3/5/10 可选)
- Level Match Rate (论文质量与期刊等级匹配)
- 领域匹配准确率
- 分质量等级 Hit@5 (强/中/弱)
"""

import json
import sys
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 导入推荐系统组件
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
from src.utils.text import clean_text
from src.utils.file_parser import extract_layout_blocks
from src.papers.section_splitter import build_paper_ast


@dataclass
class EvaluationResult:
    """评估结果"""
    # 基本统计
    total_count: int
    mode: str  # title/abstract/full
    top_k: int

    # Hit@K
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    hit_at_10: int

    # 领域匹配
    area_match_count: int

    # 推荐期刊的专业领域命中（subject_tags 命中 research_area）
    area_subject_tag_match_count: int

    # Level Match Rate
    level_match_count: int

    # MRR 和 NDCG@5
    mrr: float = 0.0
    ndcg_at_5: float = 0.0

    # 粗排命中统计
    coarse_hit_count: int = 0  # 粗排命中（实际期刊在 top 50 候选中）
    coarse_hit_in_rule_top10_count: int = 0  # 粗排候选在 RuleScorer top10 中
    coarse_hit_in_rule_top20_count: int = 0  # 粗排候选在 RuleScorer top20 中

    # 分质量等级统计 (A/B/C/D)
    level_a_count: int = 0
    level_a_hit_at_5: int = 0
    level_b_count: int = 0
    level_b_hit_at_5: int = 0
    level_c_count: int = 0
    level_c_hit_at_5: int = 0
    level_d_count: int = 0
    level_d_hit_at_5: int = 0

    # 按领域统计
    by_area: dict = None
    by_level: dict = None

    # 每篇论文详情
    paper_results: list = None


def get_paper_quality_level(strength: float) -> str:
    """根据 paper_strength 划分质量等级"""
    if strength is None:
        return "unknown"
    if strength >= 0.65:
        return "strong"
    elif strength >= 0.50:
        return "medium"
    else:
        return "weak"


def calculate_mrr(relevant_rank: int) -> float:
    """计算倒数排名（rank 从 1 开始，未命中为 0）"""
    if relevant_rank <= 0:
        return 0.0
    return 1.0 / relevant_rank


def calculate_dcg_at_k(gains: list, k: int) -> float:
    """计算 DCG@k"""
    dcg = 0.0
    for i, gain in enumerate(gains[:k]):
        dcg += gain / (i + 1)  # 或使用 1/log2(i+2)，这里用位置折扣
    return dcg


def calculate_ndcg_at_k(recommended_gains: list, ideal_gains: list, k: int) -> float:
    """计算 NDCG@k（recommended_gains 和 ideal_gains 都是相关性分数列表）"""
    dcg = calculate_dcg_at_k(recommended_gains, k)
    idcg = calculate_dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_metrics(result: EvaluationResult) -> dict:
    """计算各项指标"""
    total = result.total_count if result.total_count > 0 else 1

    metrics = {
        "Hit@1": f"{result.hit_at_1}/{total} ({result.hit_at_1*100/total:.1f}%)",
        "Hit@3": f"{result.hit_at_3}/{total} ({result.hit_at_3*100/total:.1f}%)",
        "Hit@5": f"{result.hit_at_5}/{total} ({result.hit_at_5*100/total:.1f}%)",
        "Hit@10": f"{result.hit_at_10}/{total} ({result.hit_at_10*100/total:.1f}%)",
        "Area Match Rate": f"{result.area_match_count}/{total} ({result.area_match_count*100/total:.1f}%)",
        "Level Match Rate": f"{result.level_match_count}/{total} ({result.level_match_count*100/total:.1f}%)",
        "MRR": f"{result.mrr/total:.4f}" if total > 0 else "N/A",
        "NDCG@5": f"{result.ndcg_at_5/total:.4f}" if total > 0 else "N/A",
    }

    # 分质量等级
    if result.level_a_count > 0:
        metrics["A级 Hit@5"] = f"{result.level_a_hit_at_5}/{result.level_a_count} ({result.level_a_hit_at_5*100/result.level_a_count:.1f}%)"
    if result.level_b_count > 0:
        metrics["B级 Hit@5"] = f"{result.level_b_hit_at_5}/{result.level_b_count} ({result.level_b_hit_at_5*100/result.level_b_count:.1f}%)"
    if result.level_c_count > 0:
        metrics["C级 Hit@5"] = f"{result.level_c_hit_at_5}/{result.level_c_count} ({result.level_c_hit_at_5*100/result.level_c_count:.1f}%)"
    if result.level_d_count > 0:
        metrics["D级 Hit@5"] = f"{result.level_d_hit_at_5}/{result.level_d_count} ({result.level_d_hit_at_5*100/result.level_d_count:.1f}%)"

    return metrics


def load_papers_metadata(path: str) -> list:
    """加载论文元数据"""
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def init_pipeline() -> RecommenderPipeline:
    """初始化推荐 pipeline"""
    import yaml
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # 加载配置
    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f)

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 初始化存储
    store = JournalStore()
    store.load()

    # 向量搜索器
    faiss_path = app_config["data"]["faiss_index_path"]
    metadata_path = app_config["data"]["metadata_path"]
    faiss_idx = FaissIndex(faiss_path, metadata_path)
    faiss_idx.load()
    if faiss_idx.is_loaded:
        vector_searcher = VectorSearcher(faiss_idx)
        store.set_vector_searcher(vector_searcher)

    bm25 = BM25Retriever(store)
    bm25.build_index()

    # LLM
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

    retrieval_config = app_config.get("candidate_generator", {})
    merge_weights = retrieval_config.get("merge_weights", {"bm25": 0.45, "vector": 0.35, "text": 0.20})

    generator = CandidateGenerator(store, bm25, embedding_retriever, merge_weights=merge_weights)
    # 预建 RuleScorer 的 BM25 索引（传入期刊列表）
    scorer = RuleScorer(journals=store.journals)
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

    return pipeline


def evaluate_single_paper(
    paper: dict,
    pipeline: RecommenderPipeline,
    prompts: dict,
    mode: str,
    top_k: int,
) -> dict:
    """
    并行评估单篇论文，返回结果字典（非 EvaluationResult，避免锁竞争）
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    venue = paper.get("venue", "")
    ccf_level = paper.get("ccf_level", "C")
    research_area = paper.get("research_area", [""])[0] if paper.get("research_area") else ""
    arxiv_id = paper.get("external_ids", {}).get("arXiv", "")
    pdf_path = paper.get("pdf_path", "")

    # 获取全文（如果需要）
    full_text = ""
    if mode == "full" and pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            blocks, _ = extract_layout_blocks(pdf_content, pdf_path)
            paper_ast = build_paper_ast(blocks, title=title)
            full_text = paper_ast.to_markdown()
        except Exception as e:
            print(f"\nPDF读取失败: {arxiv_id} - {e}")

    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text=full_text,
        mode=mode,
    )

    # 解析论文
    try:
        profile = pipeline.parser.parse(
            paper_input,
            prompts["paper_profile_system"],
            prompts["paper_profile_user"],
        )
    except Exception as e:
        print(f"\n解析失败: {title[:30]}... - {e}")
        return None

    # 推荐
    try:
        rec_result = pipeline.recommend(
            paper_input, profile,
            top_k=top_k,
            mode=mode,
            quality_prompts={
                "system": prompts.get("paper_quality_assessor_system", ""),
                "user": prompts.get("paper_quality_assessor_user", ""),
            },
        )
    except Exception as e:
        print(f"\n推荐失败: {title[:30]}... - {e}")
        return None

    def _normalize_venue(venue: str) -> str:
        """标准化期刊名用于比较（去除首尾空格，转小写）"""
        return venue.strip().lower() if venue else ""

    recommendations = rec_result.get("recommendations", [])
    candidates = rec_result.get("candidates", [])
    rule_ranked = rec_result.get("rule_ranked", [])
    recommended_journals = [rec.journal.journal_name for rec in recommendations]
    candidate_journal_names = [j.journal_name for j in candidates] if candidates else []
    rule_ranked_names = [j.journal_name for j, s, r in rule_ranked] if rule_ranked else []

    # 标准化 venue 用于比较
    venue_normalized = _normalize_venue(venue)
    candidate_journal_names_norm = [j.lower() for j in candidate_journal_names]
    rule_ranked_names_norm = [j.lower() for j in rule_ranked_names]
    recommended_journals_norm = [j.lower() for j in recommended_journals]

    # 计算 Hit@K
    hit_1 = venue_normalized in recommended_journals_norm[:1] if len(recommended_journals) >= 1 else False
    hit_3 = venue_normalized in recommended_journals_norm[:3] if len(recommended_journals) >= 3 else False
    hit_5 = venue_normalized in recommended_journals_norm[:5] if len(recommended_journals) >= 5 else False
    hit_10 = venue_normalized in recommended_journals_norm[:10] if len(recommended_journals) >= 10 else venue_normalized in recommended_journals_norm

    # 粗排是否命中（实际发表的期刊在 top 50 候选中）
    coarse_hit = venue_normalized in candidate_journal_names_norm if venue else False

    # 粗排候选中是否包含实际发表的期刊（用于分析 RuleScorer 的选择）
    coarse_hit_in_rule_top10 = venue_normalized in rule_ranked_names_norm[:10] if venue else False
    coarse_hit_in_rule_top20 = venue_normalized in rule_ranked_names_norm[:20] if venue else False

    q_level = profile.quality_level or "D"

    # 计算 relevant_rank（1-indexed，未命中为 0）
    relevant_rank = 0
    for i, jname in enumerate(recommended_journals_norm):
        if jname == venue_normalized:
            relevant_rank = i + 1
            break

    return {
        "arxiv": arxiv_id,
        "title": title[:50],
        "venue": venue,
        "ccf_level": ccf_level,
        "research_area": research_area,
        "recommended_journals": recommended_journals[:top_k],
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_5": hit_5,
        "hit_10": hit_10,
        "relevant_rank": relevant_rank,  # MRR 计算用
        "ndcg_gain": 1.0 if relevant_rank > 0 else 0.0,  # NDCG 计算用（二值相关性）
        "coarse_hit": coarse_hit,  # 粗排命中
        "coarse_hit_in_rule_top10": coarse_hit_in_rule_top10,  # 粗排候选在 RuleScorer top10 中
        "coarse_hit_in_rule_top20": coarse_hit_in_rule_top20,  # 粗排候选在 RuleScorer top20 中
        "paper_strength": profile.paper_strength,
        "quality_level": q_level,
        "ccf_research_area": profile.ccf_research_area,
        "area_match": bool(
            profile.ccf_research_area and research_area in profile.ccf_research_area
        ) if research_area else False,
        "level_match": q_level == ccf_level,
        "area_subject_tag_match": any(
            research_area and any(
                tag in rec.journal.subject_tags
                for tag in (research_area if isinstance(research_area, list) else [research_area])
            )
            for rec in recommendations if hasattr(rec, 'journal')
        ) if research_area and recommendations else False,
    }


def run_evaluation(
    papers: list,
    pipeline: RecommenderPipeline,
    mode: str,
    top_k: int,
    prompts: dict,
    show_progress: bool = True,
    workers: int = 4,
) -> EvaluationResult:
    """运行评估（并行）"""

    result = EvaluationResult(
        total_count=len(papers),
        mode=mode,
        top_k=top_k,
        hit_at_1=0, hit_at_3=0, hit_at_5=0, hit_at_10=0,
        area_match_count=0,
        area_subject_tag_match_count=0,
        level_match_count=0,
        coarse_hit_count=0,
        coarse_hit_in_rule_top10_count=0,
        coarse_hit_in_rule_top20_count=0,
        level_a_count=0, level_a_hit_at_5=0,
        level_b_count=0, level_b_hit_at_5=0,
        level_c_count=0, level_c_hit_at_5=0,
        level_d_count=0, level_d_hit_at_5=0,
        by_area=defaultdict(lambda: {"total": 0, "hit": 0, "area_match": 0}),
        by_level=defaultdict(lambda: {"total": 0, "hit": 0}),
        paper_results=[],
    )

    results_lock = threading.Lock()

    def update_result(paper_result: dict):
        if paper_result is None:
            return
        with results_lock:
            # 累加 Hit@K
            if paper_result["hit_1"]: result.hit_at_1 += 1
            if paper_result["hit_3"]: result.hit_at_3 += 1
            if paper_result["hit_5"]: result.hit_at_5 += 1
            if paper_result["hit_10"]: result.hit_at_10 += 1

            # 领域匹配（LLM预测的领域 vs 论文实际领域）
            if paper_result["area_match"]:
                result.area_match_count += 1
                area = paper_result["research_area"]
                result.by_area[area]["area_match"] += 1

            # 推荐期刊的 subject_tags 命中论文 research_area
            if paper_result.get("area_subject_tag_match"):
                result.area_subject_tag_match_count += 1

            # Level Match
            if paper_result["level_match"]:
                result.level_match_count += 1

            # MRR 和 NDCG@5 累加
            rank = paper_result.get("relevant_rank", 0)
            if rank > 0:
                import math
                result.mrr += 1.0 / rank
                result.ndcg_at_5 += 1.0 / math.log2(rank + 1)  # 二值相关性: DCG=1/log2(r+1), IDCG=1/log2(2)=1

            # 粗排命中统计
            if paper_result.get("coarse_hit"):
                result.coarse_hit_count += 1
            if paper_result.get("coarse_hit_in_rule_top10"):
                result.coarse_hit_in_rule_top10_count += 1
            if paper_result.get("coarse_hit_in_rule_top20"):
                result.coarse_hit_in_rule_top20_count += 1

            # 分质量等级统计
            q_level = paper_result["quality_level"]
            if q_level == "A":
                result.level_a_count += 1
                if paper_result["hit_5"]: result.level_a_hit_at_5 += 1
            elif q_level == "B":
                result.level_b_count += 1
                if paper_result["hit_5"]: result.level_b_hit_at_5 += 1
            elif q_level == "C":
                result.level_c_count += 1
                if paper_result["hit_5"]: result.level_c_hit_at_5 += 1
            else:
                result.level_d_count += 1
                if paper_result["hit_5"]: result.level_d_hit_at_5 += 1

            # 按领域统计
            area = paper_result["research_area"]
            result.by_area[area]["total"] += 1
            if paper_result["hit_5"]: result.by_area[area]["hit"] += 1

            # 按CCF等级统计
            ccf = paper_result["ccf_level"]
            result.by_level[ccf]["total"] += 1
            if paper_result["hit_5"]: result.by_level[ccf]["hit"] += 1

            # 保存单篇结果
            result.paper_results.append({
                k: v for k, v in paper_result.items()
                if k not in ["hit_1", "hit_3", "hit_5", "hit_10", "area_match", "level_match", "relevant_rank", "ndcg_gain"]
            })

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_paper,
                paper, pipeline, prompts, mode, top_k,
            ): paper
            for paper in papers
        }

        pbar = tqdm(
            as_completed(futures),
            total=len(papers),
            desc=f"评估 [{mode}/top{top_k}/w{workers}]",
            unit="篇",
        )

        for future in pbar:
            paper_result = future.result()
            update_result(paper_result)

            # 更新进度条
            n = len(result.paper_results)
            if n > 0:
                hit_k = getattr(result, f"hit_at_{top_k}", 0)
                mrr_val = result.mrr / n
                ndcg_val = result.ndcg_at_5 / n
                pbar.set_postfix({
                    f"Hit@{top_k}": f"{hit_k}/{n}({hit_k*100/n:.1f}%)",
                    "MRR": f"{mrr_val:.3f}",
                    "NDCG@5": f"{ndcg_val:.3f}",
                    "Level": f"{result.level_match_count}/{n}({result.level_match_count*100/n:.1f}%)",
                    "Area": f"{result.area_match_count}/{n}({result.area_match_count*100/n:.1f}%)",
                })

    return result


def print_report(result: EvaluationResult):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print(f"评估报告 - 模式: {result.mode} | Top-K: {result.top_k}")
    print("=" * 70)

    metrics = calculate_metrics(result)

    print(f"\n总论文数: {result.total_count}")

    print(f"\n--- Hit@K ---")
    print(f"  Hit@1:  {metrics['Hit@1']}")
    print(f"  Hit@3:  {metrics['Hit@3']}")
    print(f"  Hit@5:  {metrics['Hit@5']}")
    print(f"  Hit@10: {metrics['Hit@10']}")

    print(f"\n--- 其他指标 ---")
    total = result.total_count if result.total_count > 0 else 1
    print(f"  领域匹配准确率: {metrics['Area Match Rate']}")
    print(f"  推荐期刊subject_tags命中: {result.area_subject_tag_match_count}/{total} ({result.area_subject_tag_match_count*100/total:.1f}%)")
    print(f"  Level Match Rate: {metrics['Level Match Rate']}")
    print(f"  MRR: {metrics['MRR']}")
    print(f"  NDCG@5: {metrics['NDCG@5']}")

    # 粗排命中分析
    print(f"\n--- 粗排命中分析 ---")
    print(f"  粗排命中（top50候选包含实际期刊）: {result.coarse_hit_count}/{total} ({result.coarse_hit_count*100/total:.1f}%)")
    print(f"  粗排命中且在RuleScorer top10中: {result.coarse_hit_in_rule_top10_count}/{result.coarse_hit_count} ({result.coarse_hit_in_rule_top10_count*100/max(result.coarse_hit_count,1):.1f}%)")
    print(f"  粗排命中且在RuleScorer top20中: {result.coarse_hit_in_rule_top20_count}/{result.coarse_hit_count} ({result.coarse_hit_in_rule_top20_count*100/max(result.coarse_hit_count,1):.1f}%)")

    print(f"\n--- 分质量等级 Hit@5 ---")
    if result.level_a_count > 0:
        print(f"  A级 (n={result.level_a_count}): {metrics.get('A级 Hit@5', 'N/A')}")
    if result.level_b_count > 0:
        print(f"  B级 (n={result.level_b_count}): {metrics.get('B级 Hit@5', 'N/A')}")
    if result.level_c_count > 0:
        print(f"  C级 (n={result.level_c_count}): {metrics.get('C级 Hit@5', 'N/A')}")
    if result.level_d_count > 0:
        print(f"  D级 (n={result.level_d_count}): {metrics.get('D级 Hit@5', 'N/A')}")

    print(f"\n--- 按领域分布 ---")
    for area, stats in sorted(result.by_area.items()):
        hit_rate = stats["hit"] * 100 / stats["total"] if stats["total"] > 0 else 0
        print(f"  {area}: {stats['hit']}/{stats['total']} ({hit_rate:.1f}%)")

    print(f"\n--- 按CCF等级分布 ---")
    for level in ['A', 'B', 'C']:
        if level in result.by_level:
            stats = result.by_level[level]
            hit_rate = stats["hit"] * 100 / stats["total"] if stats["total"] > 0 else 0
            print(f"  {level}级: {stats['hit']}/{stats['total']} ({hit_rate:.1f}%)")

    print("=" * 70)


def save_results(result: EvaluationResult, output_dir: str = "data/evaluation/results"):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{result.mode}_top{result.top_k}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # 转换为可序列化的dict
    result_dict = {
        "timestamp": timestamp,
        "mode": result.mode,
        "top_k": result.top_k,
        "total_count": result.total_count,
        "metrics": {
            "hit_at_1": result.hit_at_1,
            "hit_at_3": result.hit_at_3,
            "hit_at_5": result.hit_at_5,
            "hit_at_10": result.hit_at_10,
            "area_match_count": result.area_match_count,
            "level_match_count": result.level_match_count,
            "mrr": result.mrr / result.total_count if result.total_count > 0 else 0.0,
            "ndcg_at_5": result.ndcg_at_5 / result.total_count if result.total_count > 0 else 0.0,
            "coarse_hit_count": result.coarse_hit_count,
            "coarse_hit_in_rule_top10_count": result.coarse_hit_in_rule_top10_count,
            "coarse_hit_in_rule_top20_count": result.coarse_hit_in_rule_top20_count,
            "area_subject_tag_match_count": result.area_subject_tag_match_count,
            "level_a_count": result.level_a_count,
            "level_a_hit_at_5": result.level_a_hit_at_5,
            "level_b_count": result.level_b_count,
            "level_b_hit_at_5": result.level_b_hit_at_5,
            "level_c_count": result.level_c_count,
            "level_c_hit_at_5": result.level_c_hit_at_5,
            "level_d_count": result.level_d_count,
            "level_d_hit_at_5": result.level_d_hit_at_5,
        },
        "by_area": dict(result.by_area),
        "by_level": dict(result.by_level),
        "paper_results": result.paper_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {filepath}")
    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description="期刊推荐系统评估")
    parser.add_argument("--input", "-i", default="data/evaluation/papers_metadata.jsonl",
                        help="论文元数据路径")
    parser.add_argument("--mode", "-m", choices=["title", "abstract", "full"], default="full",
                        help="推荐模式")
    parser.add_argument("--top-k", "-k", type=int, nargs="+", default=[5],
                        help="Top-K 值，可指定多个如: --top-k 3 5 10")
    parser.add_argument("--papers", "-n", type=int, default=None,
                        help="限制评估论文数量（用于测试）")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存结果")
    parser.add_argument("--workers", "-w", type=int, default=10,
                        help="并行线程数（默认4）")

    args = parser.parse_args()

    # 加载论文
    print(f"加载论文数据: {args.input}")
    papers = load_papers_metadata(args.input)
    if args.papers:
        papers = papers[:args.papers]
    print(f"共 {len(papers)} 篇论文")

    # 初始化 pipeline
    print("\n初始化推荐系统...")
    pipeline = init_pipeline()

    # 加载 prompts
    import yaml
    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 运行评估
    for top_k in args.top_k:
        print(f"\n{'='*70}")
        print(f"开始评估 | 模式: {args.mode} | Top-{top_k}")
        print(f"{'='*70}")

        result = run_evaluation(papers, pipeline, args.mode, top_k, prompts, show_progress=True, workers=args.workers)

        # 打印报告
        print_report(result)

        # 保存结果
        if not args.no_save:
            save_results(result)

    print("\n评估完成!")


if __name__ == "__main__":
    main()
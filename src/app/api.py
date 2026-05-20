"""API 路由"""
from typing import Optional, Union
import json
import yaml
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .schemas import (
    RecommendResponse,
    JournalListResponse,
    JournalListItem,
    HealthResponse,
    JournalResponse,
    PaperProfileResponse,
)
from ..recommender.pipeline import RecommenderPipeline
from ..papers.paper_parser import PaperParser
from ..papers.paper_model import PaperInput, PaperProfile
from ..journals.journal_store import JournalStore
from ..journals.vector_searcher import VectorSearcher, FaissIndex
from ..retriever.bm25_retriever import BM25Retriever
from ..retriever.embedding_retriever import EmbeddingRetriever
from ..retriever.candidate_generator import CandidateGenerator
from ..ranker.rule_scorer import RuleScorer
from ..ranker.llm_ranker import LLMRanker
from ..recommender.explainer import Explainer
from ..papers.quality_assessor import PaperQualityAssessor
from ..utils.llm import MiniMaxLLM
from ..utils.embedding import OllamaEmbedding
from ..utils.file_parser import extract_text_from_file


class RecommendRequest(BaseModel):
    """推荐请求体（支持 JSON）"""
    title: str
    abstract: str = ""
    full_text: str = ""
    mode: str = "abstract"
    top_k: int = 5
    oa_preference: str = "any"


router = APIRouter()

# 全局组件（实际应通过依赖注入）
_pipeline: Optional[RecommenderPipeline] = None
_store: Optional[JournalStore] = None


def get_pipeline() -> RecommenderPipeline:
    """获取 pipeline（懒加载初始化）"""
    global _pipeline, _store

    # 确保 .env 被加载
    from dotenv import load_dotenv
    load_dotenv(override=True)

    if _pipeline is None:
        # 加载配置
        with open("configs/app.yaml", "r", encoding="utf-8") as f:
            app_config = yaml.safe_load(f)

        with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        # 初始化存储
        _store = JournalStore()
        _store.load()

        # 初始化向量搜索器（如果 FAISS 索引存在）
        faiss_path = app_config["data"]["faiss_index_path"]
        metadata_path = app_config["data"]["metadata_path"]
        faiss_idx = FaissIndex(faiss_path, metadata_path)
        faiss_idx.load()
        if faiss_idx.is_loaded:
            vector_searcher = VectorSearcher(faiss_idx)
            _store.set_vector_searcher(vector_searcher)

        bm25 = BM25Retriever(_store)
        bm25.build_index()

        # 初始化 LLM（如果配置了 API key）
        llm = None
        try:
            import os
            api_key = app_config["minimax"]["api_key"]
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var)
            if api_key:
                llm = MiniMaxLLM(
                    api_key=api_key,
                    base_url=app_config["minimax"]["base_url"],
                    model=app_config["minimax"]["model"],
                )
        except Exception:
            pass

        embedding_client = OllamaEmbedding(
            base_url=app_config["ollama"]["base_url"],
            model=app_config["ollama"]["embedding_model"],
        )

        # 只有向量搜索可用时才创建 embedding retriever
        embedding_retriever = None
        if _store.has_vector_search():
            embedding_retriever = EmbeddingRetriever(_store, embedding_client)

        # 读取召回权重配置
        retrieval_config = app_config.get("retrieval", {})
        merge_weights = retrieval_config.get("merge_weights", {"bm25": 0.4, "vector": 0.4, "tag": 0.2})

        generator = CandidateGenerator(_store, bm25, embedding_retriever, merge_weights=merge_weights)
        scorer = RuleScorer()

        llm_ranker = None
        if llm:
            llm_ranker = LLMRanker(
                llm,
                prompts["llm_ranker_system"],
                prompts["llm_ranker_user"],
            )

        explainer = Explainer(llm, prompts["explainer_system"], prompts["explainer_user"])

        quality_assessor = None
        if llm:
            quality_assessor = PaperQualityAssessor(llm)

        _pipeline = RecommenderPipeline(
            candidate_generator=generator,
            rule_scorer=scorer,
            llm_ranker=llm_ranker,
            explainer=explainer,
            quality_assessor=quality_assessor,
        )

    return _pipeline


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: Request):
    """推荐期刊（同时支持 JSON 和 Form data）"""
    pipeline = get_pipeline()

    # 解析请求（支持 JSON 和 Form）
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        # JSON 格式
        body = await request.body()
        import json
        data = json.loads(body)
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        full_text = data.get("full_text", "")
        mode = data.get("mode", "abstract")
        top_k = data.get("top_k", 5)
        oa_preference = data.get("oa_preference", "any")
        file = None
    else:
        # Form data（从请求表单中获取）
        form = await request.form()
        title = form.get("title", "")
        abstract = form.get("abstract", "")
        full_text = ""
        mode = form.get("mode", "abstract")
        top_k = int(form.get("top_k", 5))
        oa_preference = form.get("oa_preference", "any")
        file = form.get("file")  # UploadFile 对象

    # 处理文件上传
    full_text_content = ""
    if file and mode == "full":
        content = await file.read()
        full_text_content = extract_text_from_file(content, file.filename)

    # 解析论文
    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text=full_text_content or full_text or "",
        mode=mode,
    )

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    parser = PaperParser()
    profile = parser.parse(paper_input, prompts["paper_profile_system"], prompts["paper_profile_user"])

    # 执行推荐
    quality_prompts = {
        "system": prompts.get("paper_quality_assessor_system", ""),
        "user": prompts.get("paper_quality_assessor_user", ""),
    }
    result = pipeline.recommend(
        paper_input,
        profile,
        top_k=top_k,
        mode=mode,
        oa_preference=oa_preference,
        quality_prompts=quality_prompts,
    )

    # 获取排序方法
    rank_method = result.get("rank_method", "rule")

    # 构建响应
    recommendations = []
    for rec in result.get("recommendations", []):
        recommendations.append(JournalResponse(
            journal_id=rec.journal.journal_id,
            journal_name=rec.journal.journal_name,
            score=rec.score,
            confidence=rec.confidence,
            match_reasons=rec.match_reasons,
            matched_fields=rec.matched_fields,
            tags=rec.journal.subject_tags,
            oa_type=rec.journal.oa_type,
            quartile=rec.journal.quartile,
            submission_url=rec.journal.submission_url,
            rank_method=rank_method,
        ))

    paper_profile_resp = None
    if "paper_profile" in result:
        pp = result["paper_profile"]
        paper_profile_resp = PaperProfileResponse(
            title=pp.title,
            research_area=pp.research_area,
            method_type=pp.method_type,
            paper_type=pp.paper_type,
            keywords=pp.keywords,
        )

    return RecommendResponse(
        recommendations=recommendations,
        paper_profile=paper_profile_resp,
        mode_used=result.get("mode_used", mode),
        rank_method=result.get("rank_method", "rule"),
        warning=result.get("warning"),
    )


def _apply_quality_adjustment(
    ranked: list,
    paper_profile,
) -> list:
    """应用质量软权重调整（解耦）"""
    if paper_profile.paper_strength is None:
        return ranked

    strength = paper_profile.paper_strength
    base_adjustment = 0.9 + 0.2 * (strength - 0.5)

    adjusted = []
    for journal, score, reasons in ranked:
        ccf_multiplier = {"A": 1.05, "B": 1.02, "C": 1.0}.get(journal.ccf_rating, 1.0)
        adjustment = max(0.8, min(1.08, base_adjustment * ccf_multiplier))
        new_reasons = reasons.copy()
        if strength >= 0.75:
            new_reasons.append(f"强论文调整(+{(adjustment-1)*100:.0f}%)")
        elif strength < 0.35:
            new_reasons.append(f"弱论文调整({(adjustment-1)*100:.0f}%)")
        adjusted.append((journal, score * adjustment, new_reasons))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


@router.get("/recommend/stream")
async def recommend_stream(
    title: str = Query(...),
    abstract: str = Query(""),
    mode: str = Query("abstract"),
    top_k: int = Query(5, ge=1, le=20),
    oa_preference: str = Query("any"),
):
    """流式推荐期刊（SSE）"""
    pipeline = get_pipeline()

    async def event_generator():
        import asyncio
        def sse_event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        try:
            # 阶段 1: 解析论文 (0-20%)
            yield sse_event("progress", {
                "stage": "parsing",
                "percent": 10,
                "message": "正在解析论文特征..."
            })
            await asyncio.sleep(0)  # 让出控制权，确保事件立即发送

            paper_input = PaperInput(
                title=title,
                abstract=abstract or "",
                full_text="",
                mode=mode,
            )

            with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f)

            parser = PaperParser()
            profile = parser.parse(paper_input, prompts["paper_profile_system"], prompts["paper_profile_user"])

            yield sse_event("progress", {
                "stage": "parsing",
                "percent": 20,
                "message": "论文解析完成"
            })
            await asyncio.sleep(0)

            # 阶段 1.5: 评估论文质量 (20-25%)
            yield sse_event("progress", {
                "stage": "quality",
                "percent": 22,
                "message": "正在评估论文质量..."
            })
            await asyncio.sleep(0)

            quality_prompts = {
                "system": prompts.get("paper_quality_assessor_system", ""),
                "user": prompts.get("paper_quality_assessor_user", ""),
            }
            quality = pipeline.quality_assessor.assess(
                paper_input, profile,
                quality_prompts.get("system", ""),
                quality_prompts.get("user", ""),
            ) if pipeline.quality_assessor else None

            if quality:
                profile.paper_strength = quality.paper_strength
                profile.readiness = quality.readiness
                profile.quality_level = quality.quality_level
                profile.quality_confidence = quality.confidence
                profile.quality_reasons = quality.reasons
                yield sse_event("progress", {
                    "stage": "quality",
                    "percent": 25,
                    "message": f"论文质量评估完成: {quality.quality_level} (strength={quality.paper_strength:.2f})"
                })
                await asyncio.sleep(0)

            # 阶段 2: 候选召回 (25-50%)
            yield sse_event("progress", {
                "stage": "retrieval",
                "percent": 25,
                "message": "正在召回候选期刊..."
            })
            await asyncio.sleep(0)

            query_text = title
            if abstract:
                query_text += " " + abstract

            candidates = pipeline.candidate_generator.generate(
                query_text, profile, top_k=50, mode=mode
            )

            if not candidates:
                yield sse_event("error", {"message": "未找到合适的候选期刊"})
                return

            yield sse_event("progress", {
                "stage": "retrieval",
                "percent": 50,
                "message": f"候选召回完成，找到 {len(candidates)} 个候选期刊"
            })
            await asyncio.sleep(0)

            # 阶段 3: 规则排序 (50-60%)
            yield sse_event("progress", {
                "stage": "ranking",
                "percent": 55,
                "message": "正在进行规则排序..."
            })
            await asyncio.sleep(0)

            rule_ranked = pipeline.rule_scorer.rank(
                candidates, profile, oa_preference=oa_preference, top_k=10
            )

            # 应用质量软权重调整（解耦）
            rule_ranked = _apply_quality_adjustment(rule_ranked, profile)

            # 阶段 4: LLM 精排 (60-80%)
            rank_method = "rule"
            llm_ranked = rule_ranked

            if pipeline.llm_ranker:
                yield sse_event("progress", {
                    "stage": "ranking",
                    "percent": 65,
                    "message": "正在进行 AI 智能排序..."
                })
                await asyncio.sleep(0)
                llm_ranked, rank_method = pipeline.llm_ranker.rank(rule_ranked, profile, top_k=top_k)
            else:
                # 无 LLM ranker 时，使用规则排序结果（补充 confidence）
                llm_ranked = [(j, s, r, 0.5) for j, s, r in rule_ranked[:top_k]]

            yield sse_event("progress", {
                "stage": "ranking",
                "percent": 80,
                "message": "排序完成"
            })
            await asyncio.sleep(0)

            # 阶段 5: 生成推荐理由 (80-95%)
            yield sse_event("progress", {
                "stage": "explaining",
                "percent": 85,
                "message": "正在生成推荐理由..."
            })
            await asyncio.sleep(0)

            recommendations = []
            for idx, (journal, score, reasons, confidence) in enumerate(llm_ranked):
                match_reasons = reasons
                if pipeline.explainer:
                    match_reasons = pipeline.explainer.explain(journal, profile)

                rec = {
                    "journal_id": journal.journal_id,
                    "journal_name": journal.journal_name,
                    "score": score,
                    "confidence": confidence,
                    "match_reasons": match_reasons,
                    "matched_fields": ["research_area", "method_type"],
                    "tags": journal.subject_tags,
                    "oa_type": journal.oa_type,
                    "quartile": journal.quartile,
                    "submission_url": journal.submission_url or "",
                    "rank_method": rank_method,
                }
                recommendations.append(rec)

                # 推送每个推荐结果
                progress_percent = 85 + (idx * 10 // max(len(llm_ranked), 1))
                yield sse_event("recommendation", rec)
                yield sse_event("progress", {
                    "stage": "explaining",
                    "percent": progress_percent,
                    "message": f"已生成 {idx + 1}/{len(llm_ranked)} 条推荐理由"
                })
                await asyncio.sleep(0)

            # 完成
            done_data = {
                "total": len(recommendations),
                "rank_method": rank_method,
                "mode_used": mode,
            }
            if profile.quality_level:
                done_data["quality"] = {
                    "level": profile.quality_level,
                    "paper_strength": profile.paper_strength,
                    "readiness": profile.readiness,
                    "confidence": profile.quality_confidence,
                    "reasons": profile.quality_reasons,
                }
            yield sse_event("done", done_data)

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/journals", response_model=JournalListResponse)
async def list_journals(
    subject_tag: Optional[str] = Query(None),
    oa_type: Optional[str] = Query(None),
    quartile: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """列出期刊"""
    store = get_pipeline().candidate_generator.store

    journals = store.list_journals(
        subject_tag=subject_tag,
        oa_type=oa_type,
        quartile=quartile,
        limit=limit,
        offset=offset,
    )

    return JournalListResponse(
        journals=[
            JournalListItem(
                journal_id=j.journal_id,
                journal_name=j.journal_name,
                subject_tags=j.subject_tags,
                oa_type=j.oa_type,
                quartile=j.quartile,
            )
            for j in journals
        ],
        total=store.count,
        limit=limit,
        offset=offset,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(status="ok", version="0.1.0")
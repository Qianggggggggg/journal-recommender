"""API 路由"""
from functools import lru_cache
from typing import Any, Dict, Optional, Union
import json
import os
import threading
import yaml
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.requests import Request
from fastapi.responses import StreamingResponse, Response
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
from ..papers.paper_parser import PaperParser, PaperParserError
from ..papers.paper_model import PaperInput, PaperProfile
from ..journals.journal_store import JournalStore
from ..journals.typical_abstract_store import TypicalAbstractStore
from ..journals.vector_searcher import VectorSearcher, FaissIndex
from ..retriever.bm25_retriever import BM25Retriever
from ..retriever.embedding_retriever import EmbeddingRetriever
from ..retriever.candidate_generator import CandidateGenerator
from ..retriever.typical_abstract_retriever import (
    TypicalAbstractBM25Retriever,
    TypicalAbstractEmbeddingRetriever,
    TypicalAbstractTextRetriever,
)
from ..ranker.rule_scorer import RuleScorer
from ..ranker.llm_ranker import LLMRanker, LLMRankerError
from ..utils.text import quality_adjustment_factor


# 6.5: in-memory evidence cache. Avoids re-extracting the same paper's
# evidence via the live LLM extractor (which is ~30-60s per call) on
# repeated recommendations of the same title. The cache is process-local
# (per FastAPI worker) and unbounded; the API layer is responsible
# for sizing/eviction if it becomes a memory concern.
_EVIDENCE_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_EVIDENCE_CACHE_LOCK = threading.Lock()


def _title_key(title: str) -> str:
    """Normalize a paper title for cache/snapshot lookup."""
    return " ".join(str(title or "").casefold().split())


def _cache_get(title: str) -> Optional[Dict[str, Dict[str, Any]]]:
    with _EVIDENCE_CACHE_LOCK:
        return _EVIDENCE_CACHE.get(_title_key(title))


def _cache_put(title: str, evidence: Dict[str, Dict[str, Any]]) -> None:
    if not evidence:
        return
    with _EVIDENCE_CACHE_LOCK:
        _EVIDENCE_CACHE[_title_key(title)] = dict(evidence)


def get_paper_evidence_from_pipeline(
    pipeline: RecommenderPipeline,
    paper_title: str,
    candidates: Any = None,
    paper_profile: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Resolve per-paper evidence via cache → role-ranker snapshot → online
    extractor. Returns a ``{journal_id: evidence_item}`` dict (possibly empty).

    - In-memory cache (process-local) is consulted first.
    - If the pipeline's role ranker has a snapshot and the title is in
      it, returns the snapshot's per-paper evidence.
    - Otherwise, if the pipeline has a live ``evidence_extractor`` and
      candidates + paper_profile are supplied, calls the extractor to
      fill the gap and caches the result for next time.
    - Returns ``{}`` if no source has evidence; the role ranker will
      then fall back to neutral defaults.
    """
    cached = _cache_get(paper_title)
    if cached is not None:
        return cached

    # Try the role ranker's snapshot (only present when the pipeline is
    # using LLMEvidenceRoleRanker with a loaded evidence_snapshot).
    snapshot = getattr(pipeline.llm_ranker, "evidence_snapshot", None)
    if snapshot:
        entry = snapshot.get(_title_key(paper_title))
        if entry and entry.get("evidence"):
            evidence = dict(entry["evidence"])
            _cache_put(paper_title, evidence)
            return evidence

    # Last resort: ask the live extractor to do an online extraction.
    # This is ~30-60s per call; only used when a paper is not in the
    # offline snapshot.
    extractor = getattr(pipeline, "evidence_extractor", None)
    if extractor is not None and candidates and paper_profile is not None:
        try:
            evidence = extractor.extract(candidates, paper_profile)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] online evidence extraction failed: {exc}")
            evidence = {}
        if evidence:
            _cache_put(paper_title, evidence)
        return evidence

    return {}
from ..papers.quality_assessor import PaperQualityAssessor, PaperQualityError
from ..utils.llm_config import build_minimax_llm
from ..utils.embedding import OllamaEmbedding
from ..utils.file_parser import extract_text_from_file, extract_layout_blocks
from ..papers.section_splitter import build_paper_ast
from ..utils.pdf_exporter import PDFExporter


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
STREAM_PROGRESS_INTERVAL_SECONDS = 1.0


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    """加载 prompts.yaml（带缓存，避免每次请求重新读取）"""
    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

        # 初始化 LLM（必须配置 API key）
        llm = build_minimax_llm(app_config)

        embedding_client = OllamaEmbedding(
            base_url=app_config["ollama"]["base_url"],
            model=app_config["ollama"]["embedding_model"],
            timeout=app_config.get("ollama", {}).get("timeout_seconds", 60),
        )

        # 只有向量搜索可用时才创建 embedding retriever
        embedding_retriever = None
        if _store.has_vector_search():
            embedding_retriever = EmbeddingRetriever(_store, embedding_client)

        generator = _build_candidate_generator(
            store=_store,
            bm25=bm25,
            embedding_retriever=embedding_retriever,
            embedding_client=embedding_client,
            app_config=app_config,
        )
        scorer = _build_rule_scorer(_store, app_config)

        # 6.3: build LLMEvidenceRoleRanker when evidence_role is enabled and
        # the snapshot file exists. Mirrors scripts/run_evaluation.py:410-478
        # so the API path uses the same evidence-based formula as evaluation.
        # Falls back to LLMRanker when evidence_role is disabled, missing, or
        # llm_direct is explicitly enabled.
        ranking_cfg = app_config.get("ranking", {}) or {}
        evidence_cfg = ranking_cfg.get("evidence_role", {}) or {}
        use_evidence = bool(evidence_cfg.get("enabled", False))
        use_direct_default = ranking_cfg.get("llm_direct", {}).get("enabled", False)
        evidence_snapshot = None
        evidence_extractor = None
        llm_ranker = None
        if use_evidence:
            from pathlib import Path as _Path
            from src.ranker.llm_evidence_role_ranker import (
                LLMEvidenceRoleRanker,
                load_evidence_snapshot,
            )
            snapshot_path = evidence_cfg.get("snapshot_path", "")
            sp = _Path(snapshot_path)
            if not sp.exists() or not snapshot_path:
                print(
                    f"[api warn] evidence_role.enabled=true but snapshot not found: "
                    f"{snapshot_path!r}; falling back to LLMRanker"
                )
            else:
                try:
                    evidence_snapshot = load_evidence_snapshot(str(sp))
                    from src.ranker.llm_evidence_extractor import LLMEvidenceExtractor, select_evidence_prompts
                    # Use evidence-specific timeout (defaults to llm_ranker_timeout)
                    # so a slow evidence call doesn't poison other LLM calls.
                    evidence_timeout = float(
                        evidence_cfg.get("evidence_extractor_timeout_seconds")
                        or ranking_cfg.get("llm_ranker_timeout_seconds", 200)
                    )
                    # P0 (2026-06-16): select v1 or v2 evidence prompt based on
                    # evidence_role.prompt_version. v2 adds CCF-tier calibration.
                    evidence_prompt_version = evidence_cfg.get("prompt_version", "v1")
                    evidence_sys_p, evidence_usr_p = select_evidence_prompts(
                        prompts, evidence_prompt_version
                    )
                    evidence_extractor = LLMEvidenceExtractor(
                        llm=llm,
                        system_prompt=evidence_sys_p,
                        user_prompt_template=evidence_usr_p,
                        timeout_seconds=evidence_timeout,
                    )
                    llm_ranker = LLMEvidenceRoleRanker(
                        evidence_extractor=evidence_extractor,
                        journal_store=_store,
                        accepted_paper_store=None,
                        prior_source=evidence_cfg.get("prior_source", "rule"),
                        evidence_weight=float(evidence_cfg.get("evidence_weight", 0.8)),
                        prior_weight=float(evidence_cfg.get("prior_weight", 0.2)),
                        ltr_score_weight=float(evidence_cfg.get("ltr_score_weight", 0.0)),
                        evidence_snapshot=evidence_snapshot,
                        evidence_field_weights=evidence_cfg.get("evidence_field_weights"),
                    )
                    print(
                        f"[api ok] evidence_role enabled, snapshot={snapshot_path}, "
                        f"prior_source={evidence_cfg.get('prior_source', 'rule')}, "
                        f"ltr_score_weight={evidence_cfg.get('ltr_score_weight', 0.0)}, "
                        f"papers={len(evidence_snapshot)}"
                    )
                except Exception as exc:
                    print(f"[api warn] failed to load evidence snapshot: {exc}; falling back to LLMRanker")
                    evidence_snapshot = None
                    llm_ranker = None

        if llm_ranker is None:
            if not use_direct_default:
                # Strict mode (no direct LLM fallback allowed): raise so we
                # don't silently downgrade to a ranker that ignores evidence.
                raise RuntimeError(
                    "ranking config has neither llm_direct.enabled=true nor a working "
                    "evidence_role snapshot; cannot build a ranker"
                )
            llm_ranker = LLMRanker(
                llm,
                prompts["llm_ranker_system"],
                prompts["llm_ranker_user"],
                timeout_seconds=app_config.get("ranking", {}).get("llm_ranker_timeout_seconds", 200),
            )

        quality_assessor = PaperQualityAssessor(llm)

        # 初始化论文解析器
        parser = PaperParser(llm)

        # 5.3: LTR adapter(默认 OFF 时 LTRAdapter.enabled=False,pipeline 走原路径)
        from src.ranker.ltr_adapter import LTRAdapter
        ltr_config = (app_config.get("ranking", {}).get("learned_reranker", {}) or {})
        learned_reranker = LTRAdapter(
            config=ltr_config,
            journal_store=_store,
            accepted_paper_store=None,  # API 路径不依赖 accepted_paper_store
        )

        _pipeline = RecommenderPipeline(
            candidate_generator=generator,
            rule_scorer=scorer,
            llm_ranker=llm_ranker,
            quality_assessor=quality_assessor,
            llm_anchor_guard=app_config.get("ranking", {}).get("llm_anchor_guard", {}),
            learned_reranker=learned_reranker,
            # 6.5: hand the role-ranker extractor to the pipeline so
            # get_paper_evidence_from_pipeline can do online extraction for
            # papers not in the offline snapshot.
            evidence_extractor=evidence_extractor,
        )

        # 将 parser 附加到 pipeline 以便在 API 中使用
        _pipeline.parser = parser

    return _pipeline


def _build_candidate_generator(
    store: JournalStore,
    bm25: BM25Retriever,
    embedding_retriever: Optional[EmbeddingRetriever],
    embedding_client: OllamaEmbedding,
    app_config: dict,
) -> CandidateGenerator:
    """Create the configured candidate generator, including semantic-anchor retrieval."""
    retrieval_config = app_config.get("candidate_generator", {})
    data_config = app_config.get("data", {})

    retrieval_target = retrieval_config.get("retrieval_target", "scope_text")
    merge_weights = retrieval_config.get("merge_weights", {"bm25": 0.45, "vector": 0.35, "text": 0.20})
    fusion_strategy = retrieval_config.get("fusion_strategy", "weighted_minmax")
    hybrid_scope_weight = retrieval_config.get("hybrid_scope_weight", 0.75)
    hybrid_typical_weight = retrieval_config.get("hybrid_typical_weight", 0.25)
    identity_anchor_weight = retrieval_config.get("identity_anchor_weight", 0.03)
    accepted_paper_weight = retrieval_config.get("accepted_paper_weight", 0.20)
    rrf_k = retrieval_config.get("rrf_k", 60)
    route_top_k = retrieval_config.get("route_top_k")

    typical_bm25 = None
    typical_embedding = None
    typical_text = None
    accepted_bm25 = None
    accepted_embedding = None

    if retrieval_target in {"typical_abstracts", "semantic_anchors"}:
        abstract_store = TypicalAbstractStore(
            data_config.get("typical_abstracts_dir", "data/typical_abstracts")
        )
        abstract_store.load()
        if abstract_store.count == 0:
            raise RuntimeError("已配置典型摘要召回，但未加载到任何典型摘要")

        typical_bm25 = TypicalAbstractBM25Retriever(abstract_store, store)
        typical_bm25.build_index()
        typical_text = TypicalAbstractTextRetriever(abstract_store, store)
        typical_embedding = TypicalAbstractEmbeddingRetriever(
            abstract_store=abstract_store,
            journal_store=store,
            embedding_client=embedding_client,
            faiss_path=data_config.get(
                "typical_abstracts_faiss_path",
                "data/processed/typical_abstracts_index.faiss",
            ),
            metadata_path=data_config.get(
                "typical_abstracts_metadata_path",
                "data/processed/typical_abstracts_metadata.parquet",
            ),
        )

        # accepted-paper 路由:语料缺失或索引未构建时自动禁用,不影响主流程
        from ..journals.accepted_paper_store import AcceptedPaperStore
        from ..retriever.accepted_paper_retriever import (
            AcceptedPaperBM25Retriever,
            AcceptedPaperEmbeddingRetriever,
        )

        accepted_store = AcceptedPaperStore(
            accepted_dir=data_config.get("accepted_papers_dir", "data/accepted_papers")
        )
        accepted_store.load()
        if accepted_store.count > 0:
            accepted_bm25 = AcceptedPaperBM25Retriever(accepted_store, store)
            accepted_bm25.build_index()
            accepted_embedding = AcceptedPaperEmbeddingRetriever(
                accepted_store=accepted_store,
                journal_store=store,
                embedding_client=embedding_client,
                faiss_path=data_config.get(
                    "accepted_papers_faiss_path",
                    "data/processed/accepted_papers_index.faiss",
                ),
                metadata_path=data_config.get(
                    "accepted_papers_metadata_path",
                    "data/processed/accepted_papers_metadata.parquet",
                ),
            )
            if not accepted_embedding.is_available:
                # 索引文件缺失时把 vector retriever 置 None,BM25 单路保留
                accepted_embedding = None

    return CandidateGenerator(
        store,
        bm25,
        embedding_retriever,
        merge_weights=merge_weights,
        retrieval_target=retrieval_target,
        typical_bm25_retriever=typical_bm25,
        typical_embedding_retriever=typical_embedding,
        typical_text_retriever=typical_text,
        accepted_bm25_retriever=accepted_bm25,
        accepted_embedding_retriever=accepted_embedding,
        hybrid_scope_weight=hybrid_scope_weight,
        hybrid_typical_weight=hybrid_typical_weight,
        identity_anchor_weight=identity_anchor_weight,
        accepted_paper_weight=accepted_paper_weight,
        fusion_strategy=fusion_strategy,
        rrf_k=rrf_k,
        route_top_k=route_top_k,
    )


def _build_rule_scorer(store: JournalStore, app_config: dict) -> RuleScorer:
    """Create RuleScorer with ranking.rule_scorer weight overrides."""
    ranking_config = app_config.get("ranking", {})
    rule_weights = ranking_config.get("rule_scorer", {})
    return RuleScorer(journals=store.journals, weights=rule_weights)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: Request):
    """推荐期刊（同时支持 JSON 和 Form data）"""
    pipeline = get_pipeline()

    # 解析请求（支持 JSON 和 Form）
    content_type = request.headers.get("content-type", "")
    #print(f"[DEBUG] content_type: '{content_type}'")
    body_preview = await request.body()
    #print(f"[DEBUG] body len: {len(body_preview)}, preview: {body_preview[:200]}")

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
        #print(f"[DEBUG] file size: {len(content)}, filename: {file.filename}")

        # 使用 PyMuPDF layout extraction
        blocks, _ = extract_layout_blocks(content, file.filename)

        # 构建 Paper AST
        paper_ast = build_paper_ast(blocks, title=title)
        markdown = paper_ast.to_markdown()
        #print(f"[DEBUG] Paper AST: {len(paper_ast.sections)} sections, markdown length: {len(markdown)}")

        # 用 markdown 替换 full_text
        full_text_content = markdown

    # 解析论文
    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text=full_text_content or full_text or "",
        mode=mode,
    )

    prompts = _load_prompts()

    try:
        profile = pipeline.parser.parse(paper_input, prompts["paper_profile_system"], prompts["paper_profile_user"])
    except PaperParserError as e:
        raise HTTPException(status_code=503, detail=f"论文解析失败: {e}")

    # 6.5: hybrid evidence path. Resolve per-paper evidence from cache →
    # role-ranker snapshot → live extractor. Doing it here (rather than
    # relying on the role ranker alone) means repeated requests for the
    # same paper title hit the cache instead of re-extracting.
    precomputed_evidence = get_paper_evidence_from_pipeline(
        pipeline,
        paper_input.title,
        candidates=None,            # not yet generated; use cache/snapshot first
        paper_profile=None,        # only used by the online fallback
    )

    # 执行推荐
    quality_prompts = {
        "system": prompts.get("paper_quality_assessor_system", ""),
        "user": prompts.get("paper_quality_assessor_user", ""),
    }
    try:
        result = pipeline.recommend(
            paper_input,
            profile,
            top_k=top_k,
            mode=mode,
            oa_preference=oa_preference,
            quality_prompts=quality_prompts,
            precomputed_evidence=precomputed_evidence or None,
        )
    except PaperQualityError as e:
        raise HTTPException(status_code=503, detail=f"论文质量评估失败: {e}")

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


@router.post("/recommend/pdf/from-results")
async def recommend_pdf_from_results(request: Request):
    """从已生成的推荐结果直接导出 PDF（不复用 pipeline）"""
    body = await request.body()
    data = json.loads(body)

    title = data.get("title", "")
    abstract = data.get("abstract", "")
    recommendations_data = data.get("recommendations", [])
    paper_profile_data = data.get("paper_profile")

    # 直接生成 PDF（不调用 pipeline）
    exporter = PDFExporter()

    # 构建 PaperProfile 对象（如果有）
    profile = None
    if paper_profile_data:
        from ..papers.paper_model import PaperProfile
        profile = PaperProfile(
            title=paper_profile_data.get("title", ""),
            research_area=paper_profile_data.get("research_area", []),
            method_type=paper_profile_data.get("method_type", ""),
            paper_type=paper_profile_data.get("paper_type", ""),
            keywords=paper_profile_data.get("keywords", []),
        )
        # 补充 quality 信息（从前端 SSE doneData.quality 传入）
        quality_data = data.get("quality", {})
        if quality_data:
            profile.quality_level = quality_data.get("level")
            profile.paper_strength = quality_data.get("paper_strength")
            profile.quality_confidence = quality_data.get("confidence")
            profile.readiness = quality_data.get("readiness")

    # 构建 JournalMatch 对象列表
    from ..journals.journal_model import Journal, JournalMatch
    matches = []
    for rec in recommendations_data:
        journal = Journal(
            journal_id=rec.get("journal_id", ""),
            journal_name=rec.get("journal_name", ""),
            publisher=rec.get("publisher", ""),
            scope_text=rec.get("scope_text", ""),
            subject_tags=rec.get("tags", []),
            keywords=[],
            oa_type=rec.get("oa_type", "subscription"),
            submission_url=rec.get("submission_url", ""),
            homepage_url=rec.get("homepage_url", ""),
            ccf_rating=rec.get("ccf_rating", ""),
            impact_like_score=rec.get("impact_like_score") or None,
            review_time=rec.get("review_time") or "",
            apc=rec.get("apc") or None,
        )
        match = JournalMatch(
            journal=journal,
            score=rec.get("score", 0.0),
            confidence=rec.get("confidence", 0.0),
            match_reasons=rec.get("match_reasons", []),
            matched_fields=rec.get("matched_fields", []),
        )
        matches.append(match)

    pdf_bytes = exporter.export(
        title=title,
        abstract=abstract,
        recommendations=matches,
        paper_profile=profile,
    )

    # 生成文件名
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title[:30])
    filename = f"journal_recommendation_{safe_title}.pdf"

    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.post("/recommend/pdf")
async def recommend_pdf(request: Request):
    """推荐结果 PDF 导出"""
    pipeline = get_pipeline()

    # 解析请求
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.body()
        import json
        data = json.loads(body)
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        full_text = data.get("full_text", "")
        mode = data.get("mode", "abstract")
        top_k = data.get("top_k", 5)
        oa_preference = data.get("oa_preference", "any")
    else:
        form = await request.form()
        title = form.get("title", "")
        abstract = form.get("abstract", "")
        full_text = ""
        mode = form.get("mode", "abstract")
        top_k = int(form.get("top_k", 5))
        oa_preference = form.get("oa_preference", "any")
        file = form.get("file")

        # 处理文件上传
        if file and mode == "full":
            content = await file.read()
            blocks, full_text = extract_layout_blocks(content, file.filename)
            paper_ast = build_paper_ast(blocks, title=title)
            full_text = paper_ast.to_markdown()

    # 解析论文
    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text=full_text or "",
        mode=mode,
    )

    prompts = _load_prompts()

    try:
        profile = pipeline.parser.parse(paper_input, prompts["paper_profile_system"], prompts["paper_profile_user"])
    except PaperParserError as e:
        raise HTTPException(status_code=503, detail=f"论文解析失败: {e}")

    # 执行推荐
    quality_prompts = {
        "system": prompts.get("paper_quality_assessor_system", ""),
        "user": prompts.get("paper_quality_assessor_user", ""),
    }
    try:
        result = pipeline.recommend(
            paper_input,
            profile,
            top_k=top_k,
            mode=mode,
            oa_preference=oa_preference,
            quality_prompts=quality_prompts,
        )
    except PaperQualityError as e:
        raise HTTPException(status_code=503, detail=f"论文质量评估失败: {e}")

    # 生成 PDF
    exporter = PDFExporter()
    pdf_bytes = exporter.export(
        title=title,
        abstract=abstract,
        recommendations=result.get("recommendations", []),
        paper_profile=result.get("paper_profile"),
    )

    # 生成文件名
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title[:30])
    filename = f"期刊推荐报告_{safe_title}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


def _apply_quality_adjustment(
    ranked: list,
    paper_profile,
) -> list:
    """应用质量软权重调整（解耦）"""
    if paper_profile.paper_strength is None:
        return ranked

    strength = paper_profile.paper_strength
    base_adjustment = quality_adjustment_factor(strength)

    adjusted = []
    for journal, score, reasons in ranked:
        ccf_multiplier = {"A": 1.05, "B": 1.02, "C": 1.0}.get(journal.ccf_rating, 1.0)
        adjustment = base_adjustment * ccf_multiplier
        new_reasons = reasons.copy()
        if strength >= 0.65:
            new_reasons.append(f"强论文调整(+{(adjustment-1)*100:.0f}%)")
        elif strength < 0.35:
            new_reasons.append(f"弱论文调整({(adjustment-1)*100:.0f}%)")
        adjusted.append((journal, score * adjustment, new_reasons))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


@router.get("/recommend/stream")
@router.post("/recommend/stream")
async def recommend_stream(request: Request):
    """流式推荐期刊（SSE）"""
    import json
    pipeline = get_pipeline()

    # 支持 GET（URL参数）、POST（JSON body 或 FormData）
    content_type = request.headers.get("content-type", "")
    method = request.method

    if method == "GET":
        # GET 方式：读取 query params
        title = request.query_params.get("title", "")
        abstract = request.query_params.get("abstract", "")
        full_text = ""
        mode = request.query_params.get("mode", "abstract")
        top_k = int(request.query_params.get("top_k", 5))
        oa_preference = request.query_params.get("oa_preference", "any")
    elif "application/json" in content_type:
        # POST JSON body
        body = await request.body()
        data = json.loads(body)
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        full_text = data.get("full_text", "")
        mode = data.get("mode", "abstract")
        top_k = data.get("top_k", 5)
        oa_preference = data.get("oa_preference", "any")
    else:
        # POST FormData
        form = await request.form()
        title = form.get("title", "")
        abstract = form.get("abstract", "")
        full_text = ""
        mode = form.get("mode", "abstract")
        top_k = int(form.get("top_k", 5))
        oa_preference = form.get("oa_preference", "any")
        file = form.get("file")  # UploadFile 对象

        # 处理文件上传（full-text 模式）
        if file and mode == "full":
            content = await file.read()
            #print(f"[DEBUG] file size: {len(content)}, filename: {file.filename}")

            # 使用 PyMuPDF layout extraction
            blocks, _ = extract_layout_blocks(content, file.filename)
            #print(f"[DEBUG] extracted {len(blocks)} blocks")

            # 构建 Paper AST
            paper_ast = build_paper_ast(blocks, title=title)
            markdown = paper_ast.to_markdown()
            #print(f"[DEBUG] Paper AST: {len(paper_ast.sections)} sections")

            # 用 markdown 替换 full_text
            full_text = markdown

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
            await asyncio.sleep(0)

            paper_input = PaperInput(
                title=title,
                abstract=abstract or "",
                full_text=full_text or "",
                mode=mode,
            )

            prompts = _load_prompts()

            parser = pipeline.parser
            profile = parser.parse(
                paper_input,
                prompts["paper_profile_system"],
                prompts["paper_profile_user"],
            )

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
                profile.ccf_research_area = quality.ccf_research_area
                yield sse_event("progress", {
                    "stage": "quality",
                    "percent": 25,
                    "message": f"论文质量评估完成: {quality.quality_level} (strength={quality.paper_strength:.2f}, CCF领域={quality.ccf_research_area})"
                })
                await asyncio.sleep(0)

            # 阶段 2: 候选召回 (25-50%)
            yield sse_event("progress", {
                "stage": "retrieval",
                "percent": 25,
                "message": "正在召回候选期刊..."
            })
            await asyncio.sleep(0)

            # 阶段 3: 召回、规则排序与 LLM 精排由 pipeline 内部进度回调实时上报。
            yield sse_event("progress", {
                "stage": "ranking",
                "percent": 28,
                "message": "正在启动推荐流程..."
            })
            await asyncio.sleep(0)

            progress_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def emit_pipeline_progress(payload: dict) -> None:
                loop.call_soon_threadsafe(progress_queue.put_nowait, payload)

            # 调用统一 pipeline（包含检索、规则排序、LLM 精排）。
            # 这一步可能耗时较长；放入后台线程，并把 pipeline 内部阶段进度转发为 SSE。
            recommend_task = asyncio.create_task(asyncio.to_thread(
                pipeline.recommend,
                paper_input,
                profile,
                top_k=top_k,
                mode=mode,
                oa_preference=oa_preference,
                # stream endpoint 已在前面完成质量评估，避免在 pipeline 内重复调用一次 LLM。
                quality_prompts=None,
                progress_callback=emit_pipeline_progress,
            ))
            heartbeat_count = 0
            last_pipeline_percent = 28.0
            current_progress = {
                "stage": "ranking",
                "percent": last_pipeline_percent,
                "message": "正在启动推荐流程...",
            }
            while not recommend_task.done() or not progress_queue.empty():
                try:
                    payload = await asyncio.wait_for(
                        progress_queue.get(),
                        timeout=STREAM_PROGRESS_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if recommend_task.done():
                        continue
                    heartbeat_count += 1
                    percent = min(
                        79.0,
                        max(last_pipeline_percent + 0.5, round(79.0 - 34.0 / (heartbeat_count + 1), 1)),
                    )
                    current_progress = {
                        **current_progress,
                        "percent": percent,
                    }
                    yield sse_event("progress", {
                        **current_progress,
                        "heartbeat": True,
                    })
                    last_pipeline_percent = percent
                    continue

                percent = float(payload.get("percent", last_pipeline_percent))
                percent = max(last_pipeline_percent, min(percent, 82.0))
                last_pipeline_percent = percent
                current_progress = {
                    **payload,
                    "percent": percent,
                }
                yield sse_event("progress", {
                    **current_progress,
                })

            rec_result = await recommend_task

            yield sse_event("progress", {
                "stage": "ranking",
                "percent": 80,
                "message": "排序完成"
            })
            await asyncio.sleep(0)

            # 阶段 5: 构建推荐结果并流式推送 (80-100%)
            yield sse_event("progress", {
                "stage": "building",
                "percent": 85,
                "message": "正在构建推荐结果..."
            })
            await asyncio.sleep(0)

            recommendations_list = rec_result.get("recommendations", [])
            rank_method = rec_result.get("rank_method", "rule")

            recommendations = []
            for idx, rec in enumerate(recommendations_list):
                rec_dict = {
                    "journal_id": rec.journal.journal_id,
                    "journal_name": rec.journal.journal_name,
                    "score": rec.score,
                    "confidence": rec.confidence,
                    "match_reasons": rec.match_reasons or [],
                    "matched_fields": rec.matched_fields or ["research_area", "method_type"],
                    "tags": rec.journal.subject_tags,
                    "oa_type": rec.journal.oa_type,
                    "submission_url": rec.journal.submission_url or "",
                    "homepage_url": rec.journal.homepage_url or "",
                    "publisher": rec.journal.publisher or "",
                    "ccf_rating": rec.journal.ccf_rating or "",
                    "impact_like_score": rec.journal.impact_like_score,
                    "review_time": rec.journal.review_time or "",
                    "apc": rec.journal.apc,
                    "rank_method": rank_method,
                }
                recommendations.append(rec_dict)

                # 推送每个推荐结果
                progress_percent = 85 + (idx * 10 // max(len(recommendations_list), 1))
                yield sse_event("recommendation", rec_dict)
                yield sse_event("progress", {
                    "stage": "streaming",
                    "percent": progress_percent,
                    "message": f"正在推送第 {idx + 1}/{len(recommendations_list)} 条结果"
                })
                await asyncio.sleep(0)

            # 完成
            done_data = {
                "total": len(recommendations),
                "rank_method": rank_method,
                "mode_used": rec_result.get("mode_used", mode),
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

"""API 数据模型"""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """推荐请求"""
    title: str = Field(..., description="论文标题")
    abstract: Optional[str] = Field("", description="论文摘要")
    full_text: Optional[str] = Field("", description="论文全文")
    mode: str = Field("abstract", description="模式: title/abstract/full")
    top_k: int = Field(5, ge=1, le=20, description="推荐数量")
    oa_preference: str = Field("any", description="OA偏好: any/full_oa/hybrid")


class PaperProfileResponse(BaseModel):
    """论文特征响应"""
    title: str
    research_area: List[str]
    method_type: str
    paper_type: str
    keywords: List[str]


class JournalResponse(BaseModel):
    """期刊信息响应"""
    journal_id: str
    journal_name: str
    score: float
    confidence: float
    match_reasons: List[str]
    matched_fields: List[str]
    tags: List[str]
    oa_type: str
    quartile: Optional[str]
    submission_url: Optional[str]
    rank_method: str = Field("rule", description="排序方法: rule/llm")


class RecommendResponse(BaseModel):
    """推荐响应"""
    recommendations: List[JournalResponse]
    paper_profile: Optional[PaperProfileResponse]
    mode_used: str
    rank_method: str = Field("rule", description="排序方法: rule/llm")
    warning: Optional[str] = None


class JournalListItem(BaseModel):
    """期刊列表项"""
    journal_id: str
    journal_name: str
    subject_tags: List[str]
    oa_type: str
    quartile: Optional[str]


class JournalListResponse(BaseModel):
    """期刊列表响应"""
    journals: List[JournalListItem]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
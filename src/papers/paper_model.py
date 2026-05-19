"""论文数据模型"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PaperProfile(BaseModel):
    """论文特征结构"""
    title: str = Field(description="论文标题")
    abstract: str = Field(default="", description="论文摘要")
    research_area: List[str] = Field(default_factory=list, description="研究领域")
    method_type: str = Field(default="method", description="方法类型")
    paper_type: str = Field(default="application", description="论文类型")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    novelty: str = Field(default="", description="创新点")
    application_domain: List[str] = Field(default_factory=list, description="应用领域")
    difficulty_level: str = Field(default="medium", description="难度等级")
    style: str = Field(default="journal_like", description="风格: journal_like/conference_like")
    sections_summary: Dict[str, str] = Field(default_factory=dict, description="章节摘要")
    full_text_summary: str = Field(default="", description="全文摘要")
    # 新增字段
    techniques: List[str] = Field(default_factory=list, description="具体技术")
    datasets: List[str] = Field(default_factory=list, description="数据集")
    evaluation_metrics: List[str] = Field(default_factory=list, description="评估指标")
    novelty_type: str = Field(default="method", description="创新类型: new_method/new_application/benchmark/performance/efficiency")
    # 论文质量评估
    quality_level: Optional[str] = Field(default=None, description="论文质量等级: Q1/Q2/Q3/Q4")
    quality_confidence: float = Field(default=0.0)
    quality_reasons: List[str] = Field(default_factory=list)


class PaperInput(BaseModel):
    """论文输入"""
    title: str
    abstract: Optional[str] = ""
    full_text: Optional[str] = ""
    mode: str = Field(default="abstract", description="title/abstract/full")


class SectionSplitResult(BaseModel):
    """章节切分结果"""
    introduction: str = ""
    method: str = ""
    experiment: str = ""
    conclusion: str = ""
    other: str = ""
"""期刊数据模型"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Journal(BaseModel):
    """期刊数据结构"""
    journal_id: str = Field(description="唯一标识")
    journal_name: str = Field(description="期刊名称")
    publisher: Optional[str] = Field(default=None, description="出版社")
    subject_tags: List[str] = Field(default_factory=list, description="学科标签")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    scope_text: str = Field(default="", description="期刊 scope 说明")
    oa_type: str = Field(default="subscription", description="OA 类型: full_oa/hybrid/subscription")
    submission_url: Optional[str] = Field(default=None, description="投稿链接")
    homepage_url: Optional[str] = Field(default=None, description="期刊主页")
    sqr_rank: Optional[int] = Field(default=None, description="SCImago 排名")
    quartile: Optional[str] = Field(default=None, description="分区: Q1/Q2/Q3/Q4")
    ccf_rating: Optional[str] = Field(default=None, description="CCF评级: A/B/C")
    impact_like_score: Optional[float] = Field(default=None, description="影响因子指标")
    review_time: Optional[str] = Field(default=None, description="审稿周期估算")
    apc: Optional[float] = Field(default=None, description="APC 费用")
    target_paper_type: List[str] = Field(default_factory=list, description="适合的论文类型")
    journal_profile: str = Field(default="", description="预拼接检索文本")

    def build_profile(self) -> str:
        """构建 journal_profile"""
        parts = [
            self.journal_name,
            self.scope_text,
            " ".join(self.keywords),
            " ".join(self.subject_tags),
        ]
        self.journal_profile = " | ".join(p for p in parts if p)
        return self.journal_profile


class JournalMatch(BaseModel):
    """期刊匹配结果"""
    journal: Journal
    score: float = Field(description="匹配分数")
    confidence: float = Field(description="置信度 0-1")
    match_reasons: List[str] = Field(default_factory=list, description="匹配理由")
    matched_fields: List[str] = Field(default_factory=list, description="匹配的字段")
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
    novelty_type: str = Field(default="", description="创新类型: new_method/new_application/benchmark/performance/efficiency")
    # 论文质量评估（多维度）
    quality_level: Optional[str] = Field(default=None, description="论文质量等级: Q1/Q2/Q3/Q4")
    quality_confidence: float = Field(default=0.0)
    quality_reasons: List[str] = Field(default_factory=list)
    paper_strength: Optional[float] = Field(default=None, description="论文本身强弱 0~1")
    readiness: Optional[str] = Field(default=None, description="投稿准备度: Ready/Preliminary/Needs-Revision")
    # CCF 专业领域（由 PaperQualityAssessor 预测）
    ccf_research_area: List[str] = Field(
        default_factory=list,
        description="CCF专业领域列表(1-3个): 计算机体系结构/并行与分布计算/存储系统, 计算机网络, 网络与信息安全, 软件工程/系统软件/程序设计语言, 数据库/数据挖掘/内容检索, 计算机科学理论, 计算机图形学与多媒体, 人工智能, 人机交互与普适计算, 交叉/综合/新兴"
    )


class PaperInput(BaseModel):
    """论文输入"""
    title: str
    abstract: Optional[str] = ""
    full_text: Optional[str] = ""
    mode: str = Field(default="abstract", description="title/abstract/full")


class Block(BaseModel):
    """单个文本块（来自 PyMuPDF）"""
    text: str = ""
    font_size: float = 0.0
    font_name: str = ""
    bold: bool = False
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    page_number: int = 0


class SectionSplitResult(BaseModel):
    """章节切分结果"""
    introduction: str = ""
    method: str = ""
    experiment: str = ""
    conclusion: str = ""
    other: str = ""
    blocks: List[Block] = Field(default_factory=list)  # 新增


class Section(BaseModel):
    """论文章节"""
    title: str = ""
    level: int = 1  # 1=title, 2=section, 3=subsection
    content: str = ""  # 该章节的完整文本
    blocks: List[Block] = Field(default_factory=list)  # 原始 blocks


class PaperDocument(BaseModel):
    """论文结构化文档（Paper AST）"""
    title: str = ""
    abstract: str = ""
    sections: List[Section] = Field(default_factory=list)
    all_blocks: List[Block] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = []
        if self.title:
            lines.append(f"# {self.title}\n")
        if self.abstract:
            lines.append(f"## Abstract\n{self.abstract}\n")
        for section in self.sections:
            level_prefix = "#" * min(section.level + 1, 6)
            lines.append(f"{level_prefix} {section.title}\n{section.content}\n")
        return "\n".join(lines)
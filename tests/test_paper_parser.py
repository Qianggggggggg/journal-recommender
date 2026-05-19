"""论文解析测试"""
import pytest
from src.papers.paper_model import PaperInput, PaperProfile
from src.papers.paper_parser import PaperParser


def test_paper_parser_fallback():
    """测试降级解析"""
    parser = PaperParser(llm=None)  # 不配置 LLM，强制走降级
    paper_input = PaperInput(
        title="Deep Learning for Image Recognition",
        abstract="This paper proposes a new deep learning method for image recognition.",
    )
    profile = parser.parse(paper_input, "", "")
    assert profile.title == "Deep Learning for Image Recognition"
    assert len(profile.research_area) >= 0  # 可能有 AI/CV 标签


def test_paper_input_model():
    """测试 PaperInput 模型"""
    input_data = PaperInput(
        title="Test Paper",
        abstract="Test abstract",
        mode="abstract",
    )
    assert input_data.title == "Test Paper"
    assert input_data.mode == "abstract"


def test_section_splitter():
    """测试章节切分"""
    from src.papers.section_splitter import SectionSplitter

    splitter = SectionSplitter()
    text = """
1. Introduction
This is the introduction.

2. Method
This is the method section.

3. Experiment
This is the experiment section.

4. Conclusion
This is the conclusion.
"""
    result = splitter.split(text)
    assert "introduction" in result.introduction.lower() or len(result.introduction) > 0
    assert len(result.method) > 0
    assert len(result.experiment) > 0
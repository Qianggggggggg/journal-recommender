"""论文解析测试（纯LLM版）"""
import pytest
from unittest.mock import MagicMock
from src.papers.paper_model import PaperInput, PaperProfile
from src.papers.paper_parser import PaperParser, PaperParserError


def test_paper_parser_no_llm_raises_error():
    """无 LLM 时应抛出明确错误"""
    with pytest.raises(PaperParserError, match="LLM not configured"):
        PaperParser(llm=None)


def test_paper_parser_llm_success():
    """LLM 解析成功"""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(content='{"research_area": ["AI", "NLP"], "method_type": "method", "paper_type": "application", "keywords": ["transformer", "attention"], "novelty": "new architecture", "application_domain": ["text mining"], "techniques": ["transformer"], "datasets": ["WikiSQL"], "evaluation_metrics": ["accuracy"], "novelty_type": "new_method"}')

    parser = PaperParser(llm=mock_llm)
    paper_input = PaperInput(
        title="Deep Learning for Text Mining",
        abstract="This paper proposes a new deep learning method for text mining.",
    )
    profile = parser.parse(paper_input, "system", "user")

    assert profile.title == "Deep Learning for Text Mining"
    assert "AI" in profile.research_area or "NLP" in profile.research_area


def test_paper_parser_llm_parsing_error():
    """LLM 返回格式错误时重试后抛出明确错误"""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(content="无法解析的响应")

    parser = PaperParser(llm=mock_llm)
    paper_input = PaperInput(title="Test Paper", abstract="test abstract")

    with pytest.raises(PaperParserError, match="LLM响应格式错误"):
        parser.parse(paper_input, "system", "user")

    assert mock_llm.chat.call_count == 3


def test_paper_parser_llm_network_error_retry():
    """LLM 网络错误时重试"""
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = [
        Exception("Network error"),
        Exception("Network error"),
        MagicMock(content='{"research_area": ["AI"], "method_type": "method", "paper_type": "application", "keywords": [], "novelty": "", "application_domain": [], "techniques": [], "datasets": [], "evaluation_metrics": [], "novelty_type": ""}')
    ]

    parser = PaperParser(llm=mock_llm)
    paper_input = PaperInput(title="Test Paper", abstract="test")
    profile = parser.parse(paper_input, "system", "user")

    assert mock_llm.chat.call_count == 3
    assert profile.title == "Test Paper"


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
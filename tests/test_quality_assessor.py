"""论文质量评估器测试（纯LLM版）"""
import pytest
from unittest.mock import MagicMock, patch
from src.papers.quality_assessor import PaperQualityAssessor, PaperQuality, PaperQualityError
from src.papers.paper_model import PaperInput, PaperProfile


class TestPaperQualityAssessor:
    """PaperQualityAssessor 单元测试（纯LLM）"""

    def test_no_llm_raises_error(self):
        """无 LLM 时应抛出明确错误"""
        with pytest.raises(PaperQualityError, match="LLM not configured"):
            PaperQualityAssessor(llm=None)

    def test_assess_by_llm_success(self):
        """LLM 评估成功"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 3, "rigor_score": 2, "reproducibility_score": 2, "significance_score": 2, "clarity_score": 2, "confidence": 0.8, "reasons": ["方法创新"], "novelty_evidence": "提出了新方法", "rigor_evidence": "实验充分", "reproducibility_evidence": "数据集完整", "significance_evidence": "问题重要", "clarity_evidence": "论述清晰"}')

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test Paper", abstract="This is a test abstract " * 30)
        profile = PaperProfile(
            title="Test Paper",
            novelty_type="new_method",
            datasets=["ImageNet", "COCO"],
            techniques=["transformer", "attention"],
            evaluation_metrics=["accuracy", "f1", "mAP"],
        )

        result = assessor.assess(paper_input, profile, "system", "user")

        assert isinstance(result, PaperQuality)
        assert result.quality_level == "Q1"
        assert result.paper_strength >= 0.75
        assert result.confidence == 0.8

    def test_assess_by_llm_parsing_error_retry(self):
        """LLM 返回格式错误时重试后抛出明确错误"""
        mock_llm = MagicMock()
        # 连续返回无法解析的响应
        mock_llm.chat.return_value = MagicMock(content="这是一条无法解析的响应")

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test Paper", abstract="test")
        profile = PaperProfile(title="Test Paper")

        # 重试3次后失败，抛出 PaperQualityError
        with pytest.raises(PaperQualityError, match="LLM响应格式错误"):
            assessor.assess(paper_input, profile, "system", "user")

        # 验证重试了3次
        assert mock_llm.chat.call_count == 3

    def test_assess_by_llm_network_error_retry(self):
        """LLM 网络错误时重试"""
        mock_llm = MagicMock()
        # 前两次失败，第三次成功
        mock_llm.chat.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            MagicMock(content='{"novelty_score": 2, "rigor_score": 1, "reproducibility_score": 1, "significance_score": 1, "clarity_score": 1, "confidence": 0.5, "reasons": [], "novelty_evidence": "", "rigor_evidence": "", "reproducibility_evidence": "", "significance_evidence": "", "clarity_evidence": ""}')
        ]

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test Paper", abstract="test")
        profile = PaperProfile(title="Test Paper")

        result = assessor.assess(paper_input, profile, "system", "user")

        assert mock_llm.chat.call_count == 3
        assert result.paper_strength is not None

    def test_readiness_levels(self):
        """测试三种准备度状态"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 2, "rigor_score": 1, "reproducibility_score": 1, "significance_score": 1, "clarity_score": 1, "confidence": 0.5, "reasons": [], "novelty_evidence": "", "rigor_evidence": "", "reproducibility_evidence": "", "significance_evidence": "", "clarity_evidence": ""}')
        assessor = PaperQualityAssessor(llm=mock_llm)

        # 高 strength -> Ready
        high_input = PaperInput(title="Test", abstract="A" * 500)
        high_profile = PaperProfile(
            title="Test", novelty_type="new_method",
            datasets=["d1", "d2"], techniques=["t1", "t2"],
            evaluation_metrics=["m1", "m2", "m3"],
        )
        # 模拟高评分
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 3, "rigor_score": 3, "reproducibility_score": 3, "significance_score": 3, "clarity_score": 3, "confidence": 0.9, "reasons": [], "novelty_evidence": "", "rigor_evidence": "", "reproducibility_evidence": "", "significance_evidence": "", "clarity_evidence": ""}')
        high_result = assessor.assess(high_input, high_profile, "system", "user")
        assert high_result.readiness == "Ready"

        # 低 strength -> Needs-Revision
        low_input = PaperInput(title="Test", abstract="short")
        low_profile = PaperProfile(title="Test")
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 1, "rigor_score": 0, "reproducibility_score": 0, "significance_score": 0, "clarity_score": 0, "confidence": 0.3, "reasons": [], "novelty_evidence": "", "rigor_evidence": "", "reproducibility_evidence": "", "significance_evidence": "", "clarity_evidence": ""}')
        low_result = assessor.assess(low_input, low_profile, "system", "user")
        assert low_result.readiness == "Needs-Revision"

    def test_strength_to_level_mapping(self):
        """测试 strength -> level 映射"""
        # Q1: >= 0.75
        assert PaperQuality._strength_to_level(0.8) == "Q1"
        assert PaperQuality._strength_to_level(0.75) == "Q1"
        # Q2: >= 0.55
        assert PaperQuality._strength_to_level(0.6) == "Q2"
        assert PaperQuality._strength_to_level(0.55) == "Q2"
        # Q3: >= 0.35
        assert PaperQuality._strength_to_level(0.4) == "Q3"
        assert PaperQuality._strength_to_level(0.35) == "Q3"
        # Q4: < 0.35
        assert PaperQuality._strength_to_level(0.3) == "Q4"
        assert PaperQuality._strength_to_level(0.0) == "Q4"

    def test_readiness_inference(self):
        """测试准备度推断逻辑"""
        # strength >= 0.6 && novelty >= 2 -> Ready
        assert PaperQuality._strength_to_readiness(0.7, 2) == "Ready"
        # strength >= 0.35 -> Preliminary
        assert PaperQuality._strength_to_readiness(0.5, 1) == "Preliminary"
        # else -> Needs-Revision
        assert PaperQuality._strength_to_readiness(0.3, 1) == "Needs-Revision"

    def test_paper_strength_range(self):
        """paper_strength 应在 [0, 1] 范围内"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 2, "rigor_score": 1, "reproducibility_score": 1, "significance_score": 1, "clarity_score": 1, "confidence": 0.5, "reasons": [], "novelty_evidence": "", "rigor_evidence": "", "reproducibility_evidence": "", "significance_evidence": "", "clarity_evidence": ""}')
        assessor = PaperQualityAssessor(llm=mock_llm)

        test_cases = [
            (PaperInput(title="Test", abstract=""), PaperProfile(title="Test")),
            (PaperInput(title="Test", abstract="A" * 500), PaperProfile(title="Test", novelty_type="new_method")),
        ]

        for inp, prof in test_cases:
            result = assessor.assess(inp, prof, "system", "user")
            assert 0.0 <= result.paper_strength <= 1.0

    def test_confidence_from_llm(self):
        """置信度应从 LLM 响应中提取"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 2, "rigor_score": 1, "reproducibility_score": 1, "significance_score": 1, "clarity_score": 1, "confidence": 0.95, "reasons": ["测试"], "novelty_evidence": "", "rigor_evidence": "", "reproducibility_evidence": "", "significance_evidence": "", "clarity_evidence": ""}')

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test", abstract="test")
        profile = PaperProfile(title="Test")

        result = assessor.assess(paper_input, profile, "system", "user")

        assert result.confidence == 0.95

    def test_evidence_from_llm(self):
        """证据字段应从 LLM 响应中提取"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content='{"novelty_score": 3, "rigor_score": 2, "reproducibility_score": 2, "significance_score": 2, "clarity_score": 2, "confidence": 0.8, "reasons": ["创新性强"], "novelty_evidence": "提出了新的图索引方法", "rigor_evidence": "在4个数据集上进行了充分实验", "reproducibility_evidence": "使用了标准基准数据集", "significance_evidence": "解决了现有RAG系统的关键问题", "clarity_evidence": "论文结构清晰，论述流畅"}')

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test", abstract="A" * 300)
        profile = PaperProfile(title="Test", novelty_type="new_method")

        result = assessor.assess(paper_input, profile, "system", "user")

        assert result.evidence["novelty"] == "提出了新的图索引方法"
        assert result.evidence["rigor"] == "在4个数据集上进行了充分实验"
        assert "创新性强" in result.reasons
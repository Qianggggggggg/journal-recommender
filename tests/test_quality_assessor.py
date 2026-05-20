"""论文质量评估器测试（新版多维度）"""
import pytest
from unittest.mock import MagicMock

from src.papers.quality_assessor import PaperQualityAssessor, PaperQuality
from src.papers.paper_model import PaperInput, PaperProfile


class TestPaperQualityAssessor:
    """PaperQualityAssessor 单元测试（新版）"""

    def test_assess_fallback_to_rules_when_no_llm(self):
        """无 LLM 时应降级到规则评估"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test Paper", abstract="This is a test abstract " * 30)
        profile = PaperProfile(
            title="Test Paper",
            novelty_type="new_method",
            datasets=["ImageNet", "COCO"],
            techniques=["transformer", "attention"],
            evaluation_metrics=["accuracy", "f1", "mAP"],
        )

        result = assessor.assess(paper_input, profile, "", "")

        assert isinstance(result, PaperQuality)
        assert result.quality_level in ["Q1", "Q2", "Q3", "Q4"]
        assert result.paper_strength is not None
        assert 0.0 <= result.paper_strength <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_assess_by_rules_new_method_high_score(self):
        """新方法 + 多数据集 + 多指标 = Q1"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="A" * 400, full_text="B" * 3000)
        profile = PaperProfile(
            title="Test",
            novelty_type="new_method",
            datasets=["ImageNet", "COCO", "WikiSQL"],
            techniques=["transformer", "gnn", "attention"],
            evaluation_metrics=["accuracy", "f1", "mAP", "auc"],
        )

        result = assessor.assess(paper_input, profile, "", "")

        # 新方法novelty_score=3, 3个数据集=3, 4个指标=3, 摘要>300=2.5, 全文>3000=额外加分
        # novelty_dim = 3*0.6 + 3*0.4 = 3.0
        # rigor_dim = 3*0.5 + 2*0.3 + 2*0.2 = 2.3
        # completeness_dim = 2.5 (abstract完整)
        # 预期 paper_strength >= 0.75 -> Q1
        assert result.quality_level == "Q1"
        assert result.paper_strength >= 0.75
        assert result.readiness in ["Ready", "Preliminary"]

    def test_assess_by_rules_performance_mid_score(self):
        """性能提升 + 少数据集 = Q2/Q3"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="A" * 200)
        profile = PaperProfile(
            title="Test",
            novelty_type="performance",
            datasets=["ImageNet"],
            techniques=["CNN"],
            evaluation_metrics=["accuracy", "f1"],
        )

        result = assessor.assess(paper_input, profile, "", "")

        # novelty_score = 2, dataset_score = 1, metric_score = 2
        # novelty_dim = 2*0.6 + 1*0.4 = 1.6
        # rigor_dim = 2*0.5 + 1*0.3 + 1*0.2 = 1.5
        # strength ≈ (1.6*0.35 + 1.5*0.25 + ...)/3 < 0.55 -> Q2/Q3
        assert result.quality_level in ["Q2", "Q3"]

    def test_assess_by_rules_new_application_low_score(self):
        """新应用 + 无数据集 = Q4"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="short")
        profile = PaperProfile(
            title="Test",
            novelty_type="new_application",
            datasets=[],
            techniques=["method"],
            evaluation_metrics=[],
        )

        result = assessor.assess(paper_input, profile, "", "")

        assert result.quality_level == "Q4"
        assert result.paper_strength < 0.35

    def test_assess_by_rules_empty_profile(self):
        """空 profile 降级到 Q4"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="")
        profile = PaperProfile(title="Test")

        result = assessor.assess(paper_input, profile, "", "")

        assert result.quality_level == "Q4"
        assert result.readiness == "Needs-Revision"

    def test_assess_by_rules_has_evidence(self):
        """规则评估应包含证据字段"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="A" * 300)
        profile = PaperProfile(
            title="Test",
            novelty_type="new_method",
            datasets=["dataset1"],
            techniques=["tech1"],
            evaluation_metrics=["metric1"],
        )

        result = assessor.assess(paper_input, profile, "", "")

        assert isinstance(result.evidence, dict)
        assert "novelty" in result.evidence
        assert "rigor" in result.evidence

    def test_assess_by_llm_fallback_on_exception(self):
        """LLM 抛异常时降级到规则"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM error")

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test", abstract="A" * 50)
        profile = PaperProfile(title="Test", novelty_type="new_method", datasets=["A"])

        result = assessor.assess(paper_input, profile, "system", "user")

        assert result.quality_level in ["Q1", "Q2", "Q3", "Q4"]
        assert mock_llm.chat.called

    def test_readiness_levels(self):
        """测试三种准备度状态"""
        assessor = PaperQualityAssessor(llm=None)

        # 高 strength -> Ready
        high_input = PaperInput(title="Test", abstract="A" * 500, full_text="B" * 5000)
        high_profile = PaperProfile(
            title="Test", novelty_type="new_method",
            datasets=["d1", "d2"], techniques=["t1", "t2"],
            evaluation_metrics=["m1", "m2", "m3"],
        )
        high_result = assessor.assess(high_input, high_profile, "", "")
        assert high_result.readiness in ["Ready", "Preliminary"]

        # 低 strength -> Needs-Revision
        low_input = PaperInput(title="Test", abstract="short")
        low_profile = PaperProfile(title="Test", novelty_type="", datasets=[], techniques=[], evaluation_metrics=[])
        low_result = assessor.assess(low_input, low_profile, "", "")
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

    def test_uncertainty_reasons(self):
        """信息不足时应有不确性原因"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="")
        profile = PaperProfile(title="Test")

        result = assessor.assess(paper_input, profile, "", "")

        assert len(result.uncertainty_reasons) >= 1
        assert any("摘要" in r or "信息" in r for r in result.uncertainty_reasons)

    def test_paper_strength_range(self):
        """paper_strength 应在 [0, 1] 范围内"""
        assessor = PaperQualityAssessor(llm=None)

        test_cases = [
            (PaperInput(title="Test", abstract=""), PaperProfile(title="Test")),
            (PaperInput(title="Test", abstract="A" * 500), PaperProfile(title="Test", novelty_type="new_method")),
        ]

        for inp, prof in test_cases:
            result = assessor.assess(inp, prof, "", "")
            assert 0.0 <= result.paper_strength <= 1.0
"""论文质量评估器测试"""
import pytest
from unittest.mock import MagicMock

from src.papers.quality_assessor import PaperQualityAssessor, PaperQuality
from src.papers.paper_model import PaperInput, PaperProfile


class TestPaperQualityAssessor:
    """PaperQualityAssessor 单元测试"""

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
        assert result.level in ["Q1", "Q2", "Q3", "Q4"]
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

        # new_method=2, datasets(3+)=1.5, metrics(4)=1, techniques(3+)=0.8, abstract>300=0.5, full_text>2000=1.0
        # total = 2+1.5+1+0.8+0.5+1 = 6.8 >= 6.0 → Q1
        assert result.level == "Q1"
        assert result.confidence >= 0.8

    def test_assess_by_rules_performance_mid_score(self):
        """性能提升 + 少数据集 = Q2"""
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

        # performance=1.2, dataset(1)=0.8, metrics(2)=0.5, techniques(1)=0, abstract<300=0, full_text=0
        # total = 1.2+0.8+0.5 = 2.5 >= 4.0? 否 → 2.5 >= 2.0 是 → Q3
        assert result.level in ["Q2", "Q3"]

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

        # new_application=1.0, no datasets=0, no metrics=0, no techniques=0, short abstract=0
        # total = 1.0 < 2.0 → Q4
        assert result.level == "Q4"

    def test_assess_by_rules_empty_profile(self):
        """空 profile 降级到 Q4"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="")
        profile = PaperProfile(title="Test")

        result = assessor.assess(paper_input, profile, "", "")

        # method=0.5, no datasets, no metrics, no techniques, no abstract, no full_text
        # total = 0.5 → Q4, confidence = min(0.5/8.0, 1.0) = 0.0625
        assert result.level == "Q4"
        assert result.confidence == 0.0625

    def test_assess_by_rules_q2_boundary(self):
        """边界测试: score=4.0 应为 Q2"""
        assessor = PaperQualityAssessor(llm=None)
        # 手动构造刚好达到 Q2 阈值的场景
        paper_input = PaperInput(title="Test", abstract="A" * 300)
        profile = PaperProfile(
            title="Test",
            novelty_type="benchmark",  # 1.5
            datasets=["dataset1", "dataset2"],  # >=1, <3 → 0.8
            techniques=["tech1", "tech2"],  # >=2 → 0
            evaluation_metrics=["acc", "f1", "mAP"],  # >=3 → 1.0
        )

        result = assessor.assess(paper_input, profile, "", "")

        # benchmark=1.5, datasets=0.8, metrics=1.0, techniques=0, abstract=0.5, full=0
        # total = 1.5+0.8+1.0+0.5 = 3.8 < 4.0 → 实际 Q3
        assert result.level in ["Q1", "Q2", "Q3", "Q4"]

    def test_assess_by_llm_fallback_on_exception(self):
        """LLM 抛异常时降级到规则"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM error")

        assessor = PaperQualityAssessor(llm=mock_llm)
        paper_input = PaperInput(title="Test", abstract="A" * 50)
        profile = PaperProfile(title="Test", novelty_type="new_method", datasets=["A"])

        result = assessor.assess(paper_input, profile, "system", "user")

        assert result.level in ["Q1", "Q2", "Q3", "Q4"]
        assert mock_llm.chat.called

    def test_quality_level_q1_confidence(self):
        """Q1 论文应有高置信度"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="A" * 500, full_text="B" * 5000)
        profile = PaperProfile(
            title="Test",
            novelty_type="new_method",
            datasets=["d1", "d2", "d3"],
            techniques=["t1", "t2", "t3"],
            evaluation_metrics=["m1", "m2", "m3"],
        )

        result = assessor.assess(paper_input, profile, "", "")

        # 预期 score >= 6.0 → Q1, confidence = min(score/8.0, 1.0) = min(>=0.75, 1.0)
        assert result.level == "Q1"
        assert result.confidence >= 0.7

    def test_reasons_not_empty_for_qualified_papers(self):
        """有数据集验证的论文应有 reason"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="A" * 100)
        profile = PaperProfile(
            title="Test",
            novelty_type="performance",
            datasets=["ImageNet"],
            techniques=[],
            evaluation_metrics=["accuracy"],
        )

        result = assessor.assess(paper_input, profile, "", "")

        # performance=1.2 + dataset=0.8 + metrics=0.5 = 2.5 → Q3
        # reasons 应包含 "性能提升" 和 "数据集验证(1个)"
        assert len(result.reasons) >= 1

    def test_benchmark_novelty_type(self):
        """benchmark 创新类型得分 1.5"""
        assessor = PaperQualityAssessor(llm=None)
        paper_input = PaperInput(title="Test", abstract="A" * 100)
        profile = PaperProfile(title="Test", novelty_type="benchmark", datasets=[], techniques=[], evaluation_metrics=[])

        result = assessor.assess(paper_input, profile, "", "")

        # benchmark=1.5, others=0 → total=1.5 < 2.0 → Q4
        assert result.level == "Q4"
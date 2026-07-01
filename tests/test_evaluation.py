"""评估流程测试"""
import pytest
from scripts.evaluate_recommender import (
    get_paper_quality_level,
    get_expected_ccf_levels,
    is_level_match,
    estimate_paper_strength,
)


def test_paper_quality_level():
    """测试论文质量等级划分"""
    assert get_paper_quality_level(0.8) == "strong"
    assert get_paper_quality_level(0.65) == "strong"
    assert get_paper_quality_level(0.64) == "medium"
    assert get_paper_quality_level(0.5) == "medium"
    assert get_paper_quality_level(0.49) == "weak"
    assert get_paper_quality_level(0.2) == "weak"


def test_expected_ccf_levels():
    """测试期望 CCF 等级"""
    assert "A" in get_expected_ccf_levels("strong")
    assert "B" in get_expected_ccf_levels("strong")
    assert "B" in get_expected_ccf_levels("medium")
    assert "C" in get_expected_ccf_levels("medium")
    assert "C" in get_expected_ccf_levels("weak")


def test_level_match():
    """测试 Level Match 判断"""
    # 强论文应该匹配 A 或 B
    assert is_level_match(0.8, "A")
    assert is_level_match(0.8, "B")
    assert not is_level_match(0.8, "C")

    # 中论文应该匹配 B 或 C
    assert is_level_match(0.5, "B")
    assert is_level_match(0.5, "C")
    assert not is_level_match(0.5, "A")

    # 弱论文应该匹配 C 或 N/A
    assert is_level_match(0.3, "C")
    assert is_level_match(0.3, "N/A")
    assert not is_level_match(0.3, "A")


def test_estimate_paper_strength():
    """测试论文质量估算"""
    # 无摘要
    strength = estimate_paper_strength("")
    assert 0.3 <= strength <= 0.5

    # 有强信号词
    strength = estimate_paper_strength("We propose a novel state-of-the-art method that significantly improves accuracy")
    assert strength > 0.5

    # 有弱信号词
    strength = estimate_paper_strength("This is a preliminary study with limited experiments")
    assert strength < 0.5

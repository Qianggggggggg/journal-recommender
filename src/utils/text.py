"""文本处理工具"""
import re
from typing import List


def truncate_text(text: str, max_length: int = 2000) -> str:
    """截断文本到最大长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白和特殊字符"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """简单关键词提取（基于频率）"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    from collections import Counter
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_k)]


def split_sentences(text: str) -> List[str]:
    """分句"""
    sentences = re.split(r'[。！？\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def quality_adjustment_factor(strength: float) -> float:
    """
    根据论文质量强度计算调整因子。

    公式：1.0 + 0.2 * (strength - 0.5)
    - strength >= 0.65: 因子 > 1.0，质量好的论文略微提升
    - strength == 0.5:  因子 == 1.0，中等论文不调整
    - strength < 0.5:  因子 < 1.0，质量差的论文略微降低
    """
    factor = 1.0 + 0.2 * (strength - 0.5)
    return max(0.85, min(1.10, factor))


# P1 (2026-06-16 diagnostic): strength-aware CCF-tier multiplier.
# Previous behaviour (hardcoded in pipeline.py::_apply_quality_adjustment):
#     ccf_multiplier = {"A": 1.05, "B": 1.02, "C": 1.0}
# This systematically suppressed CCF-C application-oriented journals
# (HCI, security engineering, education, network management) and cost
# 20-30 pp of hit@5 vs CCF-A on holdout240 (49-56% vs 76-85%).
#
# New behaviour: invert the bias for weak papers so a paper that is
# itself not strong enough for top venues gets a relative boost for
# matching C-tier venues. Strong papers keep the original A>B>C table.
#
# Locked by tests/test_quality_adjustment.py::TestQualityAdjustmentMultiplier.
_STRONG_PAPER_TIER_MULTIPLIER = {"A": 1.05, "B": 1.02, "C": 1.0}
_WEAK_PAPER_TIER_MULTIPLIER = {"A": 0.95, "B": 1.0, "C": 1.05}
# Threshold matches the neutral point of quality_adjustment_factor (strength=0.5
# yields base=1.0). Below this, the weak-paper table is used.
_WEAK_PAPER_STRENGTH_THRESHOLD = 0.5


def quality_adjustment_multiplier(strength, ccf_rating):
    """Return a CCF-tier multiplier that depends on paper strength.

    Behaviour:
      * strength >= 0.5 (or strength is None):
          A=1.05, B=1.02, C=1.0 (preserves the original A>B>C preference).
      * strength <  0.5:
          A=0.95, B=1.0,  C=1.05 (counter-bias: weak papers get a relative
                                     boost for matching C-tier venues).
      * Unknown / missing CCF rating: 1.0 (no bias).

    Composed with quality_adjustment_factor, the final adjustment is
    ``base * tier`` where base is monotonic in strength and tier flips
    at strength=0.5.
    """
    if strength is None:
        return 1.0
    table = (
        _WEAK_PAPER_TIER_MULTIPLIER
        if strength < _WEAK_PAPER_STRENGTH_THRESHOLD
        else _STRONG_PAPER_TIER_MULTIPLIER
    )
    if not ccf_rating:
        return 1.0
    return table.get(ccf_rating, 1.0)
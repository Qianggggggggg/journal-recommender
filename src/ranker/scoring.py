"""分数计算工具"""


class Scoring:
    """分数计算工具"""

    @staticmethod
    def normalize_scores(scores: list) -> list:
        """归一化分数到 [0, 1]"""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    @staticmethod
    def combine_scores(weights: dict, features: dict) -> float:
        """加权求和"""
        score = 0.0
        for key, weight in weights.items():
            if key in features:
                score += weight * features[key]
        return score

    @staticmethod
    def confidence_from_scores(rule_score: float, llm_score: float) -> float:
        """综合置信度"""
        # 两者加权平均
        return 0.4 * min(rule_score / 10, 1.0) + 0.6 * llm_score
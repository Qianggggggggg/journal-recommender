"""LLM 模块测试"""
import pytest
from src.utils.llm import LLMResponse, MiniMaxLLM, parse_json_response


def test_llm_response_model():
    """验证 LLMResponse 模型结构"""
    response = LLMResponse(content="test", model="test-model", usage={})
    assert response.content == "test"
    assert response.model == "test-model"


def test_parse_json_response_rejects_bare_string():
    """裸 JSON 字符串不是系统期望的结构化 LLM 响应。"""
    assert parse_json_response('\n "novelty_score"') is None


def test_parse_json_response_finds_json_after_non_json_braces():
    """LLM 先写分析且分析里含花括号时，应继续寻找后面的合法 JSON。"""
    content = """
    # 期刊匹配度分析

    论文使用统计模型，说明中可能出现集合 {microbiome, phylogeny}，这不是 JSON。

    ```json
    {
      "rankings": [
        {
          "journal_id": "bioinformatics",
          "score": 0.95,
          "reasons": ["Scope对齐：覆盖微生物组统计方法"],
          "confidence": 0.9
        }
      ]
    }
    ```
    """

    parsed = parse_json_response(content)

    assert parsed == [
        {
            "journal_id": "bioinformatics",
            "score": 0.95,
            "reasons": ["Scope对齐：覆盖微生物组统计方法"],
            "confidence": 0.9,
        }
    ]

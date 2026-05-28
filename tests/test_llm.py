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

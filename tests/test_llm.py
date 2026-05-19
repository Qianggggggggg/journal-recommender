"""LLM 模块测试"""
import pytest
from src.utils.llm import LLMResponse, MiniMaxLLM


def test_llm_response_model():
    """验证 LLMResponse 模型结构"""
    response = LLMResponse(content="test", model="test-model", usage={})
    assert response.content == "test"
    assert response.model == "test-model"
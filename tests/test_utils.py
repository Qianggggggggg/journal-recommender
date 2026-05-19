"""工具模块测试"""
import pytest
from src.utils.text import clean_text, truncate_text, extract_keywords


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("hello\x00world") == "helloworld"


def test_truncate_text():
    assert truncate_text("hello world", max_length=5) == "hello..."
    assert truncate_text("hi", max_length=10) == "hi"


def test_extract_keywords():
    keywords = extract_keywords("deep learning neural network learning deep", top_k=3)
    assert "learning" in keywords or "deep" in keywords
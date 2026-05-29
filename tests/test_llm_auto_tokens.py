#!/usr/bin/env python3
"""测试动态max_tokens功能"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.utils.llm import MiniMaxLLM


class TestChatAutoMaxTokens:
    """测试 chat_auto 方法的动态 max_tokens 功能"""

    def test_chat_auto_exists(self):
        """验证 chat_auto 方法存在"""
        llm = MiniMaxLLM(api_key="test", model="MiniMax-M2.7")
        assert hasattr(llm, 'chat_auto'), "chat_auto 方法应该存在"

    def test_chat_auto_accepts_system_and_user(self):
        """验证 chat_auto 接受 system 和 user 参数"""
        llm = MiniMaxLLM(api_key="test", model="MiniMax-M2.7")
        # 只是检查方法签名，不实际调用API
        import inspect
        sig = inspect.signature(llm.chat_auto)
        params = list(sig.parameters.keys())
        assert 'system' in params, "chat_auto 应该接受 system 参数"
        assert 'user' in params, "chat_auto 应该接受 user 参数"

    def test_estimate_tokens(self):
        """测试 token 估算功能"""
        llm = MiniMaxLLM(api_key="test", model="MiniMax-M2.7")

        # 测试中英文混合文本的 token 估算
        short_text = "Hello world"  # 约 2-3 tokens
        long_text = "Hello " * 1000  # 约 1000 tokens

        # 估算应该约为 len / 4
        short_tokens = llm._estimate_tokens(short_text)
        long_tokens = llm._estimate_tokens(long_text)

        assert short_tokens < len(short_text), "短文本估算应该 < 字符数"
        assert long_tokens < len(long_text), "长文本估算应该 < 字符数"
        assert long_tokens > short_tokens, "长文本应该有更多 tokens"

    def test_calculate_max_tokens_short_input(self):
        """测试短输入时 max_tokens 计算"""
        llm = MiniMaxLLM(api_key="test", model="MiniMax-M2.7")

        # 模拟短输入（1000字符 ≈ 250 tokens）
        max_tokens = llm._calculate_max_tokens("test", "x" * 1000)

        # 短输入应该预留较大输出空间
        assert max_tokens >= 4000, "短输入应该预留足够的输出空间"

    def test_calculate_max_tokens_long_input(self):
        """测试长输入时 max_tokens 计算"""
        llm = MiniMaxLLM(api_key="test", model="MiniMax-M2.7")

        # 模拟长输入（50000字符 ≈ 12500 tokens）
        long_input = "x" * 50000
        max_tokens = llm._calculate_max_tokens("system prompt", long_input)

        # 长输入时输出空间会被压缩，但不应该太小
        assert max_tokens >= 1500, "即使长输入也应该有最小输出空间"

    def test_calculate_max_tokens_very_long_input(self):
        """测试超长输入时 max_tokens 计算"""
        llm = MiniMaxLLM(api_key="test", model="MiniMax-M2.7")

        # 模拟超长输入（80000字符）
        very_long_input = "x" * 80000
        max_tokens = llm._calculate_max_tokens("system", very_long_input)

        # 应该能处理，不会返回 0
        assert max_tokens > 0, "超长输入也应该返回有效值"


if __name__ == "__main__":
    # 运行测试
    tests = [
        test_chat_auto_exists,
        test_chat_auto_accepts_system_and_user,
        test_estimate_tokens,
        test_calculate_max_tokens_short_input,
        test_calculate_max_tokens_long_input,
        test_calculate_max_tokens_very_long_input,
    ]

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
"""MiniMax LLM 调用封装"""
import json
import logging
import os
import re
from typing import Optional, Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict


def parse_json_response(content: str) -> dict[str, Any] | None:
    """
    从 LLM 响应中解析 JSON。

    策略：
    1. 直接尝试 json.loads 解析完整内容（如果内容本身就是 JSON）
    2. 使用正则提取第一个 JSON 对象
    3. 清理常见的 markdown 代码块格式
    """
    if not content or not content.strip():
        return None

    # 策略1：直接解析（处理完整响应本身就是 JSON 的情况）
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 策略2：清理 markdown 代码块格式
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # 移除 ```json 或 ``` 等代码块标记
        lines = cleaned.split("\n")
        # 跳过第一行（```json 或 ```）
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # 移除最后一行（```）
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # 策略3：提取 JSON 对象或数组
    json_patterns = [
        # 匹配 {...} 或 [...] 包裹的 JSON
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 嵌套支持
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    # 策略4：尝试更宽松的匹配（取最后一个 { 开始的子串）
    last_brace = content.rfind("{")
    last_bracket = content.rfind("[")
    start = max(last_brace, last_bracket)
    if start > 0:
        try:
            return json.loads(content[start:])
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法解析 JSON 响应: {content[:100]}...")
    return None


class MiniMaxLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.chat",
        model: str = "MiniMax-M2.7",
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system: str, user: str, timeout: float = 60.0) -> LLMResponse:
        """发送对话请求"""
        url = f"{self.base_url}/v1/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "choices" not in data:
            raise ValueError(f"MiniMax API error: {data.get('base_resp', {}).get('status_msg', 'Unknown error')}")
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            usage=data.get("usage", {}),
        )

    def chat_with_timeout(self, system: str, user: str, timeout: float = 180.0) -> LLMResponse:
        """发送对话请求（带超时参数）"""
        return self.chat(system, user, timeout=timeout)

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数量。

        简单估算：中文约 1 token/字符，英文约 1 token/4 字符。
        实际上中文占比越高，token 数越接近字符数。
        """
        if not text:
            return 0
        # 中文和特殊字符较多时，token 约等于字符数
        # 纯英文时约为字符数/4
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        other_chars = len(text) - chinese_chars
        return chinese_chars + other_chars // 4

    def _calculate_max_tokens(self, system: str, user: str) -> int:
        """
        根据输入长度动态计算 max_tokens。

        假设 MiniMax 上下文窗口为 200k tokens。
        """
        CONTEXT_LIMIT = 200000

        # 估算输入 tokens
        input_tokens = self._estimate_tokens(system) + self._estimate_tokens(user)

        # 计算可用输出 tokens
        available = CONTEXT_LIMIT - input_tokens

        # 设置最小和最大输出空间
        min_output = 1500
        max_output = self.max_tokens  # 不超过默认值

        if available <= 0:
            # 输入太长，返回最小值（API 可能会拒绝）
            return min_output
        elif available >= max_output:
            # 输入短，优先保证输出空间
            return max_output
        else:
            # 动态分配
            return max(min_output, min(available, max_output))

    def chat_auto(self, system: str, user: str, timeout: float = 180.0) -> LLMResponse:
        """
        发送对话请求（自动调整 max_tokens）。

        根据输入长度动态分配输出空间。
        """
        auto_max_tokens = self._calculate_max_tokens(system, user)

        url = f"{self.base_url}/v1/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": auto_max_tokens,
            "temperature": self.temperature,
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or data["choices"] is None:
            raise ValueError(f"MiniMax API error: {data.get('base_resp', {}).get('status_msg', 'Unknown error')}")
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            usage=data.get("usage", {}),
        )
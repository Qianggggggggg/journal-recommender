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
    1. 直接尝试 json.loads 解析完整内容
    2. 清理 markdown 代码块格式，再尝试解析
    3. 从第一个 [ 尝试解析数组
    4. 从第一个 { 尝试解析对象
    """
    if not content or not content.strip():
        return None

    # 策略1：直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 策略2：清理 markdown 代码块格式（处理多行和前后文字）
    cleaned = content.strip()
    # 去掉所有 ```...``` 代码块（可能有多层嵌套）
    import re as re_module
    # 移除 ```json ... ``` 和 ``` ... ``` 格式
    cleaned = re_module.sub(r'```json\s*(.*?)\s*```', r'\1', cleaned, flags=re_module.DOTALL)
    cleaned = re_module.sub(r'```\s*(.*?)\s*```', r'\1', cleaned, flags=re_module.DOTALL)
    cleaned = cleaned.strip()

    # 策略2.5：移除无效的控制字符（0x00-0x08, 0x0b-0x0c, 0x0e-0x1f，但保留 \t\n\r）
    # LLM 响应中的换行符可能被错误编码
    def remove_invalid_controls(s):
        return ''.join(c for c in s if ord(c) >= 0x20 or c in '\t\n\r')
    cleaned = remove_invalid_controls(cleaned)

    # 策略3：尝试解析数组（优先），使用 raw_decode 处理尾部多余文本
    first_bracket = cleaned.find("[")
    if first_bracket >= 0:
        try:
            result, end_idx = json.JSONDecoder().raw_decode(cleaned[first_bracket:])
            # 检查 remainder：如果有实质内容残留（不是单纯的 ] 结尾），说明不是纯数组，是对象内的数组
            remainder = cleaned[first_bracket + end_idx:].strip()
            if remainder and not remainder.startswith("]"):
                # 有实质内容残留，回退到策略4解析对象
                pass
            else:
                return result
        except json.JSONDecodeError:
            pass

    # 策略4：尝试解析对象
    first_brace = cleaned.find("{")
    if first_brace >= 0:
        try:
            result, _ = json.JSONDecoder().raw_decode(cleaned[first_brace:])
            # 兼容处理：如果是包装格式 {"rankings": [...]}，提取内层数组
            if isinstance(result, dict):
                for key in ["rankings", "results", "items", "papers", "journals"]:
                    if key in result and isinstance(result[key], list):
                        inner = result[key]
                        if len(inner) > 0:
                            return inner
            return result
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
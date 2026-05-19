"""MiniMax LLM 调用封装"""
import os
from typing import Optional

import httpx
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict


class MiniMaxLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.chat",
        model: str = "MiniMax-Text-01",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system: str, user: str) -> LLMResponse:
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
        response = httpx.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "choices" not in data:
            raise ValueError(f"MiniMax API error: {data.get('base_resp', {}).get('status_msg', 'Unknown error')}")
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            usage=data.get("usage", {}),
        )
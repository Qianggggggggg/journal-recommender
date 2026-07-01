#!/usr/bin/env python3
"""测试LLM调用，直接调用看返回什么"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

import httpx

api_key = os.getenv("MINIMAX_API_KEY")
base_url = "https://api.minimax.chat"
model = "abab6.5s-chat"

# 简单的测试文本
test_text = "This is a test."

# 构建请求
url = f"{base_url}/v1/text/chatcompletion_v2"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": test_text},
    ],
    "max_tokens": 100,
    "temperature": 0.7,
}

print("发送请求到 MiniMax API...")
try:
    response = httpx.post(url, json=payload, headers=headers, timeout=60)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text[:500] if response.text else 'Empty'}")
    data = response.json()
    print(f"解析后的data: {data}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

# 现在测试长文本
long_text = "Hello. " * 1000  # 模拟较长文本

print("\n\n测试较长文本...")
user_content = f"论文内容：\n{long_text}\n\n请提取论文标题。"

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": user_content},
    ],
    "max_tokens": 100,
    "temperature": 0.7,
}

try:
    response = httpx.post(url, json=payload, headers=headers, timeout=60)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text[:1000] if response.text else 'Empty'}")
    data = response.json()
    print(f"解析后的data: {data}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")
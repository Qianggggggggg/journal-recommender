#!/usr/bin/env python3
"""测试LLM调用，用app.yaml里的模型"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv(override=True)

import httpx

api_key = os.getenv("MINIMAX_API_KEY")

# 加载配置
with open("configs/app.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model = config["minimax"]["model"]
base_url = config["minimax"]["base_url"]

print(f"使用模型: {model}")
print(f"API: {base_url}")

# 构建请求
url = f"{base_url}/v1/text/chatcompletion_v2"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# 测试1：简单文本
print("\n测试简单文本...")
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "Hello, how are you?"},
    ],
    "max_tokens": 50,
    "temperature": 0.7,
}

try:
    response = httpx.post(url, json=payload, headers=headers, timeout=60)
    data = response.json()
    if data.get("choices"):
        print(f"成功! 响应: {data['choices'][0]['message']['content']}")
    else:
        print(f"失败: {data}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")

# 测试2：用真实PDF文本
from src.utils.file_parser import extract_text_from_file, clean_text

pdf_path = "/Users/qian/Documents/论文/1706.03762v7.pdf"
with open(pdf_path, "rb") as f:
    content = f.read()

text = extract_text_from_file(content, pdf_path)
cleaned_text = clean_text(text)

print(f"\nPDF文本长度: {len(cleaned_text)} 字符")

# 截断到2万字符测试
test_text = cleaned_text[:20000]

print("\n测试较长PDF文本（20000字符）...")
user_content = f"请用一句话总结这篇论文的主要内容。\n\n论文内容：\n{test_text}"

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
    response = httpx.post(url, json=payload, headers=headers, timeout=120)
    data = response.json()
    if data.get("choices"):
        print(f"成功! 响应: {data['choices'][0]['message']['content'][:200]}")
    else:
        print(f"失败: {data}")
except Exception as e:
    print(f"错误: {type(e).__name__}: {e}")
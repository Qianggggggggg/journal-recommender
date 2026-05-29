#!/usr/bin/env python3
"""
诊断脚本：测试 API 端点的实际响应
"""
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API 配置
API_BASE = "http://localhost:8000"

def test_recommend_full_text():
    """测试全文模式推荐"""
    # 这篇 PDF 的信息
    test_payload = {
        "title": "Attention Is All You Need",
        "abstract": "",
        "full_text": "x" * 30000,  # 模拟长文本
        "mode": "full",
        "top_k": 3,
        "oa_preference": "any"
    }

    print("测试全文模式 API...")
    print(f"输入长度: {len(test_payload['full_text'])} 字符")

    try:
        response = httpx.post(
            f"{API_BASE}/api/recommend/stream",
            json=test_payload,
            timeout=120
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}")
    except Exception as e:
        print(f"请求失败: {e}")


def test_direct_llm():
    """直接测试 LLM（绕过 API）"""
    import yaml
    from src.utils.file_parser import extract_text_from_file, clean_text
    from src.utils.llm import MiniMaxLLM

    with open("configs/app.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.getenv("MINIMAX_API_KEY")
    llm = MiniMaxLLM(api_key=api_key, model=config["minimax"]["model"])

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 读取 PDF
    pdf_path = "/Users/qian/Documents/论文/1706.03762v7.pdf"
    with open(pdf_path, "rb") as f:
        content = f.read()
    text = clean_text(extract_text_from_file(content, pdf_path))

    print(f"\n直接测试 LLM...")
    print(f"PDF 文本长度: {len(text)} 字符")

    system = prompts["paper_profile_system"]
    user = prompts["paper_profile_user"].format(
        title="Attention Is All You Need",
        abstract="",
        full_text_summary=text
    )

    print(f"auto_max_tokens: {llm._calculate_max_tokens(system, user)}")

    try:
        response = llm.chat_auto(system, user, timeout=180)
        print(f"成功! 响应长度: {len(response.content)}")
    except Exception as e:
        print(f"失败: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("诊断: context window exceeds limit")
    print("=" * 50)

    test_direct_llm()

    print("\n" + "=" * 50)
    test_recommend_full_text()
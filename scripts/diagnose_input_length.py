#!/usr/bin/env python3
"""诊断：测试实际API调用时的输入长度"""
import os
from dotenv import load_dotenv
import yaml

load_dotenv(override=True)

with open('configs/app.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('configs/prompts.yaml', 'r', encoding='utf-8') as f:
    prompts = yaml.safe_load(f)

from src.utils.file_parser import extract_text_from_file, clean_text
from src.utils.llm import MiniMaxLLM

llm = MiniMaxLLM(api_key=os.getenv('MINIMAX_API_KEY'), model=config['minimax']['model'])

# 模拟 API 入口的完整流程
# 这是 PaperParser.parse() 被调用时的实际情况

title = "Attention Is All You Need"
abstract = ""
full_text_summary = ""  # 实际会填入 PDF 内容

# 读取 PDF
pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
with open(pdf_path, 'rb') as f:
    content = f.read()

extracted_text = extract_text_from_file(content, pdf_path)
cleaned_text = clean_text(extracted_text)

print(f"PDF 提取后长度: {len(cleaned_text)} 字符")

# 填充 user prompt（模拟 PaperParser 的行为）
# user_prompt.format(title=..., abstract=..., full_text_summary=...)
user_filled = prompts['paper_profile_user'].format(
    title=title,
    abstract=abstract,
    full_text_summary=cleaned_text
)

system = prompts['paper_profile_system']

print(f"system prompt 长度: {len(system)} 字符")
print(f"user prompt 长度: {len(user_filled)} 字符")
print(f"总长度: {len(system) + len(user_filled)} 字符")
print(f"system tokens: {llm._estimate_tokens(system)}")
print(f"user tokens: {llm._estimate_tokens(user_filled)}")
print(f"总 tokens: {llm._estimate_tokens(system + user_filled)}")

# 计算 chat_auto 里的截断逻辑
CONTEXT_LIMIT = 100000  # 这是代码里的值（即使你已经改成200000，uvicorn可能用的还是旧值）
input_tokens = llm._estimate_tokens(system) + llm._estimate_tokens(user_filled)
print(f"\n=== chat_auto 截断逻辑 ===")
print(f"CONTEXT_LIMIT: {CONTEXT_LIMIT}")
print(f"input_tokens: {input_tokens}")
print(f"need truncation: {input_tokens > CONTEXT_LIMIT}")

if input_tokens > CONTEXT_LIMIT:
    system_tokens = llm._estimate_tokens(system)
    max_user_tokens = CONTEXT_LIMIT - system_tokens - 1000
    max_user_chars = int(max_user_tokens * 4)
    print(f"会截断到: {max_user_chars} 字符")
    print(f"实际 user 长度: {len(user_filled)} 字符")
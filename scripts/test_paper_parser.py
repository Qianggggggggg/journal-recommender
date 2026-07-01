#!/usr/bin/env python3
"""测试PaperParser解析这篇PDF"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)

from src.utils.file_parser import extract_text_from_file, clean_text
from src.papers.paper_model import PaperInput
from src.papers.paper_parser import PaperParser
from src.utils.llm import MiniMaxLLM
from src.utils.text import clean_text as clean_text_func
import yaml

# PDF 文件路径
pdf_path = "/Users/qian/Documents/论文/1706.03762v7.pdf"

# 读取PDF
with open(pdf_path, "rb") as f:
    content = f.read()

# 提取并清理文本
text = extract_text_from_file(content, pdf_path)
cleaned_text = clean_text(text)

print(f"PDF文本长度: {len(cleaned_text)} 字符")

# 构建PaperInput
paper_input = PaperInput(
    title="Attention Is All You Need",
    abstract="",  # 摘要模式不用abstract
    full_text=cleaned_text,
    mode="full"
)

# 加载prompt
with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

# 初始化LLM
api_key = os.getenv("MINIMAX_API_KEY")
if not api_key:
    print("错误: MINIMAX_API_KEY 未配置")
    sys.exit(1)

from src.utils.llm import MiniMaxLLM

llm = MiniMaxLLM(
    api_key=api_key,
    base_url="https://api.minimax.chat",
    model="MiniMax-M2.7",
)

# 测试PaperParser
print("\n开始解析...")
try:
    parser = PaperParser(llm)
    profile = parser.parse(
        paper_input,
        prompts["paper_profile_system"],
        prompts["paper_profile_user"]
    )
    print(f"解析成功!")
    print(f"标题: {profile.title}")
    print(f"研究领域: {profile.research_area}")
    print(f"方法类型: {profile.method_type}")
except Exception as e:
    print(f"解析失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
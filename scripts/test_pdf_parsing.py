#!/usr/bin/env python3
"""测试PDF解析和文本长度"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_parser import extract_text_from_file, clean_text

# PDF 文件路径
pdf_path = "/Users/qian/Documents/论文/1706.03762v7.pdf"

print(f"读取文件: {pdf_path}")

with open(pdf_path, "rb") as f:
    content = f.read()

print(f"文件大小: {len(content)} bytes")

# 提取文本
print("\n提取文本...")
text = extract_text_from_file(content, pdf_path)
print(f"提取后文本长度: {len(text)} 字符")

# 清理文本
print("\n清理文本...")
cleaned = clean_text(text, max_length=50000)
print(f"清理后文本长度: {len(cleaned)} 字符")

# 显示前500字符
print("\n前500字符:")
print(cleaned[:500])

# 显示后500字符
print("\n后500字符:")
print(cleaned[-500:])

# 检查是否有异常内容
print(f"\n文本中换行符数量: {text.count(chr(10))}")
print(f"文本中空格数量: {text.count(' ')}")

# 粗略估算token数量（1 token ≈ 4 字符）
estimated_tokens = len(cleaned) // 4
print(f"\n粗略估算token数量: {estimated_tokens}")
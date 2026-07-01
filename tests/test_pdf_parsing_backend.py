#!/usr/bin/env python3
"""测试后端 PDF 解析功能"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.file_parser import extract_text_from_file, clean_text

# 测试 PDF 解析
pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'

print(f"读取文件: {pdf_path}")

with open(pdf_path, 'rb') as f:
    content = f.read()

print(f"文件大小: {len(content)} bytes")

# 提取文本
text = extract_text_from_file(content, pdf_path)
print(f"提取文本长度: {len(text)} 字符")

# 清理文本
cleaned = clean_text(text, max_length=50000)
print(f"清理后长度: {len(cleaned)} 字符")

# 验证提取质量
if len(cleaned) > 1000:
    print("✓ PDF 解析成功，文本内容充足")
    print(f"前100字符: {cleaned[:100]}")
else:
    print("✗ PDF 解析可能有问题，内容太短")
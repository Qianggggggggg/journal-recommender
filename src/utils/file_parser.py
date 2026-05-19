"""文件解析工具"""
import re
from pathlib import Path


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """从文件内容提取文本"""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_content)
    elif ext in [".txt", ".md"]:
        return extract_text_from_text(file_content)
    else:
        # 尝试作为文本处理
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return file_content.decode("latin-1", errors="replace")


def extract_text_from_pdf(file_content: bytes) -> str:
    """从 PDF 提取文本"""
    try:
        from PyPDF2 import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(file_content))
        text_parts = []

        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception:
                continue

        return "\n".join(text_parts)
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def extract_text_from_text(file_content: bytes) -> str:
    """从文本文件提取"""
    # 尝试 UTF-8
    try:
        return file_content.decode("utf-8")
    except UnicodeDecodeError:
        # 尝试 GBK（中文编码）
        try:
            return file_content.decode("gbk")
        except UnicodeDecodeError:
            # 回退到 latin-1
            return file_content.decode("latin-1", errors="replace")


def clean_text(text: str, max_length: int = 50000) -> str:
    """清理文本"""
    if not text:
        return ""

    # 移除多余空白
    text = re.sub(r"\s+", " ", text)

    # 移除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?()\[\]（）【】]", "", text)

    # 截断
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()
"""文件解析工具"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LayoutBlock:
    """Layout-aware text block from PyMuPDF"""
    text: str
    font_size: float
    font_name: str
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float
    page: int



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
        logger.error(f"PDF extraction failed: {e}")
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


def extract_layout_blocks(file_content: bytes, filename: str) -> Tuple[List[LayoutBlock], str]:
    """
    使用 PyMuPDF 提取 layout-aware blocks。

    返回: (blocks, full_text)
    - blocks: 按阅读顺序排列的 LayoutBlock 列表
    - full_text: 简单拼接的全文（向后兼容）
    """
    import fitz  # PyMuPDF

    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        return [], extract_text_from_file(file_content, filename)

    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            blocks = []
            full_text_parts = []

            for page_num, page in enumerate(doc):
                # 使用 dict 模式获取 blocks（关键区别于 PyPDF2）
                blocks_data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

                for block in blocks_data.get("blocks", []):
                    if block.get("type") != 0:  # 只处理文本块
                        continue

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue

                            # 提取字体信息
                            font = span.get("font", "")
                            size = span.get("size", 0)
                            bold = "bold" in font.lower() or "heavy" in font.lower()

                            # 坐标
                            bbox = span.get("bbox", [0, 0, 0, 0])

                            layout_block = LayoutBlock(
                                text=text,
                                font_size=size,
                                font_name=font,
                                bold=bold,
                                x0=bbox[0],
                                y0=bbox[1],
                                x1=bbox[2],
                                y1=bbox[3],
                                page=page_num + 1,
                            )
                            blocks.append(layout_block)
                            full_text_parts.append(text)

            # 按阅读顺序排序（按 page, y, x）
            blocks.sort(key=lambda b: (b.page, b.y0, b.x0))

            return blocks, " ".join(full_text_parts)

    except Exception as e:
        logger.error(f"Layout extraction failed: {e}")
        return [], ""
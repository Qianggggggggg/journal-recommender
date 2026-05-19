"""文本处理工具"""
import re
from typing import List


def truncate_text(text: str, max_length: int = 2000) -> str:
    """截断文本到最大长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白和特殊字符"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """简单关键词提取（基于频率）"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    from collections import Counter
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_k)]


def split_sentences(text: str) -> List[str]:
    """分句"""
    sentences = re.split(r'[。！？\n]+', text)
    return [s.strip() for s in sentences if s.strip()]
"""SCImago 数据采集脚本"""
import json
import httpx
from typing import List

from src.journals.journal_model import Journal


def crawl_scimago(max_results: int = 500) -> List[Journal]:
    """从 SCImago 采集期刊数据（简化版）"""
    # SCImago 提供 Journal Rankings，可以按学科筛选
    # 这里使用简化版：直接返回空列表，实际使用需要 CSV 导入
    print("SCImago crawler: Use CSV import for production")
    return []


if __name__ == "__main__":
    journals = crawl_scimago()
    print(f"Crawled {len(journals)} journals from SCImago")
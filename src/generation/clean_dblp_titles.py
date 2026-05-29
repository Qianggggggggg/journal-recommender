#!/usr/bin/env python3
"""清洗 DBLP 标题"""
import json
import logging
import re
from pathlib import Path

NON_ENGLISH = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")
FILTER_KEYWORDS = ["special issue", "book review", "editorial", "erratum", "correction", "retraction", "obituary"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_valid_title(title: str) -> bool:
    t = title.lower()
    for kw in FILTER_KEYWORDS:
        if kw in t:
            return False
    if not re.search(r"[a-zA-Z]", title):
        return False
    if NON_ENGLISH.search(title):
        return False
    return True


def clean_titles(titles: list[str]) -> list[str]:
    cleaned = []
    for t in titles:
        t = re.sub(r"^[.,;:]+|[.,;:]+$", "", t.strip())
        if t and is_valid_title(t):
            cleaned.append(t)
    return cleaned


def main():
    src_dir = Path("data/dblp_titles")
    for fp in sorted(src_dir.glob("*.json")):
        if fp.name == "summary.json":
            continue
        with open(fp) as f:
            data = json.load(f)
        orig = len(data["titles"])
        data["titles"] = clean_titles(data["titles"])
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if orig != len(data["titles"]):
            logger.info(f"{data['journal_id']}: {orig} -> {len(data['titles'])}")


if __name__ == "__main__":
    main()
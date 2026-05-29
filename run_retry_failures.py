#!/usr/bin/env python3
"""重试生成失败的典型摘要"""
import json
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.generation.typical_abstract_generator import TypicalAbstractGenerator, METHOD_TYPES, NOVELTY_LEVELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["MINIMAX_API_KEY"] = "sk-cp-8nyHj4lHEq2XSWxvxwTaIWWzO9m2w7ihjzqiX4G3EOQHzrvuOxMM8ly_mIyGBSD4QJqeuz_dgA0GceWreKMcDvXtDDkqu4P-tK7Odf9r55jcIHjndPQ36eM"

# 加载期刊
journals = []
with open("data/processed/journals.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        journals.append(json.loads(line))

# 找出失败的期刊
failed_ids = ["iets", "jiis", "mscs", "cviu", "appliedintelligence", "ijprai", "www"]
failed_journals = [j for j in journals if j["journal_id"] in failed_ids]

logger.info(f"Found {len(failed_journals)} failed journals to retry")

# 初始化生成器
generator = TypicalAbstractGenerator()

def generate_for_journal(journal):
    """为单本期刊生成典型摘要"""
    jid = journal["journal_id"]
    abstracts = generator.generate_for_journal(
        journal_id=jid,
        journal_name=journal.get("journal_name", ""),
        scope_text=journal.get("scope_text", ""),
        keywords=journal.get("keywords", []),
        ccf_rating=journal.get("ccf_rating", ""),
        method_types=METHOD_TYPES[:2],
        novelty_levels=NOVELTY_LEVELS[:2],
    )
    return jid, abstracts, journal

# 并行重试（5 workers）
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(generate_for_journal, j): j for j in failed_journals}

    for i, future in enumerate(as_completed(futures)):
        journal = futures[future]
        try:
            jid, abstracts, journal_data = future.result()

            # 保存结果
            output_file = Path("data/typical_abstracts") / f"{jid}.json"
            data = {
                "journal_id": jid,
                "journal_name": journal_data.get("journal_name", ""),
                "ccf_rating": journal_data.get("ccf_rating", ""),
                "abstracts": [
                    {
                        "method_type": a.method_type,
                        "novelty_level": a.novelty_level,
                        "abstract": a.abstract,
                    }
                    for a in abstracts
                ],
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            non_empty = sum(1 for a in abstracts if a.abstract)
            logger.info(f"[{i+1}/{len(failed_journals)}] {jid}: {non_empty}/{len(abstracts)} non-empty")

        except Exception as e:
            jid = journal.get("journal_id", "UNKNOWN")
            logger.error(f"[{i+1}/{len(failed_journals)}] {jid}: Error: {e}")

logger.info("\nRetry complete!")

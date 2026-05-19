"""期刊数据标准化脚本"""
import json
from pathlib import Path
from typing import List

from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore


def normalize_journals(input_files: List[str], output_file: str) -> int:
    """合并并标准化期刊数据"""
    all_journals = []
    seen_ids = set()

    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    journal = Journal(**data)
                    # 去重
                    if journal.journal_id not in seen_ids:
                        journal.build_profile()
                        all_journals.append(journal)
                        seen_ids.add(journal.journal_id)
                except Exception as e:
                    print(f"Error parsing line: {e}")

    # 保存
    store = JournalStore(store_path=output_file)
    store.add_journals(all_journals)
    store.save()

    print(f"Normalized {len(all_journals)} journals")
    return len(all_journals)


if __name__ == "__main__":
    import sys
    files = ["data/raw/doaj/journals.jsonl", "data/raw/scimago/journals.jsonl"]
    output = "data/processed/journals.jsonl"
    normalize_journals(files, output)
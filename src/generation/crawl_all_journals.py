#!/usr/bin/env python3
"""批量爬取所有期刊的 DBLP 论文标题"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.generation.dblp_crawler import DBLPCrawler
from src.generation.dblp_url_resolver import DBLPURLResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    base_dir = Path("data/dblp_titles")
    base_dir.mkdir(exist_ok=True)

    resolver = DBLPURLResolver(
        journals_path="data/processed/journals.jsonl",
        backup_path="data/processed/journals_backup.jsonl",
    )

    crawler = DBLPCrawler(timeout=30, max_retries=3, delay=1.0)

    success_count = 0
    fail_count = 0
    no_url_count = 0

    futures = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for jid, j in resolver.current_journals.items():
            journal_name = j.get("journal_name", "")
            url = resolver.get_url(jid, journal_name)
            if not url:
                logger.warning(f"No DBLP URL for {jid}: {journal_name}")
                no_url_count += 1
                continue
            future = executor.submit(crawler.crawl_journal, jid, url)
            futures[future] = (jid, url)

        for future in as_completed(futures):
            jid, url = futures[future]
            try:
                result = future.result()
                output_path = base_dir / f"{jid}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "journal_id": result.journal_id,
                        "dblp_url": result.dblp_url,
                        "titles": result.titles,
                    }, f, ensure_ascii=False, indent=2)

                if result.titles:
                    success_count += 1
                    logger.info(f"OK {jid}: {len(result.titles)} titles")
                else:
                    fail_count += 1
                    logger.warning(f"EMPTY {jid}: no titles")

            except Exception as e:
                fail_count += 1
                logger.error(f"FAIL {jid}: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Summary: {success_count} succeeded, {fail_count} failed, {no_url_count} no URL")

    # Generate summary
    summary = []
    for fp in base_dir.glob("*.json"):
        if fp.name == "summary.json":
            continue
        with open(fp) as f:
            d = json.load(f)
        summary.append({
            "journal_id": d["journal_id"],
            "title_count": len(d["titles"]),
            "dblp_url": d["dblp_url"],
        })
    summary.sort(key=lambda x: x["title_count"], reverse=True)

    with open(base_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cov = sum(1 for s in summary if s["title_count"] >= 30)
    logger.info(f"Coverage (>30 titles): {cov}/{len(summary)} ({cov/len(summary)*100:.1f}%)")
    logger.info(f"Data saved to: {base_dir}")


if __name__ == "__main__":
    main()
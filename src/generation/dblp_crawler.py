"""DBLP 标题爬虫"""
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class DBLPJournalPapers:
    """一本期刊的论文数据"""
    journal_id: str
    titles: list[str]
    dblp_url: str
    year_range: tuple[int, int] = (2020, 2024)


class DBLPCrawler:
    """DBLP 期刊论文标题爬虫"""

    DBLP_JOURNAL_URL_TEMPLATE = "http://dblp.uni-trier.de/db/journals/{abbr}/"

    # Filter patterns for non-research entries
    FILTER_PATTERNS = [
        re.compile(r"special\s*issue", re.I),
        re.compile(r"book\s*review", re.I),
        re.compile(r"editorial", re.I),
        re.compile(r"Erratum", re.I),
        re.compile(r"Correction", re.I),
        re.compile(r"Obituary", re.I),
        re.compile(r"Retraction", re.I),
    ]

    # Non-English character patterns
    NON_ENGLISH_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")

    def __init__(self, timeout: int = 30, max_retries: int = 3, delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; JournalRecommenderBot/1.0; research project)",
            "Accept": "text/html,application/xhtml+xml",
        })

    def is_research_title(self, title: str) -> bool:
        """判断是否为有效研究论文标题"""
        for pattern in self.FILTER_PATTERNS:
            if pattern.search(title):
                return False
        if self.NON_ENGLISH_PATTERN.search(title):
            return False
        return bool(re.search(r"[a-zA-Z]", title))

    def fetch_journal_page(self, url: str) -> Optional[str]:
        """爬取期刊页面 HTML"""
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        return None

    def parse_titles_from_html(self, html: str, year_range: tuple[int, int]) -> list[str]:
        """从 HTML 解析论文标题"""
        soup = BeautifulSoup(html, "lxml")
        titles = []

        for item in soup.find_all("li", class_=["article", "inproceedings"]):
            year_elem = item.find("span", class_="year")
            if not year_elem:
                continue
            try:
                year = int(year_elem.get_text().strip())
            except ValueError:
                continue

            if not (year_range[0] <= year <= year_range[1]):
                continue

            title_elem = item.find("span", class_="title")
            if title_elem:
                title = title_elem.get_text().strip().strip(".,;:")
                if title and self.is_research_title(title):
                    titles.append(title)

        return titles

    def crawl_journal(self, journal_id: str, url: str, year_range: tuple[int, int] = (2020, 2024)) -> DBLPJournalPapers:
        """爬取单本期刊的论文标题"""
        html = self.fetch_journal_page(url)
        if not html:
            return DBLPJournalPapers(journal_id=journal_id, titles=[], dblp_url=url)
        titles = self.parse_titles_from_html(html, year_range)
        return DBLPJournalPapers(journal_id=journal_id, titles=titles, dblp_url=url)
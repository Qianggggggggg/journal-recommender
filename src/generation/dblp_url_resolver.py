"""DBLP URL 解析器：将 journal_id 映射到 DBLP URL"""
import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class DBLPURLResolver:
    """解析期刊到 DBLP URL 的映射"""

    def __init__(self, journals_path: str, backup_path: str):
        self.journals_path = journals_path
        self.backup_path = backup_path
        self._load_data()

    def _load_data(self):
        """加载并匹配期刊数据"""
        self.current_journals = {}
        with open(self.journals_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.current_journals[j["journal_id"]] = j

        self.backup_dblp = {}
        with open(self.backup_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                url = j.get("submission_url", "")
                if url and "dblp" in url.lower():
                    self.backup_dblp[j["journal_id"]] = url

        self.matched_urls = {}
        for jid in self.current_journals:
            if jid in self.backup_dblp:
                self.matched_urls[jid] = self.backup_dblp[jid]

        logger.info(f"Matched DBLP URLs: {len(self.matched_urls)}/{len(self.current_journals)}")

    def construct_url(self, journal_id: str) -> Optional[str]:
        """从 journal_id 构造 DBLP URL"""
        if journal_id:
            return f"http://dblp.uni-trier.de/db/journals/{journal_id}/"
        return None

    def search_by_name(self, journal_name: str) -> Optional[str]:
        """通过 DBLP 搜索 API 按期刊名查找"""
        params = {"q": journal_name, "format": "json", "h": 5}
        try:
            resp = requests.get("https://dblp.org/search/publ/api", params=params, timeout=15)
            data = resp.json()
            for hit in data.get("result", {}).get("hits", {}).get("hit", []):
                info = hit.get("info", {})
                if "journal" in info.get("type", "").lower():
                    return info.get("url", "")
        except Exception as e:
            logger.warning(f"DBLP search failed: {e}")
        return None

    def get_url(self, journal_id: str, journal_name: str) -> Optional[str]:
        """获取期刊的 DBLP URL"""
        # 1. 已匹配的 backup
        if journal_id in self.matched_urls:
            return self.matched_urls[journal_id]
        # 2. 从 journal_id 构造
        url = self.construct_url(journal_id)
        if url:
            return url
        # 3. 搜索备用
        return self.search_by_name(journal_name)
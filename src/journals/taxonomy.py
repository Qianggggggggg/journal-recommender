"""期刊分类体系"""
from typing import Dict, List
import yaml
from pathlib import Path


class JournalTaxonomy:
    """期刊分类体系"""

    def __init__(self, config_path: str = "configs/journal_taxonomy.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @property
    def subject_tags(self) -> List[Dict]:
        return self.config.get("subject_tags", [])

    @property
    def method_types(self) -> List[Dict]:
        return self.config.get("method_types", [])

    @property
    def paper_types(self) -> List[Dict]:
        return self.config.get("paper_types", [])

    @property
    def oa_types(self) -> List[Dict]:
        return self.config.get("oa_types", [])

    def get_subject_keywords(self, tag_id: str) -> List[str]:
        """获取学科标签的关键词"""
        for tag in self.subject_tags:
            if tag["id"] == tag_id:
                return tag.get("keywords", [])
        return []

    def match_subject_tag(self, text: str) -> List[str]:
        """根据文本匹配学科标签"""
        matched = []
        text_lower = text.lower()
        for tag in self.subject_tags:
            for keyword in tag.get("keywords", []):
                if keyword.lower() in text_lower:
                    matched.append(tag["id"])
                    break
        return list(set(matched))
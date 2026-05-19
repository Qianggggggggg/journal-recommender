"""混合召回"""
from typing import Dict, List, Optional, Tuple

from ..journals.journal_model import Journal, JournalMatch
from ..journals.journal_store import JournalStore
from ..papers.paper_model import PaperProfile
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever


class CandidateGenerator:
    """候选召回生成器（混合召回）"""

    def __init__(
        self,
        store: JournalStore,
        bm25_retriever: BM25Retriever,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
        merge_weights: Optional[Dict[str, float]] = None,
    ):
        self.store = store
        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever
        self.merge_weights = merge_weights or {"bm25": 0.35, "vector": 0.35, "tag": 0.2, "text": 0.1}

    def generate(
        self,
        query_text: str,
        paper_profile: PaperProfile,
        top_k: int = 40,
        mode: str = "abstract",
    ) -> List[Journal]:
        """生成候选期刊"""
        # 各路召回数量配置
        config = {
            "title": {"bm25": 18, "vector": 18, "tag": 15, "text": 10},
            "abstract": {"bm25": 22, "vector": 22, "tag": 18, "text": 12},
            "full": {"bm25": 25, "vector": 25, "tag": 20, "text": 15},
        }
        cfg = config.get(mode, config["abstract"])

        # 构建丰富的检索 query，包含 paper_profile 的所有关键字段
        rich_query = self._build_rich_query(query_text, paper_profile)

        # 1. BM25 召回
        bm25_results = self.bm25_retriever.retrieve(rich_query, top_k=cfg["bm25"])

        # 2. 向量检索召回
        vector_results = []
        if self.embedding_retriever:
            vector_results = self.embedding_retriever.retrieve(rich_query, top_k=cfg["vector"])

        # 3. 标签过滤
        tag_filtered = self._filter_by_tags(paper_profile, top_k=cfg["tag"])

        # 4. 文本搜索（关键词交集）
        text_results = self._text_search(paper_profile, top_k=cfg["text"])

        # 5. 合并去重
        candidates = self._merge_results(bm25_results, vector_results, tag_filtered, text_results, top_k=top_k)

        return candidates

    def _text_search(self, paper_profile: PaperProfile, top_k: int = 10) -> List[Tuple[Journal, float]]:
        """基于关键词交集的文本搜索"""
        # 构建检索文本：从 paper_profile 的多个字段提取关键词
        query_parts = [paper_profile.title]
        if paper_profile.abstract:
            query_parts.append(paper_profile.abstract)
        if paper_profile.keywords:
            query_parts.extend(paper_profile.keywords)
        if paper_profile.research_area:
            query_parts.extend(paper_profile.research_area)
        if paper_profile.techniques:
            query_parts.extend(paper_profile.techniques)

        query_text = " ".join(query_parts)
        return self.store.search_by_text(query_text, top_k=top_k)

    def _build_rich_query(self, query_text: str, paper_profile: PaperProfile) -> str:
        """构建丰富的检索 query，整合 paper_profile 的所有关键字段"""
        parts = [query_text]

        if paper_profile.techniques:
            parts.append(" ".join(paper_profile.techniques))
        if paper_profile.datasets:
            parts.append(" ".join(paper_profile.datasets))
        if paper_profile.evaluation_metrics:
            parts.append(" ".join(paper_profile.evaluation_metrics))
        if paper_profile.keywords:
            parts.append(" ".join(paper_profile.keywords))
        if paper_profile.application_domain:
            parts.append(" ".join(paper_profile.application_domain))
        if paper_profile.novelty_type:
            parts.append(paper_profile.novelty_type)

        return " ".join(parts)

    def _filter_by_tags(
        self, paper_profile: PaperProfile, top_k: int = 20
    ) -> List[Tuple[Journal, float]]:
        """标签过滤召回"""
        results = []
        for journal in self.store._journals:
            score = 0.0
            # 研究领域匹配
            if paper_profile.research_area:
                for area in paper_profile.research_area:
                    if area in journal.subject_tags:
                        score += 1.0
            # 应用领域匹配
            if paper_profile.application_domain:
                for domain in paper_profile.application_domain:
                    if domain in journal.subject_tags:
                        score += 0.8
            # 论文类型匹配
            if paper_profile.method_type in journal.target_paper_type:
                score += 0.5
            # 技术匹配（检查是否在 journal.scope_text 中出现）
            if paper_profile.techniques:
                scope_lower = journal.scope_text.lower()
                for tech in paper_profile.techniques:
                    if tech.lower() in scope_lower:
                        score += 0.3
            if score > 0:
                results.append((journal, score))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        bm25_results: List[Tuple[Journal, float]],
        vector_results: List[Tuple[Journal, float]],
        tag_results: List[Tuple[Journal, float]],
        text_results: Optional[List[Tuple[Journal, float]]] = None,
        top_k: int = 50,
    ) -> List[Journal]:
        """合并去重"""
        score_map: Dict[str, float] = {}

        # BM25 结果
        for journal, score in bm25_results:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["bm25"]

        # 向量结果
        for journal, score in vector_results:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["vector"]

        # 标签结果
        for journal, score in tag_results:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["tag"]

        # 文本搜索结果
        if text_results:
            for journal, score in text_results:
                score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["text"]

        # 排序取 top_k
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)[:top_k]

        # 返回 Journal 对象列表
        journal_map = {j.journal_id: j for j in self.store._journals}
        return [journal_map[jid] for jid in sorted_ids if jid in journal_map]
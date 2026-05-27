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
        self.merge_weights = merge_weights or {"bm25": 0.45, "vector": 0.35, "text": 0.20}

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
            "title": {"bm25": 22, "vector": 22, "text": 16},
            "abstract": {"bm25": 28, "vector": 28, "text": 14},
            "full": {"bm25": 32, "vector": 32, "text": 16},
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

        # 3. 文本搜索（关键词交集）
        text_results = self._text_search(paper_profile, top_k=cfg["text"])

        # 4. 合并去重
        candidates = self._merge_results(bm25_results, vector_results, text_results, top_k=top_k)

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
        parts = [
            query_text,  # 论文标题+摘要
            query_text,  # 重复一次以提升权重
            " ".join(paper_profile.keywords),
            " ".join(paper_profile.techniques),
            " ".join(paper_profile.application_domain),
            " ".join(paper_profile.keywords),      # 再次重复
            " ".join(paper_profile.techniques),    # 再次重复
            " ".join(paper_profile.datasets),
            " ".join(paper_profile.evaluation_metrics),
            paper_profile.novelty_type or "",
        ]
        return " ".join(parts)

    def _normalize_scores(self, results: List[Tuple[Journal, float]]) -> List[Tuple[Journal, float]]:
        """对单路结果做 min-max 归一化到 [0,1]"""
        if not results:
            return []

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # 所有分数相同，归一化为 0
            return [(journal, 0.0) for journal, _ in results]

        normalized = [(journal, (score - min_score) / (max_score - min_score)) for journal, score in results]
        return normalized

    def _merge_results(
        self,
        bm25_results: List[Tuple[Journal, float]],
        vector_results: List[Tuple[Journal, float]],
        text_results: Optional[List[Tuple[Journal, float]]] = None,
        top_k: int = 50,
    ) -> List[Journal]:
        """合并去重（各路分数归一化后加权合并）"""
        # 先对每路做 min-max 归一化
        bm25_norm = self._normalize_scores(bm25_results)
        vector_norm = self._normalize_scores(vector_results)
        text_norm = self._normalize_scores(text_results) if text_results else []

        score_map: Dict[str, float] = {}

        # BM25 结果（已归一化）
        for journal, score in bm25_norm:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["bm25"]

        # 向量结果（已归一化）
        for journal, score in vector_norm:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["vector"]

        # 文本搜索结果（已归一化）
        for journal, score in text_norm:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * self.merge_weights["text"]

        # 排序取 top_k
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)[:top_k]

        # 返回 Journal 对象列表
        journal_map = {j.journal_id: j for j in self.store._journals}
        return [journal_map[jid] for jid in sorted_ids if jid in journal_map]
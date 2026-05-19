"""计算机类期刊数据库构建脚本

策略：
1. 使用 DOAJ 数据作为 OA 期刊基础
2. 补充知名 CS 期刊数据（IEEE, Elsevier, Springer 等）
3. 手动添加一些没有通过 DOAJ API 返回的高质量期刊
"""
import json
import httpx
from pathlib import Path
from typing import List, Dict, Optional


# 知名 CS 期刊（需要手动补充完整信息）
MANUAL_JOURNALS = [
    {
        "journal_id": "tpami",
        "journal_name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "publisher": "IEEE",
        "subject_tags": ["cv", "ai", "nlp"],
        "keywords": ["pattern analysis", "machine intelligence", "computer vision", "AI", "deep learning", "neural networks"],
        "scope_text": "Coverage includes pattern recognition, machine intelligence, computer vision, natural language processing, AI",
        "oa_type": "subscription",
        "quartile": "Q1",
        "sqr_rank": 3,
        "impact_like_score": 30.0,
        "review_time": "6-12 months",
        "apc": 0,
        "target_paper_type": ["method", "experiment", "survey"],
        "submission_url": "https://mc.manuscriptcentral.com/pami",
        "homepage_url": "https://www.computer.org/web/tpami",
    },
    {
        "journal_id": "ijcv",
        "journal_name": "International Journal of Computer Vision",
        "publisher": "Springer",
        "subject_tags": ["cv", "ai"],
        "keywords": ["computer vision", "image understanding", "visual recognition", "deep learning", "3D reconstruction"],
        "scope_text": "Computer vision, image understanding, visual recognition, deep learning, 3D reconstruction",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 6,
        "impact_like_score": 18.0,
        "review_time": "6-9 months",
        "apc": 3690,
        "target_paper_type": ["method", "experiment"],
        "submission_url": "https://www.editorialmanager.com/ijcv",
        "homepage_url": "https://www.springer.com/journal/11263",
    },
    {
        "journal_id": "tnnls",
        "journal_name": "IEEE Transactions on Neural Networks and Learning Systems",
        "publisher": "IEEE",
        "subject_tags": ["ai", "nlp"],
        "keywords": ["neural networks", "deep learning", "machine learning", "reinforcement learning"],
        "scope_text": "Neural networks, machine learning, deep learning, reinforcement learning, and related areas",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 10,
        "impact_like_score": 15.0,
        "review_time": "3-6 months",
        "apc": 2500,
        "target_paper_type": ["method", "experiment"],
        "submission_url": "https://mc.manuscriptcentral.com/tnnls",
        "homepage_url": "https://ieee-tnnls.org/",
    },
    {
        "journal_id": "information-fusion",
        "journal_name": "Information Fusion",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "cv"],
        "keywords": ["information fusion", "multi-sensor", "pattern recognition", "machine learning"],
        "scope_text": "Information fusion, multi-sensor systems, pattern recognition, machine learning, computer vision",
        "oa_type": "subscription",
        "quartile": "Q1",
        "sqr_rank": 8,
        "impact_like_score": 14.0,
        "review_time": "4-6 months",
        "apc": 0,
        "target_paper_type": ["method", "experiment"],
        "submission_url": "https://www.editorialmanager.com/inffus",
        "homepage_url": "https://www.sciencedirect.com/journal/information-fusion",
    },
    {
        "journal_id": "kbs",
        "journal_name": "Knowledge-Based Systems",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "se"],
        "keywords": ["knowledge-based systems", "expert systems", "knowledge representation", "reasoning", "neural networks"],
        "scope_text": "Knowledge-based systems, expert systems, knowledge representation, reasoning, neural networks",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 20,
        "impact_like_score": 8.0,
        "review_time": "3-5 months",
        "apc": 2400,
        "target_paper_type": ["method", "system", "experiment"],
        "submission_url": "https://www.editorialmanager.com/kbs",
        "homepage_url": "https://www.sciencedirect.com/journal/knowledge-based-systems",
    },
    {
        "journal_id": "ins",
        "journal_name": "Information Sciences",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "nlp", "db"],
        "keywords": ["information sciences", "data mining", "knowledge discovery", "machine learning"],
        "scope_text": "Information sciences, data mining, knowledge discovery, machine learning, natural language processing",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 15,
        "impact_like_score": 8.5,
        "review_time": "3-5 months",
        "apc": 2800,
        "target_paper_type": ["method", "experiment"],
        "submission_url": "https://www.editorialmanager.com/ins",
        "homepage_url": "https://www.sciencedirect.com/journal/information-sciences",
    },
    {
        "journal_id": "pr",
        "journal_name": "Pattern Recognition",
        "publisher": "Elsevier",
        "subject_tags": ["cv", "ai"],
        "keywords": ["pattern recognition", "computer vision", "image processing", "machine learning", "document analysis"],
        "scope_text": "Pattern recognition, computer vision, image processing, machine learning, document analysis",
        "oa_type": "subscription",
        "quartile": "Q1",
        "sqr_rank": 25,
        "impact_like_score": 7.0,
        "review_time": "4-6 months",
        "apc": 0,
        "target_paper_type": ["method", "experiment", "survey"],
        "submission_url": "https://www.editorialmanager.com/pr",
        "homepage_url": "https://www.sciencedirect.com/journal/pattern-recognition",
    },
    {
        "journal_id": "nn",
        "journal_name": "Neural Networks",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "nlp"],
        "keywords": ["neural networks", "deep learning", "reinforcement learning", "cognitive science", "brain modeling"],
        "scope_text": "Neural networks, deep learning, reinforcement learning, cognitive science, brain modeling",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 12,
        "impact_like_score": 10.0,
        "review_time": "3-5 months",
        "apc": 3000,
        "target_paper_type": ["method", "experiment", "survey"],
        "submission_url": "https://www.editorialmanager.com/nnet",
        "homepage_url": "https://www.sciencedirect.com/journal/neural-networks",
    },
    {
        "journal_id": "jbi",
        "journal_name": "Journal of Biomedical Informatics",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "se"],
        "keywords": ["biomedical informatics", "health informatics", "machine learning", "clinical data"],
        "scope_text": "Biomedical informatics, health informatics, machine learning in medicine, clinical data analysis",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 30,
        "impact_like_score": 5.0,
        "review_time": "3-4 months",
        "apc": 2600,
        "target_paper_type": ["method", "experiment", "system"],
        "submission_url": "https://www.editorialmanager.com/jbi",
        "homepage_url": "https://www.sciencedirect.com/journal/journal-of-biomedical-informatics",
    },
    {
        "journal_id": "expert-sys",
        "journal_name": "Expert Systems",
        "publisher": "Wiley",
        "subject_tags": ["ai", "se"],
        "keywords": ["expert systems", "knowledge-based systems", "intelligent systems", "decision support"],
        "scope_text": "Expert systems, knowledge-based systems, intelligent systems, decision support systems",
        "oa_type": "hybrid",
        "quartile": "Q2",
        "sqr_rank": 45,
        "impact_like_score": 4.0,
        "review_time": "3-5 months",
        "apc": 2200,
        "target_paper_type": ["method", "system", "experiment"],
        "submission_url": "https://onlinelibrary.wiley.com/journal/1099050x",
        "homepage_url": "https://onlinelibrary.wiley.com/journal/expert-systems",
    },
    {
        "journal_id": "tcyb",
        "journal_name": "IEEE Transactions on Cybernetics",
        "publisher": "IEEE",
        "subject_tags": ["ai", "se"],
        "keywords": ["cybernetics", "systems biology", "neural networks", "robotics", "control systems"],
        "scope_text": "Cybernetics, systems biology, neural networks, robotics, intelligent control, optimization",
        "oa_type": "subscription",
        "quartile": "Q1",
        "sqr_rank": 18,
        "impact_like_score": 12.0,
        "review_time": "4-6 months",
        "apc": 0,
        "target_paper_type": ["method", "experiment", "system"],
        "submission_url": "https://mc.manuscriptcentral.com/tcyb",
        "homepage_url": "https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6224437",
    },
    {
        "journal_id": "tsmc-sys",
        "journal_name": "IEEE Transactions on Systems, Man, and Cybernetics: Systems",
        "publisher": "IEEE",
        "subject_tags": ["ai", "se", "network"],
        "keywords": ["systems engineering", "human-machine systems", "cybernetics", "decision making", "complex systems"],
        "scope_text": "Systems engineering, human-machine systems, cybernetics, decision making, complex systems",
        "oa_type": "subscription",
        "quartile": "Q1",
        "sqr_rank": 22,
        "impact_like_score": 9.0,
        "review_time": "4-6 months",
        "apc": 0,
        "target_paper_type": ["method", "system", "experiment"],
        "submission_url": "https://mc.manuscriptcentral.com/tsmc",
        "homepage_url": "https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6224437",
    },
    {
        "journal_id": "access",
        "journal_name": "IEEE Access",
        "publisher": "IEEE",
        "subject_tags": ["ai", "cv", "nlp", "se", "network", "security", "db", "theory"],
        "keywords": ["multidisciplinary", "open access", "all areas of engineering"],
        "scope_text": "Multidisciplinary open access journal covering all areas of engineering and science",
        "oa_type": "full_oa",
        "quartile": "Q1",
        "sqr_rank": 35,
        "impact_like_score": 5.0,
        "review_time": "2-3 months",
        "apc": 2500,
        "target_paper_type": ["method", "system", "experiment", "survey"],
        "submission_url": "https://mc.manuscriptcentral.com/ieeeaccess",
        "homepage_url": "https://ieeeaccess.ieee.org/",
    },
    {
        "journal_id": "neurocomputing",
        "journal_name": "Neurocomputing",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "nlp"],
        "keywords": ["neural networks", "learning systems", "deep learning", "cognitive models", "brain-machine interfaces"],
        "scope_text": "Neural networks, learning systems, deep learning, cognitive models, brain-machine interfaces",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 28,
        "impact_like_score": 6.0,
        "review_time": "3-5 months",
        "apc": 2800,
        "target_paper_type": ["method", "experiment"],
        "submission_url": "https://www.editorialmanager.com/neurocom",
        "homepage_url": "https://www.sciencedirect.com/journal/neurocomputing",
    },
    {
        "journal_id": "cse",
        "journal_name": "Computing",
        "publisher": "Springer",
        "subject_tags": ["se", "db", "theory"],
        "keywords": ["computer science", "software engineering", "algorithms", "distributed computing", "database systems"],
        "scope_text": "Computer science, software engineering, algorithms, distributed computing, database systems",
        "oa_type": "hybrid",
        "quartile": "Q2",
        "sqr_rank": 50,
        "impact_like_score": 3.5,
        "review_time": "4-6 months",
        "apc": 2100,
        "target_paper_type": ["method", "system", "experiment"],
        "submission_url": "https://www.editorialmanager.com/computing",
        "homepage_url": "https://www.springer.com/journal/160",
    },
    {
        "journal_id": "ai-review",
        "journal_name": "Artificial Intelligence Review",
        "publisher": "Springer",
        "subject_tags": ["ai"],
        "keywords": ["artificial intelligence", "machine learning", "deep learning", "neural networks", "AI applications"],
        "scope_text": "Artificial intelligence, machine learning, deep learning, neural networks, AI applications and theory",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 40,
        "impact_like_score": 5.5,
        "review_time": "4-8 months",
        "apc": 2690,
        "target_paper_type": ["survey", "method", "experiment"],
        "submission_url": "https://www.editorialmanager.com/aire",
        "homepage_url": "https://www.springer.com/journal/10462",
    },
    {
        "journal_id": "aicom",
        "journal_name": "Artificial Intelligence and Communication Management",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "nlp", "se"],
        "keywords": ["AI communication", "human-computer interaction", "intelligent systems", "multimedia"],
        "scope_text": "AI communication, human-computer interaction, intelligent multimedia systems",
        "oa_type": "hybrid",
        "quartile": "Q2",
        "sqr_rank": 55,
        "impact_like_score": 3.0,
        "review_time": "3-5 months",
        "apc": 2000,
        "target_paper_type": ["method", "experiment"],
        "submission_url": "https://www.editorialmanager.com/aicom",
        "homepage_url": "https://www.sciencedirect.com/journal/artificial-intelligence-and-communication-management",
    },
    {
        "journal_id": "dss",
        "journal_name": "Decision Support Systems",
        "publisher": "Elsevier",
        "subject_tags": ["ai", "se"],
        "keywords": ["decision support", "expert systems", "knowledge-based systems", "intelligent systems", "analytics"],
        "scope_text": "Decision support systems, expert systems, knowledge-based systems, intelligent decision making, analytics",
        "oa_type": "hybrid",
        "quartile": "Q1",
        "sqr_rank": 38,
        "impact_like_score": 5.5,
        "review_time": "4-6 months",
        "apc": 2800,
        "target_paper_type": ["method", "system", "experiment"],
        "submission_url": "https://www.editorialmanager.com/dss",
        "homepage_url": "https://www.sciencedirect.com/journal/decision-support-systems",
    },
]


def load_doaj_journals(path: str) -> List[Dict]:
    """加载 DOAJ 采集的期刊"""
    journals = []
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                journals.append(j)
    return journals


def enrich_journal_data(journal: Dict) -> Dict:
    """补充期刊数据"""
    # 确保必要的字段存在
    journal.setdefault("quartile", "Q2" if journal.get("oa_type") == "full_oa" else "Q3")
    journal.setdefault("sqr_rank", 100)
    journal.setdefault("impact_like_score", 3.0)
    journal.setdefault("review_time", "3-6 months")
    journal.setdefault("target_paper_type", ["method", "experiment"])
    journal.setdefault("submission_url", journal.get("homepage_url") or "")

    # 处理 apc 字段（可能是 dict 或 number）
    apc = journal.get("apc", 0)
    if isinstance(apc, dict):
        journal["apc"] = apc.get("price", 0)
    journal.setdefault("apc", 0)

    # 如果 scope_text 为空，用 keywords 和 description 组合
    if not journal.get("scope_text"):
        keywords = journal.get("keywords", [])
        if isinstance(keywords, list) and keywords:
            journal["scope_text"] = ", ".join(keywords[:10])

    # 计算 impact_like_score (如果没有的话，用 sqr_rank 估算)
    if not journal.get("impact_like_score") or journal.get("impact_like_score") == 0:
        if journal.get("sqr_rank"):
            journal["impact_like_score"] = max(0.1, 10.0 - (journal["sqr_rank"] / 20))

    return journal


def build_journal_database(doaj_path: str, output_path: str):
    """构建期刊数据库"""
    print("=== Building Journal Database ===")

    # 加载 DOAJ 期刊
    print(f"\n1. Loading DOAJ journals from {doaj_path}...")
    doaj_journals = load_doaj_journals(doaj_path)
    print(f"   Loaded {len(doaj_journals)} DOAJ journals")

    # 添加手动期刊
    print(f"\n2. Adding manual high-quality journals...")
    all_journals = {}

    #  Enrich DOAJ journals and add them
    for j in doaj_journals:
        j = enrich_journal_data(j)
        all_journals[j["journal_id"]] = j

    # Add manual journals
    for j in MANUAL_JOURNALS:
        j = enrich_journal_data(j)
        all_journals[j["journal_id"]] = j
    print(f"   Added {len(MANUAL_JOURNALS)} manual journals")

    # 合并去重
    journals = list(all_journals.values())
    print(f"\n3. Total unique journals: {len(journals)}")

    # 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for j in journals:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"\n4. Saved to {output_path}")

    # 统计
    q1 = sum(1 for j in journals if j.get("quartile") == "Q1")
    q2 = sum(1 for j in journals if j.get("quartile") == "Q2")
    oa = sum(1 for j in journals if j.get("oa_type") == "full_oa")
    print(f"\n5. Stats:")
    print(f"   Q1 journals: {q1}")
    print(f"   Q2 journals: {q2}")
    print(f"   Full OA journals: {oa}")

    return journals


if __name__ == "__main__":
    build_journal_database(
        doaj_path="data/raw/doaj_journals.jsonl",
        output_path="data/processed/journals.jsonl"
    )
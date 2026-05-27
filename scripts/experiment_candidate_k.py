def evaluate_single_paper(paper, pipeline, prompts, mode, coarse_top_k, final_top_k):
    """评估单篇论文（使用指定的粗排 top_k）"""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    venue = paper.get("venue", "")
    arxiv_id = paper.get("external_ids", {}).get("arXiv", "")

    paper_input = PaperInput(
        title=title,
        abstract=abstract or "",
        full_text="",
        mode=mode,
    )

    try:
        profile = pipeline.parser.parse(
            paper_input,
            prompts["paper_profile_system"],
            prompts["paper_profile_user"],
        )
    except Exception as e:
        print(f"\n解析失败: {title[:30]}... - {e}")
        return None

    # 直接调用 candidate_generator 以控制 coarse_top_k
    query_text = paper_input.title
    if paper_input.abstract:
        query_text += " " + paper_input.abstract

    candidates = pipeline.candidate_generator.generate(
        query_text, profile, top_k=coarse_top_k, mode=mode
    )
    candidate_journal_names = [j.journal_name for j in candidates] if candidates else []

    # RuleScorer 排序（使用不同的 top_k 选择）
    rule_ranked = pipeline.rule_scorer.rank(
        candidates, profile, oa_preference="any", top_k=min(coarse_top_k, 20)
    )
    rule_ranked_names = [j.journal_name for j, s, r in rule_ranked] if rule_ranked else []

    # 质量调整
    rule_ranked = pipeline._apply_quality_adjustment(rule_ranked, profile)

    # LLMRanker 精排（使用 top_k=final_top_k）
    try:
        llm_ranked, rank_method = pipeline.llm_ranker.rank(
            rule_ranked, profile, top_k=final_top_k
        )
    except Exception as e:
        # LLM 失败时降级
        llm_ranked = [(j, s, r, 0.5) for j, s, r in rule_ranked[:final_top_k]]

    recommendations = llm_ranked
    recommended_journals = [rec[0].journal_name for rec in recommendations]

    # 标准化 venue 用于比较
    venue_norm = venue.strip().lower() if venue else ""
    candidate_journal_names_norm = [j.lower() for j in candidate_journal_names]
    rule_ranked_names_norm = [j.lower() for j in rule_ranked_names]
    recommended_journals_norm = [j.lower() for j in recommended_journals]

    # 指标计算
    coarse_hit = venue_norm in candidate_journal_names_norm if venue else False
    coarse_hit_in_rule_top10 = venue_norm in rule_ranked_names_norm[:10] if venue else False
    hit_5 = venue_norm in recommended_journals_norm[:5] if venue else False

    return {
        "title": title[:40],
        "venue": venue,
        "coarse_hit": coarse_hit,
        "coarse_hit_in_rule_top10": coarse_hit_in_rule_top10,
        "hit_5": hit_5,
        "candidate_count": len(candidate_journal_names),
        "rule_ranked_names": rule_ranked_names[:10],
    }
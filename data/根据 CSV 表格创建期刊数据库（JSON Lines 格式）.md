# 任务：根据 CSV 表格创建期刊数据库（JSON Lines 格式）

请严格按照以下步骤执行：

1. 读取 CSV 文件：/Users/qian/Downloads/deepseek_csv_20260520_7a4972.csv
   - 该文件包含期刊信息，列依次为：序号、刊物简称、刊物全称、分类、类型、专业领域
   - 注意：类型全部是“期刊”，无需区分会议
   - 必须保留每一行，一个期刊都不能漏

2. 对每个期刊生成一个 JSON 对象，字段如下：
   - journal_id: 使用“刊物简称”字段，转小写，空格替换为下划线（示例：TOCS -> tocs）
   - journal_name: “刊物全称”字段原值
   - publisher: 需要根据期刊全称去官网查找，常见如 ACM / IEEE / Elsevier / Springer 等，若无法确定则标注 "unknown"
   - subject_tags: 数组，内容为“专业领域”字段（原样）
   - ccf_rating: “分类”字段（A/B/C）
   - scope_text: **必须去期刊官网查找真实的 scope / aims & scope**，不得编造。步骤如下：
        a) 根据刊物全称搜索官网（使用搜索引擎或已知 dblp 链接）
        b) 找到 “Aims & scope” 或 “About” 页面，提取 1-3 句真实描述
        c) 如果第一次失败，重试至少 2 次（不同关键词或官网入口）
        d) 实在无法获取到真实 scope 时，**必须明确写 "暂无scope"**，绝不允许自己编造内容
   - submission_url: 该期刊的投稿/官网首页链接（从 dblp 或期刊官网获取）
   - homepage_url: 同上，可与 submission_url 相同
   - keywords: 空数组即可（暂不需要）
   - oa_type: 若已知则填 "subscription" 或 "open_access"，否则 "unknown"
   - impact_like_score: 0.0（暂不评估）
   - review_time: 空字符串
   - apc: 0.0

3. 输出格式：JSON Lines，每行一个期刊的 JSON 对象。保存到 /Users/qian/PycharmProjects/paper/data/journals_output.jsonl

4. 在执行过程中，请用友好的方式告诉我：
   - 总共找到了多少个期刊
   - 哪些期刊的 scope 成功获取，哪些未获取到（标注原因）
   - 遇到的任何明显错误或无法访问的网站

开始执行。
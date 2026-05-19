// 论文投稿期刊推荐系统 - 前端逻辑

const API_BASE = '/api';

// 模式切换
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const mode = btn.dataset.mode;
        const abstractGroup = document.getElementById('abstract-group');
        const fulltextGroup = document.getElementById('fulltext-group');

        if (mode === 'title') {
            abstractGroup.classList.add('hidden');
            fulltextGroup.classList.add('hidden');
        } else if (mode === 'abstract') {
            abstractGroup.classList.remove('hidden');
            fulltextGroup.classList.add('hidden');
        } else {
            abstractGroup.classList.remove('hidden');
            fulltextGroup.classList.remove('hidden');
        }
    });
});

// 推荐按钮
document.getElementById('recommend-btn').addEventListener('click', async () => {
    const title = document.getElementById('title').value.trim();
    if (!title) {
        alert('请输入论文标题');
        return;
    }

    const mode = document.querySelector('.mode-btn.active').dataset.mode;
    const abstract = document.getElementById('abstract').value.trim();

    // 处理全文文件上传
    let fullText = '';
    if (mode === 'full') {
        const fileInput = document.getElementById('full_text');
        const file = fileInput.files[0];
        if (file) {
            fullText = await readFileContent(file);
        }
    }

    const topK = parseInt(document.getElementById('top_k').value);
    const oaPreference = document.getElementById('oa_preference').value;

    const btn = document.getElementById('recommend-btn');
    btn.disabled = true;
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = '准备中...';
    document.getElementById('results').innerHTML = '';
    document.getElementById('warning').classList.add('hidden');

    // 收集推荐结果
    const recommendations = [];

    try {
        // 构建 SSE URL
        const params = new URLSearchParams({
            title: title,
            abstract: abstract,
            mode: mode,
            top_k: topK,
            oa_preference: oaPreference,
        });

        const response = await fetch(`${API_BASE}/recommend/stream?${params}`, {
            method: 'GET',
            headers: {
                'Accept': 'text/event-stream',
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 处理 SSE 事件 (event: type\ndata: {...}\n\n)
            while (buffer.includes('\n\n')) {
                const eventEnd = buffer.indexOf('\n\n');
                const eventBlock = buffer.slice(0, eventEnd);
                buffer = buffer.slice(eventEnd + 2);

                const eventLineMatch = eventBlock.match(/^event: (\w+)$/m);
                const dataLineMatch = eventBlock.match(/^data: (.+)$/m);

                if (eventLineMatch && dataLineMatch) {
                    const eventType = eventLineMatch[1];
                    const eventData = JSON.parse(dataLineMatch[1]);

                    switch (eventType) {
                        case 'progress':
                            updateProgress(eventData);
                            break;
                        case 'recommendation':
                            recommendations.push(eventData);
                            renderRecommendation(eventData);
                            break;
                        case 'done':
                            document.getElementById('loading').classList.add('hidden');
                            break;
                        case 'error':
                            throw new Error(eventData.message || 'Unknown error');
                    }
                }
            }
        }

        // 显示警告
        if (recommendations.length === 0) {
            document.getElementById('results').innerHTML = '<p>未找到合适的推荐期刊</p>';
        }

    } catch (error) {
        document.getElementById('results').innerHTML = `
            <div class="warning">请求失败: ${error.message}</div>
        `;
    } finally {
        btn.disabled = false;
        document.getElementById('loading').classList.add('hidden');
    }
});

function updateProgress(data) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressFill.style.width = `${data.percent}%`;
    progressText.textContent = data.message || `进度 ${data.percent}%`;
}

function renderRecommendation(rec) {
    const resultsEl = document.getElementById('results');

    const rankMethodText = rec.rank_method === 'llm' ? 'AI智能排序' : '规则排序';
    const rankMethodClass = rec.rank_method === 'llm' ? 'rank-llm' : 'rank-rule';
    const methodBadge = `<span class="rank-badge ${rankMethodClass}">${rankMethodText}</span>`;

    const card = document.createElement('div');
    card.className = 'journal-card';
    card.innerHTML = `
        <div class="journal-header">
            <span class="journal-name">${rec.journal_name}</span>
            ${rec.quartile ? `<span class="journal-quartile quartile-${rec.quartile.toLowerCase()}">${rec.quartile}</span>` : ''}
            ${methodBadge}
        </div>
        <div class="journal-tags">
            ${rec.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            <span class="tag">${rec.oa_type === 'full_oa' ? '完全OA' : rec.oa_type === 'hybrid' ? '混合OA' : '订阅'}</span>
        </div>
        <ul class="journal-reasons">
            ${rec.match_reasons.map(r => `<li>${r}</li>`).join('')}
        </ul>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${rec.confidence * 100}%"></div>
        </div>
        ${rec.submission_url ? `<a href="${rec.submission_url}" target="_blank">投稿链接</a>` : ''}
    `;
    resultsEl.appendChild(card);
}

// 读取文件内容
async function readFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsText(file);
    });
}

function renderResults(data) {
    const resultsEl = document.getElementById('results');

    if (!data.recommendations || data.recommendations.length === 0) {
        resultsEl.innerHTML = '<p>未找到合适的推荐期刊</p>';
        return;
    }

    // 显示排序方法
    const rankMethodText = data.rank_method === 'llm' ? 'AI智能排序' : '规则排序';
    const rankMethodClass = data.rank_method === 'llm' ? 'rank-llm' : 'rank-rule';
    const methodBadge = `<span class="rank-badge ${rankMethodClass}">${rankMethodText}</span>`;

    resultsEl.innerHTML = data.recommendations.map(rec => `
        <div class="journal-card">
            <div class="journal-header">
                <span class="journal-name">${rec.journal_name}</span>
                ${rec.quartile ? `<span class="journal-quartile quartile-${rec.quartile.toLowerCase()}">${rec.quartile}</span>` : ''}
                ${methodBadge}
            </div>
            <div class="journal-tags">
                ${rec.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                <span class="tag">${rec.oa_type === 'full_oa' ? '完全OA' : rec.oa_type === 'hybrid' ? '混合OA' : '订阅'}</span>
            </div>
            <ul class="journal-reasons">
                ${rec.match_reasons.map(r => `<li>${r}</li>`).join('')}
            </ul>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${rec.confidence * 100}%"></div>
            </div>
            ${rec.submission_url ? `<a href="${rec.submission_url}" target="_blank">投稿链接</a>` : ''}
        </div>
    `).join('');
}
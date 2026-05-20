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
        const modeIndicator = document.getElementById('mode-indicator');

        const modeLabels = { title: '标题模式', abstract: '摘要模式', full: '全文模式' };
        modeIndicator.textContent = modeLabels[mode];

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

// 文件上传显示文件名
document.getElementById('full_text').addEventListener('change', (e) => {
    const fileName = document.getElementById('file-name');
    const file = e.target.files[0];
    fileName.textContent = file ? file.name : '上传 PDF、TXT 或 MD 文件';
});

// 重置各状态
function resetStates() {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('error-state').classList.add('hidden');
    document.getElementById('results').innerHTML = '';
    document.getElementById('result-count').textContent = '';
}

// 显示空状态
function showEmptyState() {
    resetStates();
    document.getElementById('empty-state').classList.remove('hidden');
}

function updateProgress(data) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    if (progressFill) progressFill.style.width = `${data.percent}%`;
    if (progressText) progressText.textContent = data.message || `处理中 ${data.percent}%`;
}

// 显示错误
function showError(message) {
    resetStates();
    document.getElementById('error-message').textContent = message;
    document.getElementById('error-state').classList.remove('hidden');
}

// 显示 loading
function showLoading() {
    resetStates();
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    if (progressFill) progressFill.style.width = '0%';
    if (progressText) progressText.textContent = '准备中...';
    document.getElementById('loading').classList.remove('hidden');
}

// 推荐按钮
document.getElementById('recommend-btn').addEventListener('click', async () => {
    const title = document.getElementById('title').value.trim();
    if (!title) {
        showError('请输入论文标题');
        return;
    }

    const mode = document.querySelector('.mode-btn.active').dataset.mode;
    const abstract = document.getElementById('abstract').value.trim();

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
    showLoading();

    const recommendations = [];

    try {
        const params = new URLSearchParams({
            title: title,
            abstract: abstract,
            mode: mode,
            top_k: topK,
            oa_preference: oaPreference,
        });

        const response = await fetch(`${API_BASE}/recommend/stream?${params}`, {
            method: 'GET',
            headers: { 'Accept': 'text/event-stream' },
        });

        if (!response.ok) {
            throw new Error(`请求失败 (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

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
                            break;
                        case 'done':
                            latestDoneData = eventData;
                            renderResults(recommendations, latestDoneData);
                            break;
                        case 'error':
                            throw new Error(eventData.message || 'Unknown error');
                    }
                }
            }
        }

        if (recommendations.length === 0 && !buffer.includes('event: done')) {
            showEmptyState();
        }

    } catch (error) {
        showError(error.message);
    } finally {
        btn.disabled = false;
    }
});

let latestDoneData = null;

function renderResults(recommendations, doneData = null) {
    const resultsEl = document.getElementById('results');
    const countEl = document.getElementById('result-count');

    resetStates();

    if (!recommendations || recommendations.length === 0) {
        showEmptyState();
        return;
    }

    countEl.textContent = `${recommendations.length} 个结果`;

    // 构建质量信息 HTML
    let qualityHtml = '';
    if (doneData && doneData.quality) {
        const q = doneData.quality;
        const levelClass = `quality-${q.level.toLowerCase()}`;
        const readinessLabel = {
            'Ready': '可投稿',
            'Preliminary': '待完善',
            'Needs-Revision': '需修改'
        };
        qualityHtml = `
        <div class="quality-badge-container">
            <span class="quality-badge ${levelClass}">${q.level}</span>
            <span class="quality-strength">强度 ${Math.round((q.paper_strength || 0.5) * 100)}%</span>
            <span class="quality-readiness">${readinessLabel[q.readiness] || q.readiness}</span>
            <span class="quality-confidence">置信 ${Math.round(q.confidence * 100)}%</span>
        </div>`;
    }

    resultsEl.innerHTML = qualityHtml + recommendations.map((rec, idx) => {
        const rankMethodText = rec.rank_method === 'llm' ? 'AI智能' : '规则';
        const rankMethodClass = rec.rank_method === 'llm' ? 'rank-llm' : 'rank-rule';
        const quartileClass = rec.quartile ? `quartile-${rec.quartile.toLowerCase()}` : '';
        const quartileHtml = rec.quartile ? `<span class="journal-quartile ${quartileClass}">${rec.quartile}</span>` : '';
        const ccfHtml = rec.ccf_rating ? `<span class="ccf-badge ccf-${rec.ccf_rating.toLowerCase()}">CCF-${rec.ccf_rating}</span>` : '';

        const oaLabel = rec.oa_type === 'full_oa' ? '完全OA' : rec.oa_type === 'hybrid' ? '混合OA' : '订阅';

        return `
        <div class="journal-card" data-index="${idx}">
            <div class="card-header">
                <div class="card-title-row">
                    <span class="journal-name">${rec.journal_name}</span>
                    ${quartileHtml}
                    ${ccfHtml}
                    <span class="rank-badge ${rankMethodClass}">${rankMethodText}</span>
                </div>
                <div class="card-actions">
                    <button class="expand-btn" onclick="toggleCard(${idx})" aria-label="展开详情">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M6 9l6 6 6-6"/>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="card-tags">
                    ${rec.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    <span class="tag">${oaLabel}</span>
                </div>
                <ul class="card-reasons">
                    ${rec.match_reasons.map(r => `<li>${r}</li>`).join('')}
                </ul>
                <div class="confidence-row">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${Math.round(rec.confidence * 100)}%"></div>
                    </div>
                    <span class="confidence-text">${Math.round(rec.confidence * 100)}%</span>
                </div>
            </div>
            <div class="card-details">
                <div class="details-grid">
                    ${rec.submission_url ? `
                    <div class="detail-item">
                        <label>投稿链接</label>
                        <a href="${rec.submission_url}" target="_blank">打开链接</a>
                    </div>
                    ` : ''}
                    ${rec.homepage_url ? `
                    <div class="detail-item">
                        <label>期刊主页</label>
                        <a href="${rec.homepage_url}" target="_blank">打开链接</a>
                    </div>
                    ` : ''}
                    ${rec.publisher ? `
                    <div class="detail-item">
                        <label>出版社</label>
                        <span>${rec.publisher}</span>
                    </div>
                    ` : ''}
                    ${rec.impact_like_score ? `
                    <div class="detail-item">
                        <label>影响因子</label>
                        <span>${rec.impact_like_score}</span>
                    </div>
                    ` : ''}
                    ${rec.review_time ? `
                    <div class="detail-item">
                        <label>审稿周期</label>
                        <span>${rec.review_time}</span>
                    </div>
                    ` : ''}
                    ${rec.apc ? `
                    <div class="detail-item">
                        <label>版面费</label>
                        <span>${rec.apc}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        </div>
        `;
    }).join('');
}

// 展开/收起卡片
window.toggleCard = function(index) {
    const card = document.querySelector(`.journal-card[data-index="${index}"]`);
    if (card) {
        card.classList.toggle('expanded');
    }
};

// 读取文件内容
async function readFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsText(file);
    });
}
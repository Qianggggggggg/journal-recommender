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
    document.getElementById('download-pdf-btn').classList.add('hidden');
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

    let file = null;
    if (mode === 'full') {
        const fileInput = document.getElementById('full_text');
        file = fileInput.files[0];
        if (!file) {
            showError('请上传 PDF 文件（全文模式必须上传论文）');
            return;
        }
    }

    const topK = parseInt(document.getElementById('top_k').value);
    const oaPreference = document.getElementById('oa_preference').value;

    const btn = document.getElementById('recommend-btn');
    btn.disabled = true;
    showLoading();

    const recommendations = [];

    try {
        let response;
        const params = {
            title: title,
            abstract: abstract,
            mode: mode,
            top_k: topK,
            oa_preference: oaPreference,
        };
        latestParams = params;

        if (mode === 'full' && file) {
            // full 模式：Form-data 上传文件，后端解析 PDF
            const formData = new FormData();
            formData.append('title', title);
            formData.append('abstract', abstract);
            formData.append('mode', mode);
            formData.append('top_k', topK);
            formData.append('oa_preference', oaPreference);
            formData.append('file', file);

            response = await fetch(`${API_BASE}/recommend/stream`, {
                method: 'POST',
                body: formData,
                headers: { 'Accept': 'text/event-stream' },
            });
        } else {
            // title/abstract 模式：GET URL 参数
            const params = {
                title: title,
                abstract: abstract,
                mode: mode,
                top_k: topK,
                oa_preference: oaPreference,
            };
            latestParams = params;
            const urlParams = new URLSearchParams(params);
            response = await fetch(`${API_BASE}/recommend/stream?${urlParams}`, {
                method: 'GET',
                headers: { 'Accept': 'text/event-stream' },
            });
        }

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
                            window._latestRecommendations = recommendations;
                            window._latestPaperProfile = eventData.paper_profile || null;
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
let latestParams = null;

// 下载 PDF 按钮
document.getElementById('download-pdf-btn').addEventListener('click', async () => {
    if (!latestParams || !latestDoneData) return;

    const btn = document.getElementById('download-pdf-btn');
    btn.disabled = true;
    btn.innerHTML = `<svg class="spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" stroke-dasharray="30" stroke-dashoffset="10"/></svg> 生成中...`;

    try {
        const payload = {
            ...latestParams,
            recommendations: window._latestRecommendations || [],
            paper_profile: window._latestPaperProfile || null,
            quality: latestDoneData?.quality || null,  // 论文评级、强度、置信度
        };

        const response = await fetch(`${API_BASE}/recommend/pdf/from-results`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(`下载失败 (${response.status})`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `期刊推荐报告_${latestParams.title.substring(0, 20)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

    } catch (error) {
        showError(error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg> 下载 PDF`;
    }
});

function renderResults(recommendations, doneData = null) {
    const resultsEl = document.getElementById('results');
    const countEl = document.getElementById('result-count');

    resetStates();

    if (!recommendations || recommendations.length === 0) {
        showEmptyState();
        return;
    }

    countEl.textContent = `${recommendations.length} 个结果`;
    document.getElementById('download-pdf-btn').classList.remove('hidden');

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
        const rankMethodText = '';
        const rankMethodClass = rec.rank_method === 'llm' ? 'rank-llm' : 'rank-rule';
        // 确保 ccf_rating 显示，有默认值 "N/A" 如果没有
        const ccfRating = rec.ccf_rating || '';
        const ccfClass = ccfRating ? `ccf-${ccfRating.toLowerCase()}` : '';
        const ccfHtml = ccfRating ? `<span class="ccf-badge ${ccfClass}" style="display:inline-block;padding:0.15rem 0.4rem;border-radius:3px;font-size:0.6rem;font-weight:700;margin-left:0.3rem;background:${ccfRating==='A'?'linear-gradient(135deg,#d4a017,#f5d06a)':ccfRating==='B'?'linear-gradient(135deg,#1e40af,#3b82f6)':ccfRating==='C'?'linear-gradient(135deg,#16a34a,#4ade80)':ccfRating==='D'?'linear-gradient(135deg,#6b7280,#9ca3af)':''};color:${ccfRating==='A'?'#1a1a1a':'#fff'}">CCF-${ccfRating}</span>` : '';

        const oaLabel = rec.oa_type === 'full_oa' ? '完全OA' : rec.oa_type === 'hybrid' ? '混合OA' : '订阅';

        return `
        <div class="journal-card" data-index="${idx}" style="animation-delay: ${idx * 0.08}s">
            <div class="card-header">
                <div class="card-title-row">
                    <span class="journal-name">${rec.journal_name}</span>
                    ${ccfHtml}
                    ${rankMethodText ? `<span class="rank-badge ${rankMethodClass}">${rankMethodText}</span>` : ''}
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
                    ${(rec.match_reasons || []).filter((r, i, arr) => arr.indexOf(r) === i).map(r => `<li>${r}</li>`).join('')}
                </ul>
                <div class="confidence-row">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${Math.round((rec.score || 0) * 100)}%"></div>
                    </div>
                    <span class="confidence-text">匹配度: ${Math.round((rec.score || 0) * 100)}%</span>
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
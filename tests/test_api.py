"""API 测试"""
import json
import pytest
from starlette.testclient import TestClient

from src.app.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


def test_health_endpoint(client):
    """测试健康检查接口"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_recommend_endpoint(client):
    """测试推荐接口（mock 数据）"""
    response = client.post(
        "/api/recommend",
        json={
            "title": "Deep Learning for Image Recognition",
            "abstract": "This paper proposes a new method.",
            "mode": "abstract",
            "top_k": 3,
        },
    )
    # 取决于是否有数据，可能返回 200 或空结果
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "mode_used" in data


def test_journals_endpoint(client):
    """测试期刊列表接口"""
    response = client.get("/api/journals?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "journals" in data
    assert "total" in data


def test_recommend_stream_endpoint(client):
    """测试 SSE 流式推荐接口"""
    response = client.get(
        "/api/recommend/stream",
        params={
            "title": "Deep Learning for Image Recognition",
            "abstract": "This paper proposes a new method.",
            "mode": "abstract",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # 收集所有事件 (event_type -> data)
    events_by_type = {}
    raw = response.text

    # 解析 SSE 格式: event: type\ndata: {...}\n\n
    import re
    event_pattern = re.compile(r'event: (\w+)\ndata: (.+)\n\n')

    for match in event_pattern.finditer(raw):
        event_type = match.group(1)
        event_data = json.loads(match.group(2))
        if event_type not in events_by_type:
            events_by_type[event_type] = []
        events_by_type[event_type].append(event_data)

    # 验证进度阶段
    assert "progress" in events_by_type, "Should have progress events"
    progress_events = events_by_type["progress"]
    stages = [e.get("stage") for e in progress_events if e.get("stage")]

    assert "parsing" in stages, f"Should have parsing stage, got: {stages}"
    assert "retrieval" in stages, f"Should have retrieval stage, got: {stages}"
    assert "ranking" in stages, f"Should have ranking stage, got: {stages}"

    # 验证有结果（recommendation 或 done 事件）
    assert "recommendation" in events_by_type or "done" in events_by_type, \
        f"Should have recommendation or done events, got: {list(events_by_type.keys())}"

    # 验证 done 事件（如果存在）包含必要字段
    if "done" in events_by_type:
        done_data = events_by_type["done"][0]
        assert "total" in done_data or "rank_method" in done_data
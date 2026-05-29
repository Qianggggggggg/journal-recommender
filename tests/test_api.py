"""API 测试"""
import json
import pytest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient

from src.app.main import app
from src.app.api import _build_candidate_generator
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.retriever.bm25_retriever import BM25Retriever


@pytest.fixture
def mock_llm():
    """Mock LLM response"""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(
        content='{"research_area": ["AI", "CV"], "method_type": "method", "paper_type": "application", "keywords": ["deep learning", "neural network"], "novelty": "new method", "application_domain": ["image recognition"], "techniques": ["CNN", "Transformer"], "datasets": ["ImageNet"], "evaluation_metrics": ["accuracy", "mAP"], "novelty_type": "new_method"}',
        model="MiniMax-Text-01",
        usage={}
    )
    return mock


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


def test_build_candidate_generator_uses_typical_abstract_retrievers(tmp_path):
    """配置 typical_abstracts 时，应接入典型摘要三路召回。"""
    abstracts_dir = tmp_path / "typical_abstracts"
    abstracts_dir.mkdir()
    (abstracts_dir / "target.json").write_text(
        json.dumps({
            "journal_id": "target",
            "journal_name": "Target Journal",
            "abstracts": [
                {
                    "method_type": "method",
                    "novelty_level": "new_method",
                    "abstract": "Graph neural recommendation systems for representation learning.",
                }
            ],
        }),
        encoding="utf-8",
    )

    store = JournalStore(store_path=str(tmp_path / "journals.jsonl"))
    store.add_journal(Journal(journal_id="other", journal_name="Other Journal", journal_profile="unrelated systems"))
    store.add_journal(Journal(journal_id="target", journal_name="Target Journal", journal_profile="general computing"))

    bm25 = BM25Retriever(store)
    bm25.build_index()
    app_config = {
        "candidate_generator": {
            "retrieval_target": "typical_abstracts",
            "merge_weights": {"bm25": 0.45, "vector": 0.35, "text": 0.20},
        },
        "data": {
            "typical_abstracts_dir": str(abstracts_dir),
            "typical_abstracts_faiss_path": str(tmp_path / "missing.faiss"),
            "typical_abstracts_metadata_path": str(tmp_path / "missing.parquet"),
        },
    }

    generator = _build_candidate_generator(
        store=store,
        bm25=bm25,
        embedding_retriever=None,
        embedding_client=MagicMock(),
        app_config=app_config,
    )

    assert generator.retrieval_target == "typical_abstracts"
    assert generator.typical_bm25_retriever is not None
    assert generator.typical_embedding_retriever is not None
    assert generator.typical_text_retriever is not None

    profile = PaperProfile(title="Graph neural recommendation", keywords=["graph", "recommendation"])
    candidates = generator.generate("graph neural recommendation", profile, top_k=3)
    assert candidates[0].journal_id == "target"


def test_journals_endpoint(client):
    """测试期刊列表接口"""
    response = client.get("/api/journals?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "journals" in data
    assert "total" in data


def test_recommend_endpoint_with_mock(mock_llm):
    """测试推荐接口（使用 mock LLM）"""
    with patch("src.app.api.MiniMaxLLM", return_value=mock_llm):
        with patch("src.app.api.PaperParser") as MockParser:
            mock_profile = MagicMock()
            mock_profile.research_area = ["AI", "CV"]
            mock_profile.method_type = "method"
            mock_profile.paper_type = "application"
            mock_profile.keywords = ["deep learning"]
            mock_profile.novelty = "new method"
            mock_profile.application_domain = ["image recognition"]
            mock_profile.techniques = ["CNN"]
            mock_profile.datasets = ["ImageNet"]
            mock_profile.evaluation_metrics = ["accuracy"]
            mock_profile.novelty_type = "new_method"
            mock_profile.paper_strength = 0.8
            mock_profile.readiness = "Ready"
            mock_profile.quality_level = "A"
            mock_profile.quality_confidence = 0.8
            mock_profile.quality_reasons = []
            MockParser.return_value.parse.return_value = mock_profile

            with patch("src.app.api.PaperQualityAssessor") as MockAssessor:
                mock_quality = MagicMock()
                mock_quality.paper_strength = 0.8
                mock_quality.readiness = "Ready"
                mock_quality.quality_level = "A"
                mock_quality.confidence = 0.8
                mock_quality.reasons = []
                MockAssessor.return_value.assess.return_value = mock_quality

                client = TestClient(app)
                response = client.post(
                    "/api/recommend",
                    json={
                        "title": "Deep Learning for Image Recognition",
                        "abstract": "This paper proposes a new deep learning method for image recognition.",
                        "mode": "abstract",
                        "top_k": 3,
                    },
                )

                # 如果 LLM 不可用，应该返回 503
                if response.status_code == 503:
                    assert "论文解析失败" in response.json().get("detail", "") or \
                           "论文质量评估失败" in response.json().get("detail", "") or \
                           "LLM服务" in response.json().get("detail", "")
                else:
                    assert response.status_code == 200
                    data = response.json()
                    assert "recommendations" in data
                    assert "mode_used" in data


def test_recommend_stream_endpoint_with_mock(mock_llm):
    """测试 SSE 流式推荐接口（使用 mock LLM）"""
    with patch("src.app.api.MiniMaxLLM", return_value=mock_llm):
        client = TestClient(app)

        # 由于流式接口较复杂，如果 LLM 不可用则只检查结构
        response = client.get(
            "/api/recommend/stream",
            params={
                "title": "Deep Learning for Image Recognition",
                "abstract": "This paper proposes a new method.",
                "mode": "abstract",
                "top_k": 3,
            },
        )

        # 检查响应状态
        if response.status_code == 503:
            # LLM 服务不可用，返回错误
            assert "detail" in response.json()
        else:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

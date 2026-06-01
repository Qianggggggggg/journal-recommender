import numpy as np

from src.utils.embedding import OllamaEmbedding


def test_ollama_embedding_uses_configured_timeout(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [0.1, 0.2, 0.3]}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("src.utils.embedding.requests.post", fake_post)

    embedding = OllamaEmbedding(
        base_url="http://localhost:11434",
        model="test-embedding",
        timeout=180,
    )

    result = embedding.embed("long abstract")

    assert captured["timeout"] == 180
    assert np.allclose(result, np.array([0.1, 0.2, 0.3]))

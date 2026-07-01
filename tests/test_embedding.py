import numpy as np

from src.utils.embedding import OllamaEmbedding, encode_query


def test_ollama_embedding_uses_configured_timeout(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2, 0.3]]}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("src.utils.embedding.requests.post", fake_post)

    embedding = OllamaEmbedding(
        base_url="http://localhost:11434",
        model="test-embedding",
        timeout=180,
    )

    result = embedding.embed("long abstract")

    assert captured["url"] == "http://localhost:11434/api/embed"
    assert captured["payload"] == {
        "model": "test-embedding",
        "input": "long abstract",
    }
    assert captured["timeout"] == 180
    assert np.allclose(result, np.array([0.1, 0.2, 0.3]))


def test_embed_batch_shows_configured_progress(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2]]}

    class DummyProgress:
        def __init__(self, values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

        def write(self, message):
            captured.setdefault("messages", []).append(message)

    def fake_tqdm(values, **kwargs):
        captured.update(kwargs)
        return DummyProgress(values)

    monkeypatch.setattr("src.utils.embedding.tqdm", fake_tqdm)
    monkeypatch.setattr(
        "src.utils.embedding.requests.post",
        lambda *args, **kwargs: DummyResponse(),
    )

    embedding = OllamaEmbedding(
        show_progress=True,
        progress_desc="Accepted-paper embeddings",
    )
    results = embedding.embed_batch(["first", "second"])

    assert len(results) == 2
    assert captured["total"] == 2
    assert captured["desc"] == "Accepted-paper embeddings"
    assert captured["unit"] == "paper"
    assert captured["disable"] is False


def test_embed_batch_sanitizes_and_truncates_oversized_latex_text(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2]]}

    def fake_post(url, json, timeout):
        captured["input"] = json["input"]
        return DummyResponse()

    monkeypatch.setattr("src.utils.embedding.requests.post", fake_post)

    latex_rendering = (
        r"\documentclass[12pt]{minimal} "
        r"\usepackage{amsmath} "
        r"\begin{document}$$O(4^k)$$\end{document}"
    )
    text = f"Achieving Tight {latex_rendering} Runtime Bounds. " + ("result " * 1000)

    embedding = OllamaEmbedding(max_input_chars=4000)
    embedding.embed_batch([text])

    assert len(captured["input"]) <= 4000
    assert r"\documentclass" not in captured["input"]
    assert r"\usepackage" not in captured["input"]
    assert "$$O(4^k)$$" in captured["input"]
    assert captured["input"].startswith("Achieving Tight")


def test_embed_query_adds_instruction_but_document_embedding_does_not(monkeypatch):
    inputs = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2]]}

    def fake_post(url, json, timeout):
        inputs.append(json["input"])
        return DummyResponse()

    monkeypatch.setattr("src.utils.embedding.requests.post", fake_post)
    embedding = OllamaEmbedding(query_instruction="Retrieve relevant papers.")

    embedding.embed("document title and abstract")
    embedding.embed_query("manuscript title and abstract")

    assert inputs == [
        "document title and abstract",
        "Instruct: Retrieve relevant papers.\n"
        "Query: manuscript title and abstract",
    ]


def test_encode_query_prefers_query_method_and_supports_legacy_clients():
    class QueryAwareClient:
        def embed(self, text):
            return f"document:{text}"

        def embed_query(self, text):
            return f"query:{text}"

    class LegacyClient:
        def embed(self, text):
            return f"legacy:{text}"

    assert encode_query(QueryAwareClient(), "paper") == "query:paper"
    assert encode_query(LegacyClient(), "paper") == "legacy:paper"

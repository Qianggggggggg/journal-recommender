"""Ollama Embedding 调用封装"""
import re
from typing import List
from typing import Optional

import numpy as np
import requests
from tqdm.auto import tqdm


DEFAULT_QUERY_INSTRUCTION = (
    "Given a manuscript title and abstract, retrieve journal descriptions and "
    "published papers that best match its research topic, methods, application "
    "domain, and publication scope."
)

DEFAULT_MAX_INPUT_CHARS = 4000
_LATEX_DOCUMENT_RE = re.compile(
    r"\\documentclass(?:\[[^\]]*\])?\{[^{}]*\}"
    r".*?\\begin\{document\}(.*?)\\end\{document\}",
    re.DOTALL,
)
_LATEX_PACKAGE_RE = re.compile(
    r"\\(?:usepackage|setlength)(?:\[[^\]]*\])?\{[^{}]*\}(?:\{[^{}]*\})?"
)
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_LINE_BREAK_RE = re.compile(r" *\n+ *")


def prepare_embedding_text(
    text: str,
    *,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
) -> str:
    """Remove metadata-rendering noise and cap document size for Ollama.

    Semantic Scholar/Springer abstracts may inline a complete LaTeX preamble
    around every formula. Those command-heavy strings can make Ollama's
    embedding runner terminate with a 400/EOF response. Keep the rendered math
    expression, discard the preamble, then retain a generous title/abstract
    prefix that is sufficient for retrieval.
    """
    cleaned = str(text or "")
    cleaned = _LATEX_DOCUMENT_RE.sub(lambda match: f" {match.group(1)} ", cleaned)
    cleaned = _LATEX_PACKAGE_RE.sub(" ", cleaned)
    cleaned = _HORIZONTAL_WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _LINE_BREAK_RE.sub("\n", cleaned).strip()

    if max_input_chars > 0 and len(cleaned) > max_input_chars:
        cut = cleaned.rfind(" ", 0, max_input_chars + 1)
        if cut < max_input_chars * 0.8:
            cut = max_input_chars
        cleaned = cleaned[:cut].rstrip()
    return cleaned


def _raise_for_status_with_context(response, input_text: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = (getattr(response, "text", "") or "").strip()
        detail = body[:500] if body else "<empty response>"
        raise requests.HTTPError(
            f"{exc}; input_chars={len(input_text)}; response={detail}",
            response=response,
        ) from exc


def encode_query(embedding_client, text: str) -> np.ndarray:
    """Encode a retrieval query while preserving compatibility with test doubles."""
    query_method = getattr(embedding_client, "embed_query", None)
    if callable(query_method):
        return query_method(text)
    return embedding_client.embed(text)


class OllamaEmbedding:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
        timeout: float = 60.0,
        show_progress: bool = False,
        progress_desc: str = "Embedding",
        query_instruction: Optional[str] = None,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.max_input_chars = max_input_chars
        self.query_instruction = (
            DEFAULT_QUERY_INSTRUCTION
            if query_instruction is None
            else query_instruction.strip()
        )

    def embed(self, text: str) -> np.ndarray:
        """Encode one document without a retrieval-query instruction.

        Retries on 5xx and connection errors
        (Ollama occasionally 500s when the desktop app self-updates in the
        background, or when the model is reloaded after a long idle period)."""
        import time as _time
        prepared_text = prepare_embedding_text(
            text,
            max_input_chars=self.max_input_chars,
        )
        for attempt in range(5):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": prepared_text},
                    timeout=self.timeout,
                )
                _raise_for_status_with_context(response, prepared_text)
                return np.array(response.json()["embeddings"][0])
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = (
                    status in (500, 502, 503, 504)
                    or isinstance(e, (requests.ConnectionError, requests.Timeout))
                )
                if retryable and attempt < 4:
                    wait = min(2 ** attempt, 30)
                    print(f"  [embed] {type(e).__name__} {status or ''} retry in {wait}s",
                          flush=True)
                    _time.sleep(wait)
                    continue
                raise

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a query with Qwen3's asymmetric-retrieval instruction."""
        if not self.query_instruction:
            return self.embed(text)
        instructed = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self.embed(instructed)

    def embed_batch(self, texts: List[str], concurrency: int = 1, timeout: Optional[float] = None) -> List[np.ndarray]:
        """Encode documents without query instructions, one request per item."""
        request_timeout = self.timeout if timeout is None else timeout
        results = []
        progress = tqdm(
            texts,
            total=len(texts),
            desc=self.progress_desc,
            unit="paper",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        for text in progress:
            prepared_text = prepare_embedding_text(
                text,
                max_input_chars=self.max_input_chars,
            )
            for attempt in range(5):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": prepared_text},
                        timeout=request_timeout,
                    )
                    _raise_for_status_with_context(response, prepared_text)
                    results.append(np.array(response.json()["embeddings"][0]))
                    break
                except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    retryable = (
                        status in (500, 502, 503, 504)
                        or isinstance(e, (requests.ConnectionError, requests.Timeout))
                    )
                    if retryable and attempt < 4:
                        import time as _time
                        wait = min(2 ** attempt, 30)
                        progress.write(
                            f"  [embed_batch] {type(e).__name__} "
                            f"{status or ''} retry in {wait}s"
                        )
                        _time.sleep(wait)
                        continue
                    raise
        return results

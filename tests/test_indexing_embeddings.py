from __future__ import annotations

from typing import Any

import pytest

from legal_rag.indexing import embeddings
from legal_rag.indexing.models import IndexingConfig


def test_build_embedder_missing_utopia_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UTOPIA_API_KEY", raising=False)
    config = IndexingConfig(embedding_backend="utopia", hybrid_enabled=False, embedding_api_key="", env_file=None)

    with pytest.raises(RuntimeError, match="UTOPIA_API_KEY is missing"):
        embeddings.build_embedder(config)


def test_utopia_backend_rejects_hybrid() -> None:
    config = IndexingConfig(embedding_backend="utopia", hybrid_enabled=True, embedding_api_key="secret", env_file=None)

    with pytest.raises(RuntimeError, match="dense-only"):
        embeddings.build_embedder(config)


def test_utopia_ollama_embedder_parses_legacy_response(monkeypatch: pytest.MonkeyPatch) -> None:
    requests_seen: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {"unexpected": True}

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        requests_seen.append({"url": args[0], "json": kwargs["json"]})
        return FakeResponse()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    config = IndexingConfig(
        embedding_backend="utopia",
        hybrid_enabled=False,
        utopia_embed_url="https://utopia.local/ollama/api/embeddings",
        embedding_model="test-embed-model",
        embedding_api_key="secret",
        env_file=None,
    )

    embedder = embeddings.build_embedder(config)

    assert isinstance(embedder, embeddings.UtopiaOllamaEmbedder)
    def legacy_post(*args: Any, **kwargs: Any) -> FakeResponse:
        requests_seen.append({"url": args[0], "json": kwargs["json"]})
        if isinstance(kwargs["json"].get("prompt"), str):
            response = FakeResponse()
            response.json = lambda: {"embedding": [0.4, 0.5, 0.6]}  # type: ignore[method-assign]
            return response
        return FakeResponse()

    monkeypatch.setattr(embeddings.requests, "post", legacy_post)

    assert embedder.embed_texts(["ciao"]) == [[0.4, 0.5, 0.6]]
    assert embedder.active_variant == "prompt_single"
    assert requests_seen[0]["json"]["input"] == ["ciao"]


def test_utopia_ollama_embedder_uses_and_caches_batched_embed_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    requests_seen: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        requests_seen.append({"url": url, "json": kwargs["json"]})
        if url.endswith("/embeddings"):
            return FakeResponse({"error": "wrong shape"})
        return FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    embedder = embeddings.UtopiaOllamaEmbedder(
        api_key="secret",
        model="test-embed-model",
        embed_url="https://utopia.local/ollama/api/embeddings",
    )

    assert embedder.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert embedder.active_endpoint.endswith("/embed")
    assert embedder.active_variant == "input_list"

    requests_seen.clear()
    assert embedder.embed_texts(["c", "d"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert [row["url"] for row in requests_seen] == ["https://utopia.local/ollama/api/embed"]


def test_local_bge_m3_adapter_parses_dense_and_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBgeModel:
        def __init__(self, model: str, use_fp16: bool) -> None:
            self.model = model
            self.use_fp16 = use_fp16

        def encode(self, texts: list[str], *, return_dense: bool, return_sparse: bool) -> dict[str, Any]:
            if return_sparse:
                return {"lexical_weights": [{"3": 1.5, "1": 0.5} for _ in texts]}
            return {"dense_vecs": [[1.0, 2.0, 3.0] for _ in texts]}

    monkeypatch.setitem(__import__("sys").modules, "FlagEmbedding", type("FakeFlagModule", (), {"BGEM3FlagModel": FakeBgeModel}))

    embedder = embeddings.LocalEmbeddingBackend(model="BAAI/bge-m3", hybrid_enabled=True)

    assert embedder.embed_texts(["test"]) == [[1.0, 2.0, 3.0]]
    assert embedder.embed_sparse_texts(["test"]) == [([1, 3], [0.5, 1.5])]

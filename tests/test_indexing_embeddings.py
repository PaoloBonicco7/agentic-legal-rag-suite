from __future__ import annotations

import json
from typing import Any

import pytest

from legal_indexing import embeddings
from legal_indexing.settings import IndexingConfig


def test_build_embedder_returns_langchain_utopia_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embeddings, "OpenAIEmbeddings", FakeOpenAIEmbeddings)

    config = IndexingConfig(
        embedding_provider="utopia",
        utopia_embed_api_mode="openai",
        utopia_base_url="https://utopia.local/api",
        embedding_model="test-embed-model",
        embedding_api_key="secret",
    )

    embedder = embeddings.build_embedder(config)
    assert isinstance(embedder, embeddings.LangChainUtopiaEmbedder)
    assert embedder.model_name == "test-embed-model"
    assert embedder.embed_texts(["ciao"]) == [[0.1, 0.2, 0.3]]
    assert captured["model"] == "test-embed-model"
    assert captured["openai_api_key"] == "secret"
    assert captured["openai_api_base"] == "https://utopia.local/api"


def test_build_embedder_rejects_non_utopia_provider() -> None:
    config = IndexingConfig(
        embedding_provider="deterministic",
        utopia_embed_api_mode="openai",
        utopia_base_url="https://utopia.local/api",
        embedding_model="test-embed-model",
        embedding_api_key="secret",
    )
    with pytest.raises(RuntimeError, match="Only 'utopia' is supported"):
        embeddings.build_embedder(config)


def test_build_embedder_missing_api_key_raises() -> None:
    config = IndexingConfig(
        embedding_provider="utopia",
        utopia_embed_api_mode="openai",
        utopia_base_url="https://utopia.local/api",
        embedding_model="test-embed-model",
        embedding_api_key="",
    )
    with pytest.raises(RuntimeError, match="UTOPIA_API_KEY is missing"):
        embeddings.build_embedder(config)


def test_debug_utopia_embedding_connection_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            self._kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0, 1.0] for _ in texts]

    monkeypatch.setattr(embeddings, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(
        embeddings,
        "discover_utopia_models",
        lambda *args, **kwargs: {
            "ok": True,
            "count": 1,
            "status_code": 200,
            "embedding_like_models": ["test-embed-model"],
            "models": ["test-embed-model"],
        },
    )

    config = IndexingConfig(
        embedding_provider="utopia",
        utopia_embed_api_mode="openai",
        utopia_base_url="https://utopia.local/api",
        embedding_model="test-embed-model",
        embedding_api_key="secret",
    )

    out = embeddings.debug_utopia_embedding_connection(config, probe_text="debug probe")
    assert out["success"] is True
    assert out["vector_size"] == 4
    assert out["model"] == "test-embed-model"
    assert out["base_url"] == "https://utopia.local/api"
    assert isinstance(out["latency_ms"], float)


def test_debug_utopia_embedding_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            self._kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("network unreachable")

    monkeypatch.setattr(embeddings, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(
        embeddings,
        "discover_utopia_models",
        lambda *args, **kwargs: {
            "ok": True,
            "count": 1,
            "status_code": 200,
            "embedding_like_models": ["test-embed-model"],
            "models": ["test-embed-model"],
        },
    )

    config = IndexingConfig(
        embedding_provider="utopia",
        utopia_embed_api_mode="openai",
        utopia_base_url="https://utopia.local/api",
        embedding_model="test-embed-model",
        embedding_api_key="secret",
    )

    with pytest.raises(RuntimeError) as exc_info:
        embeddings.debug_utopia_embedding_connection(config)

    msg = str(exc_info.value)
    assert "Utopia embedding debug check failed:" in msg
    payload = msg.split("Utopia embedding debug check failed:", maxsplit=1)[1].strip()
    details = json.loads(payload)
    assert details["success"] is False
    assert details["error_type"] == "RuntimeError"
    assert "All embedding probe attempts failed" in details["error"]
    assert "network unreachable" in details["attempt_errors"]["test-embed-model"]


def test_build_embedder_auto_mode_falls_back_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            self._kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("openai 500")

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {"embeddings": [[0.4, 0.5, 0.6]]}

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr(embeddings, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    config = IndexingConfig(
        embedding_provider="utopia",
        utopia_embed_api_mode="auto",
        utopia_base_url="https://utopia.local/api",
        utopia_embed_url="https://utopia.local/ollama/api/embed",
        embedding_model="test-embed-model",
        embedding_api_key="secret",
    )

    embedder = embeddings.build_embedder(config)
    assert isinstance(embedder, embeddings.AutoUtopiaEmbedder)
    vectors = embedder.embed_texts(["ciao"])
    assert vectors == [[0.4, 0.5, 0.6]]
    assert embedder.active_mode == "ollama"

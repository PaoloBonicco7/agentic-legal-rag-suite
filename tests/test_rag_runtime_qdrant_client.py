from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import legal_indexing.rag_runtime.langgraph_app as langgraph_app
from legal_indexing.rag_runtime.config import RagRuntimeConfig


class _EmbedderStub:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


def _patch_prepare_runtime_dependencies(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        langgraph_app,
        "validate_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            errors=[],
            warnings=[],
            counts={},
            is_valid=True,
        ),
    )
    monkeypatch.setattr(
        langgraph_app,
        "resolve_index_contract",
        lambda cfg: SimpleNamespace(
            collection_name="laws_test",
            qdrant_path=cfg.resolved_qdrant_path,
            eval_reference_coverage=1.0,
            sparse_vector_name=None,
            sparse_artifacts_path=None,
            run_id="test_run",
        ),
    )
    monkeypatch.setattr(langgraph_app, "build_embedder", lambda *args, **kwargs: _EmbedderStub())
    monkeypatch.setattr(
        langgraph_app,
        "collection_vector_capabilities",
        lambda *args, **kwargs: SimpleNamespace(
            sparse_enabled=False,
            sparse_vector_names=[],
            dense_vector_name=None,
        ),
    )
    monkeypatch.setattr(
        langgraph_app.QdrantRetriever,
        "from_sparse_artifact",
        staticmethod(lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setattr(langgraph_app, "get_vector_size", lambda *args, **kwargs: 8)
    monkeypatch.setattr(
        langgraph_app,
        "introspect_payload_schema",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        langgraph_app,
        "assert_required_payload_fields",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        langgraph_app,
        "LegalGraphAdapter",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        langgraph_app,
        "build_law_catalog",
        lambda *args, **kwargs: SimpleNamespace(),
    )


def test_prepare_runtime_uses_remote_qdrant_client_when_url_is_set(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _patch_prepare_runtime_dependencies(monkeypatch)
    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(dict(kwargs))

        def collection_exists(self, *, collection_name: str) -> bool:
            return bool(collection_name)

        def close(self) -> None:
            return None

    monkeypatch.setattr(langgraph_app, "QdrantClient", _FakeClient)

    cfg = RagRuntimeConfig(
        dataset_dir=tmp_path / "dataset",
        qdrant_path=tmp_path / "qdrant_local",
        indexing_artifacts_root=tmp_path / "artifacts",
        collection_name="laws_test",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_prefer_remote=True,
        llm_provider="disabled",
    )

    resources = langgraph_app.prepare_runtime(cfg)
    try:
        assert len(calls) == 1
        assert calls[0].get("url") == "http://127.0.0.1:6333"
        assert "path" not in calls[0]
    finally:
        resources.close()


def test_prepare_runtime_falls_back_to_local_path_when_remote_is_disabled(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _patch_prepare_runtime_dependencies(monkeypatch)
    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(dict(kwargs))

        def collection_exists(self, *, collection_name: str) -> bool:
            return bool(collection_name)

        def close(self) -> None:
            return None

    monkeypatch.setattr(langgraph_app, "QdrantClient", _FakeClient)

    cfg = RagRuntimeConfig(
        dataset_dir=tmp_path / "dataset",
        qdrant_path=tmp_path / "qdrant_local",
        indexing_artifacts_root=tmp_path / "artifacts",
        collection_name="laws_test",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_prefer_remote=False,
        llm_provider="disabled",
    )

    resources = langgraph_app.prepare_runtime(cfg)
    try:
        assert len(calls) == 1
        assert calls[0].get("path") == str(cfg.resolved_qdrant_path)
        assert "url" not in calls[0]
    finally:
        resources.close()

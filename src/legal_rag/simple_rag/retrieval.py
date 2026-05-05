"""Dense Qdrant retrieval helpers for the simple RAG baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.io import read_json

from .models import RetrievedChunkRecord, SimpleRagConfig


def resolve_index_manifest_path(path_value: str) -> Path:
    """Resolve the configured index manifest path, including the <latest> sentinel."""
    raw = Path(path_value)
    if "<latest>" not in raw.parts:
        return raw
    latest_index = raw.parts.index("<latest>")
    runs_dir = Path(*raw.parts[:latest_index]) if latest_index else Path(".")
    suffix = Path(*raw.parts[latest_index + 1 :])
    candidates = [path for path in runs_dir.iterdir() if path.is_dir() and (path / suffix).exists()]
    if not candidates:
        raise FileNotFoundError(f"No index manifest found under {runs_dir} for pattern {path_value!r}")
    latest = max(candidates, key=lambda path: _index_run_sort_key(path, suffix))
    return latest / suffix


def _index_run_sort_key(run_dir: Path, suffix: Path) -> tuple[int, str, str, float, str]:
    manifest_path = run_dir / suffix
    try:
        manifest = read_json(manifest_path)
    except Exception:
        manifest = {}
    run_id = str(manifest.get("run_id") or run_dir.name)
    collection_name = str(manifest.get("collection_name") or "")
    is_demo = run_id.startswith("notebook_demo") or collection_name.endswith("_notebook_demo")
    created_at = str(manifest.get("created_at") or "")
    return (0 if is_demo else 1, created_at, run_id, run_dir.stat().st_mtime, run_dir.name)


def load_index_manifest(config: SimpleRagConfig) -> tuple[Path, dict[str, Any]]:
    """Load the resolved index manifest."""
    path = resolve_index_manifest_path(config.index_manifest_path)
    return path, read_json(path)


def resolve_collection_name(config: SimpleRagConfig, index_manifest: dict[str, Any]) -> str:
    """Resolve collection name, using the manifest unless config explicitly overrides it."""
    if "collection_name" not in config.model_fields_set and index_manifest.get("collection_name"):
        return str(index_manifest["collection_name"])
    return config.collection_name


def connect_qdrant(config: SimpleRagConfig, index_manifest: dict[str, Any]) -> QdrantClient:
    """Create a Qdrant client for the indexed collection."""
    manifest_config = dict(index_manifest.get("config") or {})
    qdrant_info = dict(index_manifest.get("qdrant") or {})
    qdrant_url = str(
        index_manifest.get("qdrant_url") or qdrant_info.get("url") or manifest_config.get("qdrant_url") or ""
    ).strip()
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    qdrant_path = str(index_manifest.get("qdrant_path") or qdrant_info.get("path") or manifest_config.get("index_dir") or config.index_dir)
    return QdrantClient(path=qdrant_path)


def build_static_filter(static_filters: dict[str, Any]) -> qmodels.Filter | None:
    """Build a Qdrant exact-match filter from run-level static filters."""
    must: list[Any] = []
    for key, value in sorted(static_filters.items()):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [item for item in value if item is not None]
            if not values:
                continue
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=list(values))))
        else:
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
    return qmodels.Filter(must=must) if must else None


def _dense_vector_name(client: QdrantClient, *, collection_name: str) -> str | None:
    info = client.get_collection(collection_name=collection_name)
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if isinstance(vectors, dict):
        if "dense" in vectors:
            return "dense"
        if vectors:
            return str(next(iter(vectors)))
    return None


def search_dense(
    client: QdrantClient,
    *,
    collection_name: str,
    embedder: SupportsEmbedding,
    query_text: str,
    limit: int,
    static_filters: dict[str, Any],
) -> list[RetrievedChunkRecord]:
    """Embed a query and search the dense vector index."""
    vector = embedder.embed_texts([query_text])[0]
    vector_name = _dense_vector_name(client, collection_name=collection_name)
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        using=vector_name,
        query_filter=build_static_filter(static_filters),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    chunks: list[RetrievedChunkRecord] = []
    for point in response.points:
        payload = dict(point.payload or {})
        chunks.append(
            RetrievedChunkRecord(
                chunk_id=str(payload.get("chunk_id") or point.id),
                score=float(point.score),
                text=str(payload.get("text") or ""),
                payload=payload,
            )
        )
    return chunks

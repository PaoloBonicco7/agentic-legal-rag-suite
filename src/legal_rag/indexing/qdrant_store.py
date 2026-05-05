"""Qdrant collection, upload and validation helpers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .models import FILTERABLE_FIELDS, IndexingConfig

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass(frozen=True)
class PreparedPoint:
    """One Qdrant point before vector upload."""

    chunk_id: str
    point_id: str
    embedding_text: str
    payload: dict[str, Any]
    content_hash: str


def safe_collection_component(value: str, *, max_len: int = 48) -> str:
    """Return a Qdrant-safe collection name component."""
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", (value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return (text or "default")[:max_len]


def build_collection_name(config: IndexingConfig, *, dataset_hash: str) -> str:
    """Return the configured collection name."""
    return safe_collection_component(config.collection_name, max_len=128)


def connect_qdrant(config: IndexingConfig) -> QdrantClient:
    """Create a Qdrant client in local file mode unless a URL is explicitly configured."""
    if config.qdrant_url:
        return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    config.resolved_index_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.resolved_index_dir))


def _distance_enum(distance: str) -> qmodels.Distance:
    value = (distance or "").strip().lower()
    if value == "cosine":
        return qmodels.Distance.COSINE
    if value == "dot":
        return qmodels.Distance.DOT
    if value == "euclid":
        return qmodels.Distance.EUCLID
    if value == "manhattan":
        return qmodels.Distance.MANHATTAN
    raise ValueError(f"Unsupported qdrant distance: {distance!r}")


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def ensure_collection(
    client: QdrantClient,
    config: IndexingConfig,
    *,
    collection_name: str,
    vector_size: int,
) -> tuple[bool, int, dict[str, str]]:
    """Create or validate a collection according to the configured rebuild policy."""
    exists = bool(client.collection_exists(collection_name=collection_name))
    removed_count = 0
    if exists and config.force_rebuild:
        removed_count = collection_point_count(client, collection_name=collection_name)
        client.delete_collection(collection_name=collection_name)
        exists = False
    if exists:
        info = client.get_collection(collection_name=collection_name)
        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        existing_size = None
        if isinstance(vectors, dict):
            existing_size = getattr(vectors.get(DENSE_VECTOR_NAME), "size", None)
        else:
            existing_size = getattr(vectors, "size", None)
        if existing_size is not None and int(existing_size) != int(vector_size):
            raise RuntimeError(
                f"Existing collection vector size mismatch: collection={collection_name!r}, "
                f"existing={existing_size}, requested={vector_size}"
            )
        sparse_vectors = getattr(getattr(getattr(info, "config", None), "params", None), "sparse_vectors", None)
        if config.hybrid_enabled and not sparse_vectors:
            raise RuntimeError(f"Existing collection {collection_name!r} does not contain required sparse vector schema")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: qmodels.VectorParams(size=int(vector_size), distance=_distance_enum(config.qdrant_distance))
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=False))
            }
            if config.hybrid_enabled
            else None,
            on_disk_payload=config.qdrant_on_disk_payload,
            hnsw_config=qmodels.HnswConfigDiff(m=config.qdrant_hnsw_m, ef_construct=config.qdrant_hnsw_ef_construct),
        )
    payload_index_statuses = create_payload_indexes(client, collection_name=collection_name)
    return (not exists, removed_count, payload_index_statuses)


def create_payload_indexes(client: QdrantClient, *, collection_name: str) -> dict[str, str]:
    """Create idempotent payload indexes for required filter fields."""
    schemas: dict[str, Any] = {
        "chunk_id": qmodels.PayloadSchemaType.KEYWORD,
        "law_id": qmodels.PayloadSchemaType.KEYWORD,
        "law_status": qmodels.PayloadSchemaType.KEYWORD,
        "index_views": qmodels.PayloadSchemaType.KEYWORD,
        "article_id": qmodels.PayloadSchemaType.KEYWORD,
        "article_status": qmodels.PayloadSchemaType.KEYWORD,
        "relation_types": qmodels.PayloadSchemaType.KEYWORD,
        "law_date": qmodels.PayloadSchemaType.KEYWORD,
        "law_number": qmodels.PayloadSchemaType.INTEGER,
    }
    statuses: dict[str, str] = {}
    for field_name, schema in schemas.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
            statuses[field_name] = "created"
        except Exception as exc:
            text = str(exc).lower()
            statuses[field_name] = "exists_or_ignored" if "already" in text or "exists" in text else f"error: {exc}"
    return statuses


def fetch_existing_content_hashes(client: QdrantClient, *, collection_name: str, point_ids: Sequence[str]) -> dict[str, str]:
    """Return existing content hashes for a set of deterministic point ids."""
    out: dict[str, str] = {}
    for batch in _chunks(list(point_ids), 256):
        records = client.retrieve(collection_name=collection_name, ids=list(batch), with_payload=["content_hash"], with_vectors=False)
        for record in records:
            value = (record.payload or {}).get("content_hash")
            if isinstance(value, str) and value:
                out[str(record.id)] = value
    return out


def upload_point_batch(
    client: QdrantClient,
    *,
    collection_name: str,
    points: list[PreparedPoint],
    vectors: list[list[float]],
    max_retries: int,
    sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
) -> None:
    """Upload one point batch with a small retry loop."""
    qdrant_points: list[qmodels.PointStruct] = []
    for idx, (point, vector) in enumerate(zip(points, vectors, strict=True)):
        point_vector: dict[str, Any] = {DENSE_VECTOR_NAME: [float(x) for x in vector]}
        if sparse_vectors is not None:
            indices, values = sparse_vectors[idx]
            point_vector[SPARSE_VECTOR_NAME] = qmodels.SparseVector(
                indices=[int(item) for item in indices],
                values=[float(item) for item in values],
            )
        qdrant_points.append(qmodels.PointStruct(id=point.point_id, vector=point_vector, payload=point.payload))
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            client.upsert(collection_name=collection_name, points=qdrant_points, wait=True)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(min(2**attempt, 5))
    assert last_error is not None
    raise last_error


def collection_point_count(client: QdrantClient, *, collection_name: str) -> int:
    """Return exact collection point count."""
    return int(client.count(collection_name=collection_name, exact=True).count)


def validate_no_duplicate_chunk_ids(client: QdrantClient, *, collection_name: str) -> dict[str, Any]:
    """Scroll payloads and verify chunk ids are unique."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    offset: Any = None
    scanned = 0
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for record in records:
            scanned += 1
            chunk_id = str((record.payload or {}).get("chunk_id") or "")
            if not chunk_id:
                continue
            if chunk_id in seen:
                duplicates.add(chunk_id)
            seen.add(chunk_id)
        if next_offset is None:
            break
        offset = next_offset
    return {"ok": not duplicates, "total_points_scanned": scanned, "duplicate_chunk_ids": sorted(duplicates), "duplicate_count": len(duplicates)}


def payload_index_fields() -> tuple[str, ...]:
    """Return the payload fields expected to have Qdrant indexes."""
    return tuple(FILTERABLE_FIELDS)

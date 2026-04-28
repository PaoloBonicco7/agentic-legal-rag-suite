from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import warnings

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .embeddings import SupportsEmbedding
from .settings import IndexingConfig, safe_collection_component
from .sparse import SparseVectorData


@dataclass(frozen=True)
class PreparedPoint:
    chunk_id: str
    point_id: str
    embedding_text: str
    payload: dict[str, Any]
    content_hash: str


@dataclass(frozen=True)
class SyncFailure:
    chunk_id: str
    stage: str
    error: str


@dataclass(frozen=True)
class SyncStats:
    total_chunks: int
    to_process: int
    embedded: int
    skipped: int
    upserted: int
    failures: tuple[SyncFailure, ...]

    @property
    def failure_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True)
class CollectionVectorCapabilities:
    dense_vector_size: int
    dense_vector_name: str | None
    sparse_enabled: bool
    sparse_vector_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense_vector_size": self.dense_vector_size,
            "dense_vector_name": self.dense_vector_name,
            "sparse_enabled": self.sparse_enabled,
            "sparse_vector_names": list(self.sparse_vector_names),
        }


def _distance_enum(distance: str) -> qmodels.Distance:
    val = (distance or "").strip().lower()
    if val == "cosine":
        return qmodels.Distance.COSINE
    if val == "dot":
        return qmodels.Distance.DOT
    if val == "euclid":
        return qmodels.Distance.EUCLID
    if val == "manhattan":
        return qmodels.Distance.MANHATTAN
    raise ValueError(f"Unsupported qdrant distance: {distance!r}")


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _to_primitive(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(v) for v in value]
    return value


def build_collection_name(config: IndexingConfig, *, dataset_hash: str) -> str:
    if config.collection_name:
        return config.collection_name

    prefix = safe_collection_component(config.collection_prefix, max_len=24)
    ds = safe_collection_component(dataset_hash[:12], max_len=12)
    profile = safe_collection_component(config.chunking_profile.profile_id, max_len=16)
    model = safe_collection_component(config.embedding_model, max_len=20)
    return f"{prefix}_{ds}_{profile}_{model}"


def _dense_vector_from_config(
    vectors: Any,
) -> tuple[int, str | None]:
    if isinstance(vectors, qmodels.VectorParams):
        return int(vectors.size), None

    if isinstance(vectors, dict):
        preferred_order = ("dense", "default", "")
        for key in preferred_order:
            item = vectors.get(key)
            if item is not None and hasattr(item, "size"):
                return int(item.size), str(key) if key else None
        first_key = next(iter(vectors.keys()), None)
        if first_key is not None:
            first = vectors[first_key]
            if hasattr(first, "size"):
                name = str(first_key).strip() or None
                return int(first.size), name
    raise RuntimeError("Unable to infer dense vector config from collection")


def collection_vector_capabilities(
    client: QdrantClient, collection_name: str
) -> CollectionVectorCapabilities:
    info = client.get_collection(collection_name)
    vectors = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    dense_size, dense_name = _dense_vector_from_config(vectors)

    params = getattr(getattr(info, "config", None), "params", None)
    sparse_cfg = getattr(params, "sparse_vectors", None)
    sparse_names = tuple(sorted(sparse_cfg.keys())) if isinstance(sparse_cfg, dict) else tuple()

    return CollectionVectorCapabilities(
        dense_vector_size=dense_size,
        dense_vector_name=dense_name,
        sparse_enabled=bool(sparse_names),
        sparse_vector_names=sparse_names,
    )


def get_vector_size(client: QdrantClient, collection_name: str) -> int:
    caps = collection_vector_capabilities(client, collection_name)
    return int(caps.dense_vector_size)


def ensure_collection(
    client: QdrantClient,
    config: IndexingConfig,
    *,
    collection_name: str,
    vector_size: int,
) -> bool:
    collection_exists = bool(client.collection_exists(collection_name=collection_name))

    if collection_exists:
        caps = collection_vector_capabilities(client, collection_name)
        existing_size = caps.dense_vector_size
        if existing_size != int(vector_size):
            raise RuntimeError(
                "Existing collection vector size mismatch: "
                f"collection={collection_name!r}, existing={existing_size}, requested={vector_size}"
            )
        if config.sparse_enabled and config.sparse_vector_name not in set(caps.sparse_vector_names):
            msg = (
                "Existing collection does not expose requested sparse vector "
                f"{config.sparse_vector_name!r}. "
                "Run indexing on a fresh collection to enable hybrid sparse retrieval."
            )
            if config.strict_validation:
                raise RuntimeError(msg)
            warnings.warn(msg)
    else:
        sparse_cfg = None
        if config.sparse_enabled:
            sparse_cfg = {
                config.sparse_vector_name: qmodels.SparseVectorParams()
            }
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=int(vector_size),
                distance=_distance_enum(config.qdrant_distance),
            ),
            sparse_vectors_config=sparse_cfg,
            on_disk_payload=bool(config.qdrant_on_disk_payload),
            hnsw_config=qmodels.HnswConfigDiff(
                m=int(config.qdrant_hnsw_m),
                ef_construct=int(config.qdrant_hnsw_ef_construct),
            ),
        )

    # Idempotent payload indexes for common filters.
    field_schemas: list[tuple[str, Any]] = [
        ("chunk_id", qmodels.PayloadSchemaType.KEYWORD),
        ("law_id", qmodels.PayloadSchemaType.KEYWORD),
        ("article_id", qmodels.PayloadSchemaType.KEYWORD),
        ("index_views", qmodels.PayloadSchemaType.KEYWORD),
        ("law_status", qmodels.PayloadSchemaType.KEYWORD),
        ("article_is_abrogated", qmodels.PayloadSchemaType.BOOL),
        ("article_order_in_law", qmodels.PayloadSchemaType.INTEGER),
        ("passage_start_order", qmodels.PayloadSchemaType.INTEGER),
        ("passage_end_order", qmodels.PayloadSchemaType.INTEGER),
        ("article_chunk_order", qmodels.PayloadSchemaType.INTEGER),
        ("relation_types", qmodels.PayloadSchemaType.KEYWORD),
        ("law_year", qmodels.PayloadSchemaType.INTEGER),
    ]
    for field_name, schema in field_schemas:
        try:
            with warnings.catch_warnings():
                # In embedded mode Qdrant emits a UserWarning because payload indexes
                # are not used for optimization. Filters still work correctly, so
                # we silence only this known warning to keep notebook output clean.
                warnings.filterwarnings(
                    "ignore",
                    message="Payload indexes have no effect in the local Qdrant.*",
                    category=UserWarning,
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
        except Exception:
            # In practice Qdrant may raise if index already exists. Safe to ignore.
            pass

    return not collection_exists


def fetch_existing_content_hashes(
    client: QdrantClient, collection_name: str, point_ids: Sequence[str]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for batch in _chunks(point_ids, 256):
        records = client.retrieve(
            collection_name=collection_name,
            ids=list(batch),
            with_payload=["content_hash"],
            with_vectors=False,
        )
        for rec in records:
            rid = str(rec.id)
            payload = rec.payload or {}
            val = payload.get("content_hash")
            if isinstance(val, str) and val:
                out[rid] = val
    return out


def _upsert_vectors(
    client: QdrantClient,
    collection_name: str,
    batch_points: list[PreparedPoint],
    batch_vectors: list[list[float]],
    *,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
    sparse_vectors_by_chunk: dict[str, SparseVectorData] | None = None,
) -> None:
    to_upsert: list[qmodels.PointStruct] = []
    for point, vector in zip(batch_points, batch_vectors, strict=True):
        sparse = (
            sparse_vectors_by_chunk.get(point.chunk_id)
            if sparse_vectors_by_chunk is not None
            else None
        )
        vector_payload: Any
        needs_named_payload = dense_vector_name is not None or (
            sparse_vector_name is not None and sparse is not None and bool(sparse.indices)
        )
        if needs_named_payload:
            dense_key = dense_vector_name if dense_vector_name is not None else ""
            vector_payload = {dense_key: vector}
            if sparse_vector_name is not None and sparse is not None and sparse.indices:
                vector_payload[sparse_vector_name] = qmodels.SparseVector(
                    indices=list(sparse.indices),
                    values=list(sparse.values),
                )
        else:
            vector_payload = vector
        to_upsert.append(
            qmodels.PointStruct(id=point.point_id, vector=vector_payload, payload=point.payload)
        )
    with warnings.catch_warnings():
        # Embedded/local Qdrant warns when collection size grows >20k points.
        # This is expected in PoC local mode; keep output clean and actionable.
        warnings.filterwarnings(
            "ignore",
            message="Local mode is not recommended for collections with more than 20,000 points.*",
            category=UserWarning,
        )
        client.upsert(collection_name=collection_name, points=to_upsert, wait=True)


def sync_points_incremental(
    client: QdrantClient,
    *,
    collection_name: str,
    points: Sequence[PreparedPoint],
    embedder: SupportsEmbedding,
    force_reembed: bool,
    embed_batch_size: int,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
    sparse_vectors_by_chunk: dict[str, SparseVectorData] | None = None,
    upsert_batch_size: int = 64,
) -> SyncStats:
    chunk_ids_seen: set[str] = set()
    point_ids_seen: set[str] = set()
    for point in points:
        if point.chunk_id in chunk_ids_seen:
            raise ValueError(f"Duplicate chunk_id in prepared points: {point.chunk_id}")
        if point.point_id in point_ids_seen:
            raise ValueError(f"Duplicate point_id in prepared points: {point.point_id}")
        chunk_ids_seen.add(point.chunk_id)
        point_ids_seen.add(point.point_id)

    existing_hashes = fetch_existing_content_hashes(
        client, collection_name, [p.point_id for p in points]
    )

    to_process: list[PreparedPoint] = []
    skipped = 0
    for p in points:
        existing_hash = existing_hashes.get(p.point_id)
        if not force_reembed and existing_hash == p.content_hash:
            skipped += 1
            continue
        to_process.append(p)

    failures: list[SyncFailure] = []
    embedded = 0
    upserted = 0

    for batch in _chunks(to_process, max(1, int(embed_batch_size))):
        batch_list = list(batch)
        texts = [p.embedding_text for p in batch_list]

        try:
            vectors = embedder.embed_texts(texts)
            if len(vectors) != len(batch_list):
                raise RuntimeError(
                    "Embedding backend returned unexpected vector count "
                    f"(expected={len(batch_list)}, got={len(vectors)})"
                )
        except Exception as exc:
            if len(batch_list) == 1:
                failures.append(
                    SyncFailure(
                        chunk_id=batch_list[0].chunk_id,
                        stage="embedding",
                        error=str(exc),
                    )
                )
                continue
            # Fallback to single-item embedding to isolate faulty chunk(s).
            for point in batch_list:
                try:
                    vec = embedder.embed_texts([point.embedding_text])
                    if len(vec) != 1:
                        raise RuntimeError("Embedding backend returned unexpected vector count for single item")
                    _upsert_vectors(
                        client,
                        collection_name,
                        [point],
                        vec,
                        dense_vector_name=dense_vector_name,
                        sparse_vector_name=sparse_vector_name,
                        sparse_vectors_by_chunk=sparse_vectors_by_chunk,
                    )
                    embedded += 1
                    upserted += 1
                except Exception as inner_exc:
                    failures.append(
                        SyncFailure(
                            chunk_id=point.chunk_id,
                            stage="embedding_or_upsert",
                            error=str(inner_exc),
                        )
                    )
            continue

        # Vectorization succeeded for the whole batch.
        embedded += len(batch_list)
        try:
            for upsert_batch_points, upsert_batch_vectors in zip(
                _chunks(batch_list, upsert_batch_size),
                _chunks(vectors, upsert_batch_size),
                strict=True,
            ):
                _upsert_vectors(
                    client,
                    collection_name,
                    list(upsert_batch_points),
                    [list(v) for v in upsert_batch_vectors],
                    dense_vector_name=dense_vector_name,
                    sparse_vector_name=sparse_vector_name,
                    sparse_vectors_by_chunk=sparse_vectors_by_chunk,
                )
                upserted += len(upsert_batch_points)
        except Exception as exc:
            if len(batch_list) == 1:
                failures.append(
                    SyncFailure(
                        chunk_id=batch_list[0].chunk_id,
                        stage="upsert",
                        error=str(exc),
                    )
                )
                continue
            # Fallback to single upserts.
            for point, vector in zip(batch_list, vectors, strict=True):
                try:
                    _upsert_vectors(
                        client,
                        collection_name,
                        [point],
                        [vector],
                        dense_vector_name=dense_vector_name,
                        sparse_vector_name=sparse_vector_name,
                        sparse_vectors_by_chunk=sparse_vectors_by_chunk,
                    )
                    upserted += 1
                except Exception as inner_exc:
                    failures.append(
                        SyncFailure(
                            chunk_id=point.chunk_id,
                            stage="upsert",
                            error=str(inner_exc),
                        )
                    )

    return SyncStats(
        total_chunks=len(points),
        to_process=len(to_process),
        embedded=embedded,
        skipped=skipped,
        upserted=upserted,
        failures=tuple(failures),
    )


def collection_stats(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    info = client.get_collection(collection_name)
    count = client.count(collection_name=collection_name, exact=True)
    return {
        "collection_name": collection_name,
        "points_count_exact": int(count.count),
        "collection_info": _to_primitive(info),
    }


def validate_no_duplicate_chunk_ids(
    client: QdrantClient, collection_name: str
) -> dict[str, Any]:
    seen: set[str] = set()
    duplicates: list[str] = []
    offset: Any = None
    total = 0

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        if not records:
            break
        for rec in records:
            total += 1
            chunk_id = str((rec.payload or {}).get("chunk_id") or "")
            if not chunk_id:
                continue
            if chunk_id in seen:
                duplicates.append(chunk_id)
            else:
                seen.add(chunk_id)
        if next_offset is None:
            break
        offset = next_offset

    return {
        "total_points_scanned": total,
        "duplicate_chunk_ids": sorted(set(duplicates)),
        "duplicate_count": len(set(duplicates)),
        "ok": len(duplicates) == 0,
    }


def validate_filtered_query(
    client: QdrantClient,
    *,
    collection_name: str,
    sample_point_id: str,
    law_id: str,
) -> dict[str, Any]:
    records = client.retrieve(
        collection_name=collection_name,
        ids=[sample_point_id],
        with_payload=False,
        with_vectors=True,
    )
    if not records:
        return {
            "ok": False,
            "reason": f"Sample point {sample_point_id!r} not found",
            "matches": [],
        }

    vector_obj: Any = records[0].vector
    vector: list[float] | None = None
    if isinstance(vector_obj, dict):
        # Prefer unnamed dense key, then any dense list-like value.
        for key in ("", "dense", "default"):
            cur = vector_obj.get(key)
            if isinstance(cur, list) and cur:
                vector = [float(x) for x in cur]
                break
        if vector is None:
            for cur in vector_obj.values():
                if isinstance(cur, list) and cur:
                    vector = [float(x) for x in cur]
                    break
    elif isinstance(vector_obj, list):
        vector = [float(x) for x in vector_obj]

    if vector is None or not vector:
        return {
            "ok": False,
            "reason": "Cannot run filter validation query: sample vector missing",
            "matches": [],
        }

    query_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="index_views", match=qmodels.MatchValue(value="current")
            ),
            qmodels.FieldCondition(
                key="law_status", match=qmodels.MatchValue(value="current")
            ),
            qmodels.FieldCondition(key="law_id", match=qmodels.MatchValue(value=law_id)),
        ]
    )

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        query_filter=query_filter,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    points = response.points if hasattr(response, "points") else []
    violations: list[str] = []
    matches: list[dict[str, Any]] = []

    for p in points:
        payload = p.payload or {}
        matches.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "law_id": payload.get("law_id"),
                "index_views": payload.get("index_views"),
                "law_status": payload.get("law_status"),
                "score": float(getattr(p, "score", 0.0)),
            }
        )
        idx_views = payload.get("index_views") or []
        if "current" not in idx_views:
            violations.append(f"chunk_id={payload.get('chunk_id')} missing current index_view")
        if payload.get("law_status") != "current":
            violations.append(f"chunk_id={payload.get('chunk_id')} law_status != current")
        if payload.get("law_id") != law_id:
            violations.append(f"chunk_id={payload.get('chunk_id')} law_id mismatch")

    return {
        "ok": len(violations) == 0,
        "query_law_id": law_id,
        "sample_point_id": sample_point_id,
        "matches": matches,
        "violations": violations,
    }


__all__ = [
    "PreparedPoint",
    "SyncFailure",
    "SyncStats",
    "CollectionVectorCapabilities",
    "build_collection_name",
    "get_vector_size",
    "collection_vector_capabilities",
    "ensure_collection",
    "sync_points_incremental",
    "collection_stats",
    "validate_no_duplicate_chunk_ids",
    "validate_filtered_query",
]

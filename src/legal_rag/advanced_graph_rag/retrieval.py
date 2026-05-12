"""Advanced Qdrant retrieval helpers for graph-aware RAG."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http import models as qmodels

from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.io import read_json, read_jsonl
from legal_rag.simple_rag.models import RetrievedChunkRecord
from legal_rag.simple_rag.retrieval import build_static_filter, resolve_index_manifest_path

from .models import AdvancedRagConfig, GraphRelationUsed

_DENSE_VECTOR_NAME_CACHE: dict[tuple[int, str], str | None] = {}
_SPARSE_VECTOR_NAME_CACHE: dict[tuple[int, str], str | None] = {}
T = TypeVar("T")


def load_index_manifest(config: AdvancedRagConfig) -> tuple[Path, dict[str, Any]]:
    """Load the resolved index manifest."""
    path = resolve_index_manifest_path(config.index_manifest_path)
    return path, read_json(path)


def load_simple_rag_manifest(config: AdvancedRagConfig) -> tuple[Path, dict[str, Any]]:
    """Load the referenced simple RAG manifest."""
    path = Path(config.simple_rag_manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Simple RAG manifest not found: {path}")
    return path, read_json(path)


def resolve_collection_name(config: AdvancedRagConfig, index_manifest: dict[str, Any]) -> str:
    """Resolve collection name, using the manifest unless config explicitly overrides it."""
    if "collection_name" not in config.model_fields_set and index_manifest.get("collection_name"):
        return str(index_manifest["collection_name"])
    return config.collection_name


def connect_qdrant(config: AdvancedRagConfig, index_manifest: dict[str, Any]) -> QdrantClient:
    """Create a Qdrant client for the indexed collection."""
    target = resolve_qdrant_target(config, index_manifest)
    qdrant_url = str(target.get("url") or "").strip()
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(path=str(target["path"]))


def resolve_qdrant_target(config: AdvancedRagConfig, index_manifest: dict[str, Any]) -> dict[str, str | None]:
    """Resolve the Qdrant target declared by the index manifest."""
    manifest_config = dict(index_manifest.get("config") or {})
    qdrant_info = dict(index_manifest.get("qdrant") or {})
    qdrant_url = str(
        index_manifest.get("qdrant_url") or qdrant_info.get("url") or manifest_config.get("qdrant_url") or ""
    ).strip()
    qdrant_path = str(
        index_manifest.get("qdrant_path") or qdrant_info.get("path") or manifest_config.get("index_dir") or config.index_dir
    )
    return {
        "mode": "server" if qdrant_url else "local_path",
        "url": qdrant_url or None,
        "path": qdrant_path,
    }


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
    vector_name = dense_vector_name(client, collection_name=collection_name)
    response = _qdrant_call(
        lambda: client.query_points(
            collection_name=collection_name,
            query=vector,
            using=vector_name,
            query_filter=build_static_filter(static_filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    )
    return [_point_to_chunk(point) for point in response.points]


def search_hybrid(
    client: QdrantClient,
    *,
    collection_name: str,
    embedder: SupportsEmbedding,
    query_text: str,
    limit: int,
    rrf_k: int,
    static_filters: dict[str, Any],
    index_manifest: dict[str, Any],
) -> list[RetrievedChunkRecord]:
    """Run dense+sparse retrieval fused by Qdrant RRF."""
    dense_name = dense_vector_name(client, collection_name=collection_name)
    sparse_name = sparse_vector_name(client, collection_name=collection_name)
    if sparse_name is None or manifest_disables_sparse(index_manifest):
        raise RuntimeError("Hybrid retrieval requires an index with sparse vectors, but none was declared or found")

    dense_vector = embedder.embed_texts([query_text])[0]
    sparse_vector = embed_sparse_query(embedder, query_text)
    qdrant_filter = build_static_filter(static_filters)
    response = _qdrant_call(
        lambda: client.query_points(
            collection_name=collection_name,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_vector,
                    using=dense_name,
                    filter=qdrant_filter,
                    limit=limit,
                ),
                qmodels.Prefetch(
                    query=sparse_vector,
                    using=sparse_name,
                    filter=qdrant_filter,
                    limit=limit,
                ),
            ],
            query=qmodels.RrfQuery(rrf=qmodels.Rrf(k=int(rrf_k))),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    )
    return [_point_to_chunk(point) for point in response.points]


def fetch_law_chunks(
    client: QdrantClient | None,
    *,
    collection_name: str,
    law_id: str,
    static_filters: dict[str, Any],
    limit: int,
    article_label_norm: str | None = None,
    allowed_chunk_ids: set[str] | None = None,
    graph: "GraphIndex | None" = None,
) -> list[RetrievedChunkRecord]:
    """Fetch chunks for one law via Qdrant payload filters."""
    if client is None:
        if graph is None:
            return []
        return fetch_law_chunks_from_graph(
            graph,
            law_id=law_id,
            static_filters=static_filters,
            limit=limit,
            article_label_norm=article_label_norm,
            allowed_chunk_ids=allowed_chunk_ids,
        )
    filters = dict(static_filters)
    filters["law_id"] = law_id
    if article_label_norm:
        filters["article_label_norm"] = article_label_norm
    selected: list[RetrievedChunkRecord] = []
    seen: set[str] = set()
    offset: Any = None
    page_limit = max(limit, min(64, limit * 4))
    while len(selected) < limit:
        records, offset = _qdrant_call(
            lambda: client.scroll(
                collection_name=collection_name,
                scroll_filter=build_static_filter(filters),
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        )
        if not records:
            break
        for record in records:
            chunk = _point_to_chunk(record, score=0.0)
            if allowed_chunk_ids is not None and chunk.chunk_id not in allowed_chunk_ids:
                continue
            if chunk.chunk_id in seen:
                continue
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            if len(selected) >= limit:
                break
        if offset is None:
            break
    return selected


def fetch_law_chunks_from_graph(
    graph: "GraphIndex",
    *,
    law_id: str,
    static_filters: dict[str, Any],
    limit: int,
    article_label_norm: str | None = None,
    allowed_chunk_ids: set[str] | None = None,
) -> list[RetrievedChunkRecord]:
    """Fetch graph-expanded chunks from the clean chunks snapshot."""
    if article_label_norm:
        chunk_ids = graph.chunk_ids_by_law_article.get((law_id, article_label_norm), [])
    else:
        chunk_ids = graph.chunk_ids_by_law.get(law_id, [])
    selected: list[RetrievedChunkRecord] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
            continue
        if chunk_id in seen:
            continue
        payload = graph.chunk_payload_by_id.get(chunk_id)
        if payload is None or not _payload_matches_filters(payload, static_filters):
            continue
        selected.append(_payload_to_chunk(payload, score=0.0))
        seen.add(chunk_id)
        if len(selected) >= limit:
            break
    return selected


def rank_law_chunks(
    client: QdrantClient,
    *,
    collection_name: str,
    query_vector: list[float],
    law_id: str,
    static_filters: dict[str, Any],
    limit: int,
    article_label_norm: str | None = None,
    allowed_chunk_ids: set[str] | None = None,
) -> list[RetrievedChunkRecord]:
    """Rank chunks inside a target law using the same dense query vector."""
    filters = dict(static_filters)
    filters["law_id"] = law_id
    if article_label_norm:
        filters["article_label_norm"] = article_label_norm
    vector_name = dense_vector_name(client, collection_name=collection_name)
    response = _qdrant_call(
        lambda: client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=build_static_filter(filters),
            limit=max(limit, min(64, limit * 4)),
            with_payload=True,
            with_vectors=False,
        )
    )
    selected: list[RetrievedChunkRecord] = []
    seen: set[str] = set()
    for point in response.points:
        chunk = _point_to_chunk(point)
        if allowed_chunk_ids is not None and chunk.chunk_id not in allowed_chunk_ids:
            continue
        if chunk.chunk_id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        if len(selected) >= limit:
            break
    return selected


class GraphIndex:
    """Explicit law graph and clean chunks loaded from step 01 outputs."""

    def __init__(self, *, edges: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
        self.edges_by_source: dict[str, list[dict[str, Any]]] = {}
        self.chunk_ids_by_law: dict[str, list[str]] = {}
        self.chunk_ids_by_law_article: dict[tuple[str, str], list[str]] = {}
        self.chunk_payload_by_id: dict[str, dict[str, Any]] = {}
        self.edge_triples: set[tuple[str, str, str]] = set()
        for edge in edges:
            src = str(edge.get("src_law_id") or "")
            dst = str(edge.get("dst_law_id") or "")
            relation_type = str(edge.get("relation_type") or "").strip().upper()
            if not src or not dst or not relation_type:
                continue
            self.edges_by_source.setdefault(src, []).append(edge)
            self.edge_triples.add((src, dst, relation_type))
        for chunk in chunks:
            law_id = str(chunk.get("law_id") or "")
            chunk_id = str(chunk.get("chunk_id") or "")
            if law_id and chunk_id:
                self.chunk_payload_by_id[chunk_id] = dict(chunk)
                self.chunk_ids_by_law.setdefault(law_id, []).append(chunk_id)
                article_label_norm = str(chunk.get("article_label_norm") or "").strip()
                if article_label_norm:
                    self.chunk_ids_by_law_article.setdefault((law_id, article_label_norm), []).append(chunk_id)

    @classmethod
    def from_dir(cls, laws_dir: str | Path) -> "GraphIndex":
        """Load explicit edges and chunk ids from the clean legal dataset."""
        root = Path(laws_dir)
        return cls(edges=read_jsonl(root / "edges.jsonl"), chunks=read_jsonl(root / "chunks.jsonl"))


def expand_with_graph(
    client: QdrantClient | None,
    *,
    collection_name: str,
    graph: GraphIndex,
    seeds: Sequence[RetrievedChunkRecord],
    relation_types: Sequence[str],
    static_filters: dict[str, Any],
    max_chunks_per_law: int,
    embedder: SupportsEmbedding | None = None,
    query_text: str = "",
    max_chunks_total: int | None = None,
    min_edge_confidence: float = 0.0,
) -> tuple[list[RetrievedChunkRecord], list[GraphRelationUsed]]:
    """Expand retrieved candidates through explicit source-law graph edges."""
    allowed = {str(value).strip().upper() for value in relation_types}
    seed_law_ids = _unique(str(chunk.payload.get("law_id") or "") for chunk in seeds)
    if not seed_law_ids:
        return [], []
    query_vector = embedder.embed_texts([query_text])[0] if client is not None and embedder is not None and query_text else None
    effective_max_total = max_chunks_total or max_chunks_per_law * max(1, len(seed_law_ids))
    chunks: list[RetrievedChunkRecord] = []
    relations: list[GraphRelationUsed] = []
    seen_triples: set[tuple[str, str, str]] = set()
    seen_chunk_ids: set[str] = set()
    accepted_by_target_law: dict[str, int] = {}
    for source_law_id in seed_law_ids:
        for edge in graph.edges_by_source.get(source_law_id, []):
            if len(chunks) >= effective_max_total:
                return chunks, relations
            target_law_id = str(edge.get("dst_law_id") or "")
            relation_type = str(edge.get("relation_type") or "").strip().upper()
            if not target_law_id or relation_type not in allowed:
                continue
            if _edge_confidence(edge) < min_edge_confidence:
                continue
            triple = (source_law_id, target_law_id, relation_type)
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            article_label_norm = str(edge.get("dst_article_label_norm") or "").strip() or None
            allowed_chunk_ids = _allowed_target_chunk_ids(graph, target_law_id, article_label_norm)
            remaining = min(
                max_chunks_per_law - accepted_by_target_law.get(target_law_id, 0),
                effective_max_total - len(chunks),
            )
            if not allowed_chunk_ids or remaining <= 0:
                continue
            fetched: list[RetrievedChunkRecord] = []
            if query_vector is not None:
                try:
                    fetched = rank_law_chunks(
                        client,
                        collection_name=collection_name,
                        query_vector=query_vector,
                        law_id=target_law_id,
                        static_filters=static_filters,
                        limit=remaining,
                        article_label_norm=article_label_norm,
                        allowed_chunk_ids=allowed_chunk_ids,
                    )
                except ResponseHandlingException:
                    fetched = []
            if len(fetched) < remaining:
                try:
                    fallback = fetch_law_chunks(
                        client,
                        collection_name=collection_name,
                        law_id=target_law_id,
                        static_filters=static_filters,
                        limit=remaining,
                        article_label_norm=article_label_norm,
                        allowed_chunk_ids=allowed_chunk_ids,
                        graph=graph,
                    )
                except ResponseHandlingException:
                    fallback = fetch_law_chunks_from_graph(
                        graph,
                        law_id=target_law_id,
                        static_filters=static_filters,
                        limit=remaining,
                        article_label_norm=article_label_norm,
                        allowed_chunk_ids=allowed_chunk_ids,
                    )
                fetched = _dedupe_chunks([*fetched, *fallback])[:remaining]
            new_chunks = [chunk for chunk in fetched if chunk.chunk_id not in seen_chunk_ids]
            if not new_chunks:
                continue
            chunks.extend(new_chunks)
            seen_chunk_ids.update(chunk.chunk_id for chunk in new_chunks)
            accepted_by_target_law[target_law_id] = accepted_by_target_law.get(target_law_id, 0) + len(new_chunks)
            relations.append(
                GraphRelationUsed(
                    source_law_id=source_law_id,
                    target_law_id=target_law_id,
                    relation_type=relation_type,
                )
            )
    return chunks, relations


def _allowed_target_chunk_ids(graph: GraphIndex, law_id: str, article_label_norm: str | None) -> set[str]:
    if article_label_norm:
        article_chunk_ids = set(graph.chunk_ids_by_law_article.get((law_id, article_label_norm), []))
        if article_chunk_ids:
            return article_chunk_ids
    return set(graph.chunk_ids_by_law.get(law_id, []))


def _edge_confidence(edge: dict[str, Any]) -> float:
    raw = edge.get("confidence")
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def dense_vector_name(client: QdrantClient, *, collection_name: str) -> str | None:
    """Return the dense vector name, or None for unnamed-vector collections."""
    cache_key = (id(client), collection_name)
    if cache_key in _DENSE_VECTOR_NAME_CACHE:
        return _DENSE_VECTOR_NAME_CACHE[cache_key]
    info = _qdrant_call(lambda: client.get_collection(collection_name=collection_name))
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    vector_name: str | None = None
    if isinstance(vectors, dict):
        if "dense" in vectors:
            vector_name = "dense"
        elif vectors:
            vector_name = str(next(iter(vectors)))
    _DENSE_VECTOR_NAME_CACHE[cache_key] = vector_name
    return vector_name


def sparse_vector_name(client: QdrantClient, *, collection_name: str) -> str | None:
    """Return the sparse vector name if the collection has one."""
    cache_key = (id(client), collection_name)
    if cache_key in _SPARSE_VECTOR_NAME_CACHE:
        return _SPARSE_VECTOR_NAME_CACHE[cache_key]
    info = _qdrant_call(lambda: client.get_collection(collection_name=collection_name))
    sparse = getattr(getattr(getattr(info, "config", None), "params", None), "sparse_vectors", None)
    vector_name: str | None = None
    if isinstance(sparse, dict):
        if "sparse" in sparse:
            vector_name = "sparse"
        elif sparse:
            vector_name = str(next(iter(sparse)))
    _SPARSE_VECTOR_NAME_CACHE[cache_key] = vector_name
    return vector_name


def manifest_disables_sparse(index_manifest: dict[str, Any]) -> bool:
    """Return True when the manifest explicitly says hybrid/sparse is disabled."""
    values = [
        index_manifest.get("hybrid_enabled"),
        (index_manifest.get("config") or {}).get("hybrid_enabled") if isinstance(index_manifest.get("config"), dict) else None,
        (index_manifest.get("vectors") or {}).get("sparse_enabled") if isinstance(index_manifest.get("vectors"), dict) else None,
    ]
    return any(value is False for value in values)


def _qdrant_call(operation: Callable[[], T], *, attempts: int = 5) -> T:
    """Retry transient Qdrant HTTP disconnects from long diagnostic sweeps."""
    last_error: ResponseHandlingException | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except ResponseHandlingException as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def embed_sparse_query(embedder: SupportsEmbedding, query_text: str) -> qmodels.SparseVector:
    """Embed one query using a sparse embedding method exposed by the query embedder."""
    for method_name in ("embed_sparse_texts", "sparse_embed_texts"):
        method = getattr(embedder, method_name, None)
        if callable(method):
            rows = method([query_text])
            if not isinstance(rows, list) or len(rows) != 1:
                raise RuntimeError(f"{method_name} must return one sparse vector per input")
            return _coerce_sparse_vector(rows[0])
    raise RuntimeError("Hybrid retrieval requires an embedder with embed_sparse_texts() or sparse_embed_texts()")


def _coerce_sparse_vector(value: Any) -> qmodels.SparseVector:
    if isinstance(value, qmodels.SparseVector):
        return value
    if isinstance(value, dict):
        return qmodels.SparseVector(
            indices=[int(item) for item in value.get("indices", [])],
            values=[float(item) for item in value.get("values", [])],
        )
    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if indices is not None and values is not None:
        return qmodels.SparseVector(indices=[int(item) for item in indices], values=[float(item) for item in values])
    raise RuntimeError(f"Unsupported sparse vector shape: {type(value).__name__}")


def _point_to_chunk(point: Any, *, score: float | None = None) -> RetrievedChunkRecord:
    payload = dict(point.payload or {})
    return _payload_to_chunk(
        payload,
        score=float(score if score is not None else getattr(point, "score", 0.0)),
        fallback_id=point.id,
    )


def _payload_to_chunk(payload: dict[str, Any], *, score: float, fallback_id: Any | None = None) -> RetrievedChunkRecord:
    return RetrievedChunkRecord(
        chunk_id=str(payload.get("chunk_id") or fallback_id or ""),
        score=score,
        text=str(payload.get("text") or ""),
        payload=payload,
    )


def _payload_matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = payload.get(key)
        expected_values = list(expected) if isinstance(expected, (list, tuple, set)) else [expected]
        if isinstance(actual, list):
            if not any(value in actual for value in expected_values):
                return False
        elif actual not in expected_values:
            return False
    return True


def _dedupe_chunks(chunks: list[RetrievedChunkRecord]) -> list[RetrievedChunkRecord]:
    out: list[RetrievedChunkRecord] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        out.append(chunk)
        seen.add(chunk.chunk_id)
    return out


def _unique(values: Sequence[str] | Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out

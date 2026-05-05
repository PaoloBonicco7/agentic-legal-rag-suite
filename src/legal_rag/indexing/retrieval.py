"""Diagnostic retrieval helpers for validating the Qdrant index contract."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .embeddings import SupportsEmbedding, supports_sparse_embedding
from .qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


class RetrievedChunk(BaseModel):
    """One chunk returned by diagnostic Qdrant retrieval."""

    chunk_id: str
    score: float
    text: str
    payload: dict[str, Any]


def _conditions(key: str, values: Sequence[str] | None) -> list[qmodels.FieldCondition]:
    if not values:
        return []
    return [qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=[str(value) for value in values]))]


def build_qdrant_filter(
    *,
    law_ids: Sequence[str] | None = None,
    law_status: str | None = None,
    index_view: str | None = None,
    article_ids: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
) -> qmodels.Filter | None:
    """Build a Qdrant filter for the metadata fields guaranteed by the contract."""
    must: list[Any] = []
    must.extend(_conditions("law_id", law_ids))
    must.extend(_conditions("article_id", article_ids))
    must.extend(_conditions("relation_types", relation_types))
    if law_status:
        must.append(qmodels.FieldCondition(key="law_status", match=qmodels.MatchValue(value=law_status)))
    if index_view:
        must.append(qmodels.FieldCondition(key="index_views", match=qmodels.MatchValue(value=index_view)))
    return qmodels.Filter(must=must) if must else None


def search_index(
    client: QdrantClient,
    *,
    collection_name: str,
    embedder: SupportsEmbedding,
    query: str,
    limit: int = 5,
    law_ids: Sequence[str] | None = None,
    law_status: str | None = None,
    index_view: str | None = None,
    article_ids: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    retrieval_mode: str = "dense",
) -> list[RetrievedChunk]:
    """Embed a query and run a diagnostic filtered search against Qdrant."""
    qdrant_filter = build_qdrant_filter(
        law_ids=law_ids,
        law_status=law_status,
        index_view=index_view,
        article_ids=article_ids,
        relation_types=relation_types,
    )
    vector = embedder.embed_texts([query])[0]
    if retrieval_mode == "hybrid":
        if not supports_sparse_embedding(embedder):
            raise RuntimeError("Hybrid diagnostic search requires an embedder with embed_sparse_texts().")
        indices, values = embedder.embed_sparse_texts([query])[0]  # type: ignore[attr-defined]
        response = client.query_points(
            collection_name=collection_name,
            prefetch=[
                qmodels.Prefetch(query=vector, using=DENSE_VECTOR_NAME, filter=qdrant_filter, limit=limit * 2),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=limit * 2,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            using=DENSE_VECTOR_NAME,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    out: list[RetrievedChunk] = []
    for point in response.points:
        payload = point.payload or {}
        out.append(
            RetrievedChunk(
                chunk_id=str(payload.get("chunk_id") or point.id),
                score=float(point.score),
                text=str(payload.get("text") or ""),
                payload=dict(payload),
            )
        )
    return out

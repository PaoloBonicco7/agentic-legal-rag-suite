from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_indexing.embeddings import SupportsEmbedding
from legal_indexing.sparse import SparseEncoder

from .config import (
    AdvancedHybridConfig,
    QdrantPayloadFieldMap,
    ThresholdDirection,
)


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return tuple()
    out: list[str] = []
    for item in value:
        cur = _as_str(item)
        if cur is not None:
            out.append(cur)
    return tuple(out)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    payload: dict[str, Any]
    point_id: str

    @property
    def law_id(self) -> str | None:
        return _as_str(self.payload.get("law_id"))

    @property
    def article_id(self) -> str | None:
        return _as_str(self.payload.get("article_id"))

    @property
    def source_chunk_ids(self) -> tuple[str, ...]:
        return _as_str_list(self.payload.get("source_chunk_ids"))

    @property
    def source_passage_ids(self) -> tuple[str, ...]:
        return _as_str_list(self.payload.get("source_passage_ids"))

    @property
    def index_views(self) -> tuple[str, ...]:
        return _as_str_list(self.payload.get("index_views"))

    @property
    def law_status(self) -> str | None:
        return _as_str(self.payload.get("law_status"))

    @property
    def law_date(self) -> str | None:
        return _as_str(self.payload.get("law_date"))


@dataclass(frozen=True)
class RetrievalBatchResult:
    name: str
    query: str
    top_k: int
    retrieved: tuple[RetrievedChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query": self.query,
            "top_k": self.top_k,
            "retrieved_chunk_ids": [x.chunk_id for x in self.retrieved],
            "retrieved_count": len(self.retrieved),
        }


@dataclass(frozen=True)
class HybridRetrievalResult:
    retrieved: tuple[RetrievedChunk, ...]
    dense_retrieved: tuple[RetrievedChunk, ...]
    sparse_retrieved: tuple[RetrievedChunk, ...]
    retrieval_mode: str
    overlap_count: int
    added_by_sparse: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_count": len(self.retrieved),
            "dense_retrieved_count": len(self.dense_retrieved),
            "sparse_retrieved_count": len(self.sparse_retrieved),
            "retrieval_mode": self.retrieval_mode,
            "overlap_count": self.overlap_count,
            "added_by_sparse": self.added_by_sparse,
        }


@dataclass(frozen=True)
class PayloadSchemaInspection:
    collection_name: str
    sample_size: int
    inspected_points: int
    field_presence: dict[str, int]
    required_fields: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    sample_payload_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "sample_size": self.sample_size,
            "inspected_points": self.inspected_points,
            "field_presence": dict(self.field_presence),
            "required_fields": list(self.required_fields),
            "missing_required_fields": list(self.missing_required_fields),
            "sample_payload_keys": list(self.sample_payload_keys),
        }


def _has_value(payload: dict[str, Any], field_name: str) -> bool:
    if field_name not in payload:
        return False
    value = payload[field_name]
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def introspect_payload_schema(
    client: QdrantClient,
    *,
    collection_name: str,
    field_map: QdrantPayloadFieldMap,
    sample_size: int = 128,
) -> PayloadSchemaInspection:
    sample_size = max(1, int(sample_size))
    required_fields = field_map.required_fields()
    field_presence: dict[str, int] = {key: 0 for key in required_fields}
    payload_keys: set[str] = set()

    inspected = 0
    offset: Any = None
    while inspected < sample_size:
        limit = min(256, sample_size - inspected)
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        for rec in records:
            payload = dict(rec.payload or {})
            payload_keys.update(payload.keys())
            inspected += 1
            for field_name in required_fields:
                if _has_value(payload, field_name):
                    field_presence[field_name] += 1
            if inspected >= sample_size:
                break

        if next_offset is None:
            break
        offset = next_offset

    missing = tuple(sorted([k for k, v in field_presence.items() if v == 0]))
    return PayloadSchemaInspection(
        collection_name=collection_name,
        sample_size=sample_size,
        inspected_points=inspected,
        field_presence=field_presence,
        required_fields=required_fields,
        missing_required_fields=missing,
        sample_payload_keys=tuple(sorted(payload_keys)),
    )


def assert_required_payload_fields(inspection: PayloadSchemaInspection) -> None:
    if inspection.inspected_points <= 0:
        raise RuntimeError(
            f"Collection {inspection.collection_name!r} has zero points; cannot introspect payload."
        )
    if not inspection.missing_required_fields:
        return
    missing = ", ".join(inspection.missing_required_fields)
    raise RuntimeError(
        "Qdrant payload is missing required fields for RAG runtime: "
        f"{missing}. Check `src/legal_indexing/metadata.py` and the indexing pipeline."
    )


def build_view_filter(
    field_map: QdrantPayloadFieldMap, view: str
) -> qmodels.Filter | None:
    view_norm = (view or "").strip().lower()
    if view_norm in {"", "none"}:
        return None
    if view_norm == "current":
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=field_map.index_views, match=qmodels.MatchValue(value="current")
                ),
                qmodels.FieldCondition(
                    key=field_map.law_status, match=qmodels.MatchValue(value="current")
                ),
            ]
        )
    if view_norm == "historical":
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=field_map.index_views,
                    match=qmodels.MatchValue(value="historical"),
                )
            ]
        )
    raise ValueError(f"Unsupported view filter: {view!r}")


def build_law_filter(
    field_map: QdrantPayloadFieldMap, law_ids: Sequence[str]
) -> qmodels.Filter | None:
    values = [v for v in [str(x).strip() for x in law_ids] if v]
    if not values:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=field_map.law_id,
                match=qmodels.MatchAny(any=values),
            )
        ]
    )


def build_article_filter(
    field_map: QdrantPayloadFieldMap, article_ids: Sequence[str]
) -> qmodels.Filter | None:
    values = [v for v in [str(x).strip() for x in article_ids] if v]
    if not values:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=field_map.article_id,
                match=qmodels.MatchAny(any=values),
            )
        ]
    )


def build_law_status_filter(
    field_map: QdrantPayloadFieldMap, law_status: str | None
) -> qmodels.Filter | None:
    status = str(law_status or "").strip().lower()
    if not status:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=field_map.law_status,
                match=qmodels.MatchValue(value=status),
            )
        ]
    )


def build_relation_type_filter(
    field_map: QdrantPayloadFieldMap, relation_types: Sequence[str]
) -> qmodels.Filter | None:
    values = [str(x).strip().upper() for x in relation_types if str(x).strip()]
    if not values:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=field_map.relation_types,
                match=qmodels.MatchAny(any=values),
            )
        ]
    )


def build_law_date_filter(
    field_map: QdrantPayloadFieldMap,
    *,
    year_from: int | None,
    year_to: int | None,
) -> qmodels.Filter | None:
    if year_from is None and year_to is None:
        return None

    lower = f"{int(year_from):04d}-01-01" if year_from is not None else None
    upper = f"{int(year_to):04d}-12-31" if year_to is not None else None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=field_map.law_date,
                range=qmodels.DatetimeRange(gte=lower, lte=upper),
            )
        ]
    )


def build_chunk_id_filter(
    field_map: QdrantPayloadFieldMap, chunk_ids: Sequence[str]
) -> qmodels.Filter | None:
    values = [v for v in [str(x).strip() for x in chunk_ids] if v]
    if not values:
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key=field_map.chunk_id,
                match=qmodels.MatchAny(any=values),
            )
        ]
    )


def merge_filters(
    left: qmodels.Filter | None, right: qmodels.Filter | None
) -> qmodels.Filter | None:
    if left is None:
        return right
    if right is None:
        return left
    return qmodels.Filter(
        must=[*(left.must or []), *(right.must or [])],
        should=[*(left.should or []), *(right.should or [])] or None,
        must_not=[*(left.must_not or []), *(right.must_not or [])] or None,
        min_should=left.min_should or right.min_should,
    )


def merge_retrieved(
    primary: Sequence[RetrievedChunk], secondary: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for item in list(primary) + list(secondary):
        key = item.chunk_id.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _score_gate(
    docs: list[RetrievedChunk],
    *,
    score_threshold: float | None,
    threshold_direction: ThresholdDirection | None,
) -> list[RetrievedChunk]:
    if score_threshold is None:
        return docs
    if threshold_direction == "gte":
        return [d for d in docs if d.score >= score_threshold]
    if threshold_direction == "lte":
        return [d for d in docs if d.score <= score_threshold]
    raise ValueError(
        "score_threshold is set but threshold_direction is invalid. Expected 'gte' or 'lte'."
    )


def _rrf_fuse(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    *,
    k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[RetrievedChunk]:
    if not dense and not sparse:
        return []

    dense_rank = {doc.chunk_id: idx for idx, doc in enumerate(dense, start=1)}
    sparse_rank = {doc.chunk_id: idx for idx, doc in enumerate(sparse, start=1)}

    by_chunk: dict[str, RetrievedChunk] = {}
    for doc in dense + sparse:
        by_chunk.setdefault(doc.chunk_id, doc)

    rows: list[tuple[str, float]] = []
    for chunk_id in by_chunk.keys():
        score = 0.0
        if chunk_id in dense_rank:
            score += dense_weight * (1.0 / float(k + dense_rank[chunk_id]))
        if chunk_id in sparse_rank:
            score += sparse_weight * (1.0 / float(k + sparse_rank[chunk_id]))
        rows.append((chunk_id, score))

    rows.sort(key=lambda x: (-x[1], x[0]))
    out: list[RetrievedChunk] = []
    for chunk_id, fused_score in rows:
        base = by_chunk[chunk_id]
        payload = dict(base.payload)
        payload["hybrid_dense_rank"] = dense_rank.get(chunk_id)
        payload["hybrid_sparse_rank"] = sparse_rank.get(chunk_id)
        payload["hybrid_dense_score"] = (
            float(next(d.score for d in dense if d.chunk_id == chunk_id))
            if chunk_id in dense_rank
            else None
        )
        payload["hybrid_sparse_score"] = (
            float(next(d.score for d in sparse if d.chunk_id == chunk_id))
            if chunk_id in sparse_rank
            else None
        )
        out.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                score=float(fused_score),
                text=base.text,
                payload=payload,
                point_id=base.point_id,
            )
        )
    return out


def _weighted_sum_fuse(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    *,
    dense_weight: float,
    sparse_weight: float,
) -> list[RetrievedChunk]:
    if not dense and not sparse:
        return []

    def _normalize(items: list[RetrievedChunk]) -> dict[str, float]:
        if not items:
            return {}
        scores = [float(x.score) for x in items]
        low = min(scores)
        high = max(scores)
        if high <= low:
            return {x.chunk_id: 1.0 for x in items}
        return {x.chunk_id: (float(x.score) - low) / (high - low) for x in items}

    dense_norm = _normalize(dense)
    sparse_norm = _normalize(sparse)
    by_chunk: dict[str, RetrievedChunk] = {}
    for doc in dense + sparse:
        by_chunk.setdefault(doc.chunk_id, doc)

    rows: list[tuple[str, float]] = []
    for chunk_id in by_chunk.keys():
        score = dense_weight * dense_norm.get(chunk_id, 0.0) + sparse_weight * sparse_norm.get(
            chunk_id, 0.0
        )
        rows.append((chunk_id, score))
    rows.sort(key=lambda x: (-x[1], x[0]))

    out: list[RetrievedChunk] = []
    for chunk_id, fused_score in rows:
        base = by_chunk[chunk_id]
        payload = dict(base.payload)
        payload["hybrid_dense_score"] = dense_norm.get(chunk_id)
        payload["hybrid_sparse_score"] = sparse_norm.get(chunk_id)
        out.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                score=float(fused_score),
                text=base.text,
                payload=payload,
                point_id=base.point_id,
            )
        )
    return out


def retrieve_multi_queries(
    retriever: "QdrantRetriever",
    *,
    queries: Sequence[str],
    top_k_primary: int,
    top_k_secondary: int,
    query_filter: qmodels.Filter | None = None,
    score_threshold: float | None = None,
    threshold_direction: ThresholdDirection | None = None,
    dedupe_by_chunk_id: bool = True,
) -> tuple[list[RetrievedChunk], list[RetrievalBatchResult]]:
    cleaned_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        cur = str(query or "").strip()
        if not cur or cur in seen_queries:
            continue
        seen_queries.add(cur)
        cleaned_queries.append(cur)

    if not cleaned_queries:
        return [], []

    all_batches: list[RetrievalBatchResult] = []
    merged: list[RetrievedChunk] = []
    for idx, query in enumerate(cleaned_queries):
        top_k = int(top_k_primary) if idx == 0 else int(top_k_secondary)
        docs = retriever.query(
            query,
            top_k=max(1, top_k),
            query_filter=query_filter,
            score_threshold=score_threshold,
            threshold_direction=threshold_direction,
        )
        all_batches.append(
            RetrievalBatchResult(
                name="primary" if idx == 0 else f"rewrite_{idx}",
                query=query,
                top_k=max(1, top_k),
                retrieved=tuple(docs),
            )
        )
        if dedupe_by_chunk_id:
            merged = merge_retrieved(merged, docs)
        else:
            merged.extend(docs)
    return merged, all_batches


def retrieve_multi_queries_hybrid(
    retriever: "QdrantRetriever",
    *,
    queries: Sequence[str],
    top_k_primary: int,
    top_k_secondary: int,
    query_filter: qmodels.Filter | None = None,
    score_threshold: float | None = None,
    threshold_direction: ThresholdDirection | None = None,
    dedupe_by_chunk_id: bool = True,
    hybrid_config: AdvancedHybridConfig,
) -> tuple[list[RetrievedChunk], list[RetrievalBatchResult], list[dict[str, Any]]]:
    cleaned_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        cur = str(query or "").strip()
        if not cur or cur in seen_queries:
            continue
        seen_queries.add(cur)
        cleaned_queries.append(cur)

    if not cleaned_queries:
        return [], [], []

    all_batches: list[RetrievalBatchResult] = []
    merged: list[RetrievedChunk] = []
    hybrid_stats: list[dict[str, Any]] = []

    for idx, query in enumerate(cleaned_queries):
        top_k = int(top_k_primary) if idx == 0 else int(top_k_secondary)
        hres = retriever.query_hybrid(
            query,
            top_k=max(1, top_k),
            query_filter=query_filter,
            score_threshold=score_threshold,
            threshold_direction=threshold_direction,
            hybrid_config=hybrid_config,
        )
        batch_name = "primary" if idx == 0 else f"rewrite_{idx}"
        all_batches.append(
            RetrievalBatchResult(
                name=batch_name,
                query=query,
                top_k=max(1, top_k),
                retrieved=tuple(hres.retrieved),
            )
        )
        hybrid_stats.append({"name": batch_name, "query": query, **hres.to_dict()})

        # Add channel-level trace batches for diagnostics.
        all_batches.append(
            RetrievalBatchResult(
                name=f"{batch_name}_dense",
                query=query,
                top_k=int(hybrid_config.dense_top_k),
                retrieved=tuple(hres.dense_retrieved),
            )
        )
        all_batches.append(
            RetrievalBatchResult(
                name=f"{batch_name}_sparse",
                query=query,
                top_k=int(hybrid_config.sparse_top_k),
                retrieved=tuple(hres.sparse_retrieved),
            )
        )

        if dedupe_by_chunk_id:
            merged = merge_retrieved(merged, hres.retrieved)
        else:
            merged.extend(list(hres.retrieved))

    return merged, all_batches, hybrid_stats


class QdrantRetriever:
    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        embedder: SupportsEmbedding,
        field_map: QdrantPayloadFieldMap,
        dense_vector_name: str | None = None,
        sparse_vector_name: str | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.field_map = field_map
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.sparse_encoder = sparse_encoder

    @classmethod
    def from_sparse_artifact(
        cls,
        *,
        sparse_artifact_path: Path | None,
        **kwargs: Any,
    ) -> "QdrantRetriever":
        sparse_encoder = None
        if sparse_artifact_path is not None and sparse_artifact_path.exists():
            sparse_encoder = SparseEncoder.load_json(sparse_artifact_path)
        return cls(sparse_encoder=sparse_encoder, **kwargs)

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embedder.embed_texts([query])
        if len(vectors) != 1:
            raise RuntimeError(
                "Embedding backend returned unexpected query vector count "
                f"(expected=1, got={len(vectors)})"
            )
        return list(vectors[0])

    def _normalize_point(self, point: Any) -> RetrievedChunk:
        payload = dict(point.payload or {})
        chunk_id = _as_str(payload.get(self.field_map.chunk_id)) or str(point.id)
        text = (
            _as_str(payload.get(self.field_map.text))
            or _as_str(payload.get(self.field_map.text_for_embedding))
            or ""
        )
        return RetrievedChunk(
            chunk_id=chunk_id,
            score=float(getattr(point, "score", 0.0)),
            text=text,
            payload=payload,
            point_id=str(point.id),
        )

    def query(
        self,
        query: str,
        *,
        top_k: int,
        query_filter: qmodels.Filter | None = None,
        score_threshold: float | None = None,
        threshold_direction: ThresholdDirection | None = None,
    ) -> list[RetrievedChunk]:
        return self.query_dense(
            query,
            top_k=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            threshold_direction=threshold_direction,
        )

    def query_dense(
        self,
        query: str,
        *,
        top_k: int,
        query_filter: qmodels.Filter | None = None,
        score_threshold: float | None = None,
        threshold_direction: ThresholdDirection | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        if score_threshold is not None and threshold_direction is None:
            raise ValueError(
                "score_threshold is set but threshold_direction is None. "
                "Set one of: 'gte', 'lte'."
            )
        vector = self.embed_query(query)
        kwargs: dict[str, Any] = {}
        if self.dense_vector_name is not None:
            kwargs["using"] = self.dense_vector_name
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
            **kwargs,
        )
        points = response.points if hasattr(response, "points") else []
        out = [self._normalize_point(p) for p in points]
        return _score_gate(
            out,
            score_threshold=score_threshold,
            threshold_direction=threshold_direction,
        )

    def query_sparse(
        self,
        query: str,
        *,
        top_k: int,
        query_filter: qmodels.Filter | None = None,
        min_sparse_score: float | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        if self.sparse_vector_name is None or self.sparse_encoder is None:
            return []

        sparse = self.sparse_encoder.transform(query, is_query=True)
        if not sparse.indices:
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=qmodels.SparseVector(indices=list(sparse.indices), values=list(sparse.values)),
            using=self.sparse_vector_name,
            query_filter=query_filter,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )
        points = response.points if hasattr(response, "points") else []
        out = [self._normalize_point(p) for p in points]
        if min_sparse_score is not None:
            out = [d for d in out if d.score >= float(min_sparse_score)]
        return out

    def query_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        query_filter: qmodels.Filter | None = None,
        score_threshold: float | None = None,
        threshold_direction: ThresholdDirection | None = None,
        hybrid_config: AdvancedHybridConfig,
    ) -> HybridRetrievalResult:
        dense_k = min(max(0, int(hybrid_config.dense_top_k)), max(1, int(top_k * 2)))
        sparse_k = min(max(0, int(hybrid_config.sparse_top_k)), max(1, int(top_k * 3)))

        dense_docs: list[RetrievedChunk] = []
        sparse_docs: list[RetrievedChunk] = []
        retrieval_mode = "dense_only"

        if dense_k > 0:
            dense_docs = self.query_dense(
                query,
                top_k=dense_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                threshold_direction=threshold_direction,
            )

        sparse_available = bool(self.sparse_vector_name and self.sparse_encoder and sparse_k > 0)
        if hybrid_config.enabled and sparse_available:
            sparse_docs = self.query_sparse(
                query,
                top_k=sparse_k,
                query_filter=query_filter,
                min_sparse_score=hybrid_config.min_sparse_score,
            )

        if hybrid_config.enabled and sparse_available and sparse_docs:
            retrieval_mode = "hybrid"
            if hybrid_config.fusion_method == "weighted_sum":
                fused = _weighted_sum_fuse(
                    dense_docs,
                    sparse_docs,
                    dense_weight=hybrid_config.dense_weight,
                    sparse_weight=hybrid_config.sparse_weight,
                )
            else:
                fused = _rrf_fuse(
                    dense_docs,
                    sparse_docs,
                    k=max(1, int(hybrid_config.rrf_k)),
                    dense_weight=max(0.0, float(hybrid_config.dense_weight)),
                    sparse_weight=max(0.0, float(hybrid_config.sparse_weight)),
                )
            fused = fused[: max(1, int(top_k))]
        else:
            if hybrid_config.enabled and sparse_available and not sparse_docs:
                retrieval_mode = "fallback_dense"
            fused = dense_docs[: max(1, int(top_k))]

        if score_threshold is not None:
            fused = _score_gate(
                fused,
                score_threshold=score_threshold,
                threshold_direction=threshold_direction,
            )

        dense_ids = {x.chunk_id for x in dense_docs}
        sparse_ids = {x.chunk_id for x in sparse_docs}
        overlap_count = len(dense_ids & sparse_ids)
        added_by_sparse = len([cid for cid in sparse_ids if cid not in dense_ids])

        return HybridRetrievalResult(
            retrieved=tuple(fused),
            dense_retrieved=tuple(dense_docs),
            sparse_retrieved=tuple(sparse_docs),
            retrieval_mode=retrieval_mode,
            overlap_count=overlap_count,
            added_by_sparse=added_by_sparse,
        )

    def retrieve_by_chunk_ids(
        self, chunk_ids: Sequence[str], *, limit: int | None = None
    ) -> list[RetrievedChunk]:
        query_filter = build_chunk_id_filter(self.field_map, chunk_ids)
        if query_filter is None:
            return []
        max_items = len(chunk_ids) if limit is None else max(1, int(limit))
        out: list[RetrievedChunk] = []
        offset: Any = None
        while len(out) < max_items:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(256, max_items - len(out)),
                offset=offset,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for rec in records:
                out.append(self._normalize_point(rec))
                if len(out) >= max_items:
                    break
            if next_offset is None:
                break
            offset = next_offset
        return out


__all__ = [
    "RetrievedChunk",
    "RetrievalBatchResult",
    "HybridRetrievalResult",
    "PayloadSchemaInspection",
    "QdrantRetriever",
    "assert_required_payload_fields",
    "introspect_payload_schema",
    "build_view_filter",
    "build_law_filter",
    "build_article_filter",
    "build_law_status_filter",
    "build_relation_type_filter",
    "build_law_date_filter",
    "merge_filters",
    "merge_retrieved",
    "retrieve_multi_queries",
    "retrieve_multi_queries_hybrid",
]

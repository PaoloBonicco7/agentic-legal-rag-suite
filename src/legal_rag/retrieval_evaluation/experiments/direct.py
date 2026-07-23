"""Direct (dense/hybrid) retrieval experiment engine.

Owns the in-memory candidate cache that is later reused by graph, rerank and
query-rewriting experiments: every downstream stage starts from the same
candidate list to avoid double-paying for retrieval.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.simple_rag.models import RetrievedChunkRecord

from ..evaluator import ChunkAvailabilityIndex, evaluate_candidate_set, retrieve_direct
from ..models import QuestionTarget
from ._shared import compute_answer_overlap_metrics, parallel_map_ordered


class DirectExperimentCache:
    """In-memory candidate cache keyed by index, query, retrieval, and filter identity.

    Shared across the diagnostic experiments so a single retrieval call serves
    direct, graph, rerank and query_rewriting evaluation passes.
    """

    def __init__(self) -> None:
        self._store: dict[tuple, list[RetrievedChunkRecord]] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def _key(
        retrieval_mode: str,
        top_k: int,
        rrf_k: int | None,
        *,
        collection_identity: str = "",
        query_text: str = "",
        filters: Mapping[str, Any] | None = None,
        exact: bool = False,
    ) -> tuple:
        filter_identity = json.dumps(
            _stable_json_value(dict(filters or {})),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            collection_identity,
            query_text,
            retrieval_mode,
            int(top_k),
            rrf_k,
            filter_identity,
            bool(exact),
        )

    def get_or_retrieve(
        self,
        *,
        dataset: str,
        target: QuestionTarget,
        client: QdrantClient,
        collection_name: str,
        embedder: SupportsEmbedding,
        retrieval_mode: str,
        top_k: int,
        rrf_k: int | None,
        filter_name: str,
        filters: dict[str, Any],
        rrf_k_default: int,
        index_manifest: dict[str, Any],
        exact: bool = False,
    ) -> list[RetrievedChunkRecord]:
        """Return cached candidates, retrieving lazily if missing."""
        effective_rrf_k = int(rrf_k or rrf_k_default) if retrieval_mode == "hybrid" else None
        identity = build_collection_identity(collection_name, index_manifest)
        key = self._key(
            retrieval_mode,
            top_k,
            effective_rrf_k,
            collection_identity=identity,
            query_text=target.question,
            filters=filters,
            exact=exact,
        )
        with self._lock:
            cached = self._store.get(key)
            # Hybrid fusion depends on its cutoff; only dense rankings are prefix-reusable.
            if cached is None and retrieval_mode == "dense":
                larger_keys = [
                    candidate_key
                    for candidate_key in self._store
                    if candidate_key[:3] == key[:3]
                    and candidate_key[3] >= int(top_k)
                    and candidate_key[4:] == key[4:]
                ]
                if larger_keys:
                    larger_key = min(larger_keys, key=lambda candidate_key: candidate_key[3])
                    cached = list(self._store[larger_key][:top_k])
                    self._store[key] = cached
        if cached is not None:
            return cached
        retrieved = retrieve_direct(
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            query_text=target.question,
            limit=top_k,
            retrieval_mode=retrieval_mode,
            static_filters=filters,
            rrf_k=int(effective_rrf_k or rrf_k_default),
            index_manifest=index_manifest,
            exact=exact,
        )
        with self._lock:
            self._store[key] = retrieved
        return retrieved


def run_direct_experiment(
    *,
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
    cache: DirectExperimentCache,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    index_manifest: dict[str, Any],
    rrf_k_default: int,
    availability: ChunkAvailabilityIndex,
    modes: Sequence[str],
    filter_names: Sequence[str],
    filter_variants: Mapping[str, dict[str, Any]],
    top_k_values: Sequence[int],
    hybrid_top_k_values: Sequence[int] | None = None,
    hybrid_rrf_k_values: Sequence[int] | None = None,
    hybrid_filters_enabled: bool = False,
    exact_values: Sequence[bool] = (False,),
    show_progress: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Run direct (dense + hybrid) retrieval over the full sweep and return rows."""
    hybrid_top_k_values = list(hybrid_top_k_values or top_k_values)
    hybrid_rrf_k_values = list(hybrid_rrf_k_values or [60])
    collection_identity = build_collection_identity(collection_name, index_manifest)

    plan: list[tuple] = []
    for mode in modes:
        mode_top_k_values = (
            hybrid_top_k_values
            if mode == "hybrid"
            else sorted((int(value) for value in top_k_values), reverse=True)
        )
        mode_rrf_k_values = hybrid_rrf_k_values if mode == "hybrid" else [None]
        mode_filter_names = (
            list(filter_names)
            if mode == "dense" or hybrid_filters_enabled
            else ["none"]
        )
        mode_exact_values = [False] if mode == "hybrid" else [bool(value) for value in exact_values]
        plan.extend(
            itertools.product(
                [mode],
                mode_filter_names,
                mode_top_k_values,
                mode_rrf_k_values,
                mode_exact_values,
                _flatten_targets(targets_by_dataset),
            )
        )

    def process(entry: tuple) -> dict[str, Any]:
        retrieval_mode, filter_name, top_k, rrf_k, exact, (dataset, target) = entry
        filters = filter_variants[filter_name]
        retrieved = cache.get_or_retrieve(
            dataset=dataset,
            target=target,
            client=qdrant_client,
            collection_name=collection_name,
            embedder=embedder,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            rrf_k=rrf_k,
            filter_name=filter_name,
            filters=filters,
            rrf_k_default=rrf_k_default,
            index_manifest=index_manifest,
            exact=exact,
        )
        row = evaluate_candidate_set(
            target=target,
            retrieved=retrieved,
            expanded=[],
            availability=availability,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            rrf_k=rrf_k if retrieval_mode == "hybrid" else None,
            filter_name=filter_name,
            metadata_filters=filters,
            exact=exact,
            collection_identity=collection_identity,
        ).to_json_record()
        row["dataset"] = dataset
        # Flatten article_hit_at_k dicts into top-level boolean columns so the
        # pandas aggregation in summarize_direct() can mean them directly.
        for k_str, hit in (row.get("direct_article_hit_at_k") or {}).items():
            row[f"direct_article_hit_at_{k_str}"] = bool(hit)
        for k_str, hit in (row.get("post_article_hit_at_k") or {}).items():
            row[f"post_article_hit_at_{k_str}"] = bool(hit)
        row.update(compute_answer_overlap_metrics(target, retrieved, prefix="direct"))
        return row

    rows = parallel_map_ordered(
        plan,
        process,
        max_workers=max_workers,
        description="direct experiments",
        show_progress=show_progress,
    )
    return pd.DataFrame(rows)


def summarize_direct(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate direct/hybrid rows per (dataset, mode, filter, top_k, rrf_k)."""
    if df.empty:
        return df
    working = df.copy()
    if "exact" not in working:
        working["exact"] = False
    agg_kwargs: dict[str, tuple[str, str]] = {
        "questions": ("qid", "nunique"),
        "article_hit": ("direct_article_hit", "mean"),
        "all_expected_articles_hit": ("direct_all_expected_articles_hit", "mean"),
        "law_hit": ("direct_law_hit", "mean"),
        "article_mrr": ("direct_article_mrr", "mean"),
        "law_only_false_positive_rate": ("law_only_false_positive", "mean"),
        "filter_excluded_rate": ("filter_excluded", "mean"),
        "avg_best_answer_overlap": ("direct_best_expected_article_answer_overlap", "mean"),
    }
    for column in working.columns:
        if column.startswith("direct_article_hit_at_"):
            suffix = column[len("direct_article_hit_at_") :]
            agg_kwargs[f"article_hit_at_{suffix}"] = (column, "mean")
    return (
        working.groupby(
            ["dataset", "retrieval_mode", "filter_name", "top_k", "rrf_k", "exact"],
            dropna=False,
        )
        .agg(**agg_kwargs)
        .reset_index()
        .sort_values(["dataset", "retrieval_mode", "filter_name", "top_k", "rrf_k", "exact"])
    )


def summarize_direct_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Same as `summarize_direct`, broken down by question difficulty level."""
    if df.empty:
        return df
    working = df.copy()
    if "exact" not in working:
        working["exact"] = False
    return (
        working.groupby(
            [
                "dataset",
                "level",
                "retrieval_mode",
                "filter_name",
                "top_k",
                "rrf_k",
                "exact",
            ],
            dropna=False,
        )
        .agg(
            questions=("qid", "nunique"),
            article_hit=("direct_article_hit", "mean"),
            law_hit=("direct_law_hit", "mean"),
            article_mrr=("direct_article_mrr", "mean"),
            law_only_false_positive_rate=("law_only_false_positive", "mean"),
        )
        .reset_index()
    )


def _flatten_targets(
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
) -> list[tuple[str, QuestionTarget]]:
    return [(dataset, target) for dataset, targets in targets_by_dataset.items() for target in targets]


def build_collection_identity(
    collection_name: str,
    index_manifest: Mapping[str, Any],
) -> str:
    """Build a stable identity for the collection and manifest under evaluation."""
    payload = {
        "collection_name": str(collection_name),
        "index_manifest": _stable_json_value(dict(index_manifest)),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_stable_json_value(item) for item in value), key=str)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

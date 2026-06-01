"""Graph expansion experiment engine.

Expands a direct/hybrid candidate set through the legal graph and measures the
incremental hit / noise tradeoff per configuration. Reuses the shared
`DirectExperimentCache` from `direct.py` so retrieval is paid for once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from legal_rag.advanced_graph_rag.retrieval import GraphIndex, expand_with_graph
from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.simple_rag.models import RetrievedChunkRecord

from ..evaluator import ChunkAvailabilityIndex, dedupe_chunks, evaluate_candidate_set
from ..models import QuestionTarget
from ._shared import compute_answer_overlap_metrics, wrap_progress
from .direct import DirectExperimentCache


def select_graph_base_configs(
    direct_summary_df: pd.DataFrame,
    *,
    datasets: Sequence[str],
    base_config_names: Sequence[str],
    baseline_top_k: int,
) -> list[dict[str, Any]]:
    """Resolve the `base_config_names` into concrete (mode, filter, top_k, rrf_k) rows.

    Each base name in `base_config_names` maps a strategy onto the direct sweep:

    - ``dense@K`` / ``hybrid@K``: fixed top_k, picks the best filter/rrf_k row.
    - ``dense_best`` / ``hybrid_best``: best row of that mode by article_hit.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for dataset in datasets:
        subset_all = direct_summary_df[direct_summary_df["dataset"] == dataset]
        for name in base_config_names:
            subset = subset_all.copy()
            if name == "dense@10":
                subset = subset[
                    (subset["retrieval_mode"] == "dense")
                    & (subset["filter_name"] == "none")
                    & (subset["top_k"] == baseline_top_k)
                ]
            elif name == "dense_best":
                subset = subset[
                    (subset["retrieval_mode"] == "dense")
                    & (subset["filter_name"] == "none")
                ].sort_values(["article_hit", "article_mrr", "top_k"], ascending=[False, False, True]).head(1)
            elif name == "hybrid_best":
                subset = subset[
                    (subset["retrieval_mode"] == "hybrid")
                    & (subset["filter_name"] == "none")
                ].sort_values(
                    ["article_hit", "article_mrr", "top_k", "rrf_k"], ascending=[False, False, True, True]
                ).head(1)
            elif name.startswith("dense@"):
                top_k = int(name.split("@", 1)[1])
                subset = subset[
                    (subset["retrieval_mode"] == "dense")
                    & (subset["filter_name"] == "none")
                    & (subset["top_k"] == top_k)
                ]
            elif name.startswith("hybrid@"):
                top_k = int(name.split("@", 1)[1])
                subset = subset[
                    (subset["retrieval_mode"] == "hybrid")
                    & (subset["filter_name"] == "none")
                    & (subset["top_k"] == top_k)
                ].sort_values(["article_hit", "article_mrr", "rrf_k"], ascending=[False, False, True]).head(1)
            else:
                raise ValueError(f"Unsupported graph base config: {name}")
            if subset.empty:
                continue
            row = subset.iloc[0]
            key = (
                dataset,
                str(row["retrieval_mode"]),
                str(row["filter_name"]),
                int(row["top_k"]),
                None if pd.isna(row.get("rrf_k")) else int(row["rrf_k"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "dataset": dataset,
                    "graph_base_config_name": name,
                    "retrieval_mode": key[1],
                    "filter_name": key[2],
                    "top_k": key[3],
                    "rrf_k": key[4],
                    "base_article_hit": float(row["article_hit"]),
                    "base_law_hit": float(row["law_hit"]),
                }
            )
    return rows


def run_graph_experiment(
    *,
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
    base_configs: Sequence[dict[str, Any]],
    cache: DirectExperimentCache,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    index_manifest: dict[str, Any],
    rrf_k_default: int,
    graph: GraphIndex,
    availability: ChunkAvailabilityIndex,
    filter_variants: Mapping[str, dict[str, Any]],
    relation_type_variants: Mapping[str, Sequence[str]],
    seed_k_values: Sequence[int],
    max_chunks_per_law_values: Sequence[int],
    min_edge_confidence_values: Sequence[float],
    relation_set_names: Sequence[str],
    max_expanded_chunks_total: int,
    expand_only_problem_cases: bool = True,
    backend: str = "qdrant_ranked",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Expand the selected `base_configs` over the graph and return per-question rows.

    The expansion can be skipped for questions whose direct candidates already
    cover the expected article (`expand_only_problem_cases=True`), which keeps
    sweeps cheap while still letting the graph stage attempt the harder cases.
    """
    relation_sets = {
        name: list(relation_type_variants[name])
        for name in relation_set_names
        if name in relation_type_variants
    }
    if not relation_sets:
        return pd.DataFrame()

    plan = [
        (base_config, seed_k, max_chunks_per_law, min_confidence, (relation_name, relation_types), target)
        for base_config in base_configs
        for seed_k in seed_k_values
        for max_chunks_per_law in max_chunks_per_law_values
        for min_confidence in min_edge_confidence_values
        for relation_name, relation_types in relation_sets.items()
        for target in targets_by_dataset[base_config["dataset"]]
    ]

    rows: list[dict[str, Any]] = []
    for base_config, seed_k, max_chunks_per_law, min_confidence, (relation_name, relation_types), target in wrap_progress(
        plan, description="graph experiments", enabled=show_progress
    ):
        dataset = base_config["dataset"]
        filter_name = base_config["filter_name"]
        filters = filter_variants[filter_name]
        retrieved = cache.get_or_retrieve(
            dataset=dataset,
            target=target,
            client=qdrant_client,
            collection_name=collection_name,
            embedder=embedder,
            retrieval_mode=base_config["retrieval_mode"],
            top_k=base_config["top_k"],
            rrf_k=base_config["rrf_k"],
            filter_name=filter_name,
            filters=filters,
            rrf_k_default=rrf_k_default,
            index_manifest=index_manifest,
        )
        should_expand = True
        if expand_only_problem_cases:
            should_expand = not _has_expected_article(target, retrieved)
        if should_expand:
            expanded, _relations = _expand_graph_candidates(
                retrieved=retrieved,
                target=target,
                graph=graph,
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                embedder=embedder,
                filters=filters,
                seed_k=seed_k,
                max_chunks_per_law=max_chunks_per_law,
                min_confidence=min_confidence,
                relation_types=relation_types,
                max_expanded_chunks_total=max_expanded_chunks_total,
                backend=backend,
            )
        else:
            expanded = []
        candidates = dedupe_chunks([*retrieved, *expanded])
        row = evaluate_candidate_set(
            target=target,
            retrieved=retrieved,
            expanded=expanded,
            availability=availability,
            retrieval_mode=base_config["retrieval_mode"],
            top_k=base_config["top_k"],
            rrf_k=base_config["rrf_k"],
            filter_name=filter_name,
            metadata_filters=filters,
            graph_expansion_enabled=True,
            graph_expansion_seed_k=seed_k,
            max_chunks_per_expanded_law=max_chunks_per_law,
            min_edge_confidence=min_confidence,
        ).to_json_record()
        row["dataset"] = dataset
        row.update(compute_answer_overlap_metrics(target, retrieved, prefix="direct"))
        row.update(compute_answer_overlap_metrics(target, candidates, prefix="post"))
        row["graph_base_config_name"] = base_config["graph_base_config_name"]
        row["relation_set_name"] = relation_name
        row["relation_types"] = list(relation_types)
        row["graph_expansion_attempted"] = should_expand
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_graph(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate graph rows by (base_config, seed_k, relation_set, top_k, ...)."""
    if df.empty:
        return df
    summary = (
        df.groupby(
            [
                "dataset",
                "graph_base_config_name",
                "retrieval_mode",
                "filter_name",
                "top_k",
                "rrf_k",
                "graph_expansion_seed_k",
                "max_chunks_per_expanded_law",
                "min_edge_confidence",
                "relation_set_name",
            ],
            dropna=False,
        )
        .agg(
            questions=("qid", "nunique"),
            expansion_attempt_rate=("graph_expansion_attempted", "mean"),
            direct_article_hit=("direct_article_hit", "mean"),
            post_article_hit=("post_article_hit", "mean"),
            post_law_hit=("post_law_hit", "mean"),
            post_article_mrr=("post_article_mrr", "mean"),
            graph_incremental_hits=("graph_incremental_hit", "sum"),
            expanded_expected_article_hits=("expanded_expected_article_hits", "sum"),
            expansion_noise_ratio=("expansion_noise_ratio", "mean"),
            filter_excluded_rate=("filter_excluded", "mean"),
        )
        .reset_index()
    )
    summary["delta_vs_direct"] = summary["post_article_hit"] - summary["direct_article_hit"]
    return summary.sort_values(
        ["dataset", "delta_vs_direct", "graph_incremental_hits", "post_article_hit", "expansion_noise_ratio"],
        ascending=[True, False, False, False, True],
    )


def best_graph_by_dataset(graph_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the winning graph configuration per dataset based on hit + low noise."""
    if graph_summary_df.empty:
        return graph_summary_df
    return (
        graph_summary_df.sort_values(
            ["dataset", "delta_vs_direct", "graph_incremental_hits", "post_article_hit", "expansion_noise_ratio"],
            ascending=[True, False, False, False, True],
        )
        .groupby("dataset", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def graph_scenario_config(row: pd.Series) -> dict[str, Any]:
    """Build the config dict for a graph waterfall scenario row."""
    return {
        "retrieval_mode": str(row["retrieval_mode"]),
        "base_config": str(row.get("graph_base_config_name", "custom")),
        "filter": str(row["filter_name"]),
        "top_k": int(row["top_k"]),
        "rrf_k": None if "rrf_k" not in row or pd.isna(row["rrf_k"]) else int(row["rrf_k"]),
        "seed_k": int(row["graph_expansion_seed_k"]),
        "max_chunks_per_law": int(row["max_chunks_per_expanded_law"]),
        "min_confidence": float(row["min_edge_confidence"]),
        "relations": str(row["relation_set_name"]),
        "expansion_noise_ratio": (
            None if pd.isna(row.get("expansion_noise_ratio")) else float(row["expansion_noise_ratio"])
        ),
    }


def _has_expected_article(target: QuestionTarget, candidates: Sequence[RetrievedChunkRecord]) -> bool:
    expected = set(target.expected_article_ids)
    return any(str(chunk.payload.get("article_id") or "") in expected for chunk in candidates)


def _expand_graph_candidates(
    *,
    retrieved: Sequence[RetrievedChunkRecord],
    target: QuestionTarget,
    graph: GraphIndex,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    filters: dict[str, Any],
    seed_k: int,
    max_chunks_per_law: int,
    min_confidence: float,
    relation_types: Sequence[str],
    max_expanded_chunks_total: int,
    backend: str,
):
    use_qdrant_ranked = backend == "qdrant_ranked"
    return expand_with_graph(
        qdrant_client if use_qdrant_ranked else None,
        collection_name=collection_name,
        graph=graph,
        seeds=list(retrieved)[:seed_k],
        relation_types=list(relation_types),
        static_filters=filters,
        max_chunks_per_law=max_chunks_per_law,
        embedder=embedder if use_qdrant_ranked else None,
        query_text=target.question if use_qdrant_ranked else "",
        max_chunks_total=max_expanded_chunks_total,
        min_edge_confidence=min_confidence,
    )

"""LLM rerank diagnostic experiment.

Picks the best hybrid base config for each dataset, samples a deterministic
30-question pilot (or runs the full dataset when `full_run=True`), and scores
the candidates through a cache-aware LLM rerank step.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from legal_rag.advanced_graph_rag.prompts import RERANK_PROMPT_VERSION
from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.llm import StructuredChatClient

from ..evaluator import RerankCache, evaluate_with_rerank, score_rerank_candidates
from ..models import QuestionTarget
from ._shared import parallel_map_ordered
from .direct import DirectExperimentCache
from .hybrid import best_hybrid_rrf_k


def sample_rerank_targets(
    targets: Sequence[QuestionTarget],
    *,
    sample_size: int,
    seed: int,
    full_run: bool = False,
) -> list[QuestionTarget]:
    """Deterministic 30-question pilot sampler (or pass-through when `full_run=True`)."""
    if full_run or len(targets) <= sample_size:
        return list(targets)
    rng = random.Random(seed)
    return rng.sample(list(targets), sample_size)


def run_rerank_experiment(
    *,
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
    direct_summary_df: pd.DataFrame,
    cache: DirectExperimentCache,
    rerank_cache: RerankCache,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    index_manifest: dict[str, Any],
    rrf_k_default: int,
    llm_client: StructuredChatClient,
    rerank_model: str,
    timeout_seconds: int,
    input_k_values: Sequence[int],
    output_k_values: Sequence[int],
    question_sample: int,
    sample_seed: int,
    full_run: bool = False,
    show_progress: bool = True,
    max_workers: int = 1,
    request_delay_seconds: float = 0.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    """Run the rerank pilot and return (rows_df, failures, cache_stats)."""
    work: list[dict[str, Any]] = []
    for dataset in targets_by_dataset:
        base_rrf_k = best_hybrid_rrf_k(direct_summary_df, dataset=dataset)
        if base_rrf_k is None:
            continue
        selected_targets = sample_rerank_targets(
            targets_by_dataset[dataset],
            sample_size=question_sample,
            seed=sample_seed,
            full_run=full_run,
        )
        for target in selected_targets:
            for rerank_input_k in input_k_values:
                candidates = cache.get_or_retrieve(
                    dataset=dataset,
                    target=target,
                    client=qdrant_client,
                    collection_name=collection_name,
                    embedder=embedder,
                    retrieval_mode="hybrid",
                    top_k=rerank_input_k,
                    rrf_k=base_rrf_k,
                    filter_name="none",
                    filters={},
                    rrf_k_default=rrf_k_default,
                    index_manifest=index_manifest,
                )
                work.append(
                    {
                        "dataset": dataset,
                        "base_rrf_k": base_rrf_k,
                        "target": target,
                        "rerank_input_k": rerank_input_k,
                        "candidates": list(candidates[:rerank_input_k]),
                    }
                )

    def call_rerank(item: dict[str, Any]) -> dict[str, Any]:
        try:
            scores, cache_hit = score_rerank_candidates(
                llm_client=llm_client,
                question=item["target"].question,
                candidates=item["candidates"],
                model=rerank_model,
                timeout_seconds=timeout_seconds,
                cache=rerank_cache,
                prompt_version=RERANK_PROMPT_VERSION,
            )
            return {**item, "scores": scores, "cache_hit": cache_hit, "error": None}
        except Exception as exc:
            return {**item, "scores": None, "cache_hit": False, "error": f"{type(exc).__name__}: {exc}"}

    results = parallel_map_ordered(
        work,
        call_rerank,
        max_workers=max_workers,
        description="rerank calls",
        show_progress=show_progress,
        request_delay_seconds=request_delay_seconds,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    for result in results:
        if result["error"] is not None:
            failures.append(
                {
                    "dataset": result["dataset"],
                    "qid": result["target"].qid,
                    "rerank_input_k": result["rerank_input_k"],
                    "error": result["error"],
                }
            )
            continue
        cache_hit = bool(result["cache_hit"])
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        for rerank_output_k in output_k_values:
            row = evaluate_with_rerank(
                target=result["target"],
                candidates=result["candidates"],
                scores=result["scores"],
                rerank_input_k=result["rerank_input_k"],
                rerank_output_k=rerank_output_k,
                retrieval_mode="hybrid",
                top_k=result["rerank_input_k"],
                rrf_k=result["base_rrf_k"],
                filter_name="none",
                metadata_filters={},
                base_scenario="hybrid_if_available",
                rerank_model=rerank_model,
                cache_hit=cache_hit,
            ).to_json_record()
            row["dataset"] = result["dataset"]
            row["prompt_version"] = RERANK_PROMPT_VERSION
            rows.append(row)

    stats = {
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "failures": len(failures),
        "rows": len(rows),
    }
    return pd.DataFrame(rows), failures, stats


def summarize_rerank(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rerank rows by (dataset, input_k, output_k, rrf_k)."""
    if df.empty:
        return df
    summary = (
        df.groupby(
            ["dataset", "retrieval_mode", "top_k", "rrf_k", "rerank_input_k", "rerank_output_k", "rerank_model"],
            dropna=False,
        )
        .agg(
            questions=("qid", "nunique"),
            article_hit=("reranked_article_hit", "mean"),
            law_hit=("reranked_law_hit", "mean"),
            article_mrr=("reranked_article_mrr", "mean"),
            pre_article_hit=("pre_rerank_article_hit", "mean"),
            cache_hit_rate=("cache_hit", "mean"),
            recovered=("rerank_recovered_article", "sum"),
            demoted=("rerank_demoted_article", "sum"),
        )
        .reset_index()
    )
    summary["delta_vs_pre_article_hit"] = summary["article_hit"] - summary["pre_article_hit"]
    summary["net_recovered"] = summary["recovered"] - summary["demoted"]
    return summary


def rerank_best_impact(rerank_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the winning rerank configuration per dataset by hit + delta."""
    if rerank_summary_df.empty:
        return rerank_summary_df
    return (
        rerank_summary_df.sort_values(
            ["dataset", "article_hit", "delta_vs_pre_article_hit", "article_mrr", "rerank_input_k", "rerank_output_k"],
            ascending=[True, False, False, False, True, True],
        )
        .groupby("dataset", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def failure_rate(failures: Sequence[Mapping[str, Any]], *, cache_stats: Mapping[str, int]) -> float:
    """Failure ratio relative to attempted rerank calls (failures + hits + misses)."""
    attempts = len(failures) + int(cache_stats.get("cache_hits", 0)) + int(cache_stats.get("cache_misses", 0))
    return 0.0 if attempts == 0 else len(failures) / attempts


def default_cache_path(cache_root: Path, *, model: str) -> Path:
    """Convenience wrapper around `rerank_cache_path` for the notebook."""
    from ..evaluator import rerank_cache_path

    return rerank_cache_path(cache_root, model=model)

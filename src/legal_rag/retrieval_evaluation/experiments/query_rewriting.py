"""Query rewriting / HyDE / multi-query diagnostic experiment.

For each strategy, generates one or more query variants (cached per strategy +
model + prompt version), retrieves candidates over the best hybrid
configuration, and scores hit/MRR against expected references.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.llm import StructuredChatClient

from ..evaluator import evaluate_query_rewrite
from ..models import QuestionTarget
from ..query_rewriting import (
    QUERY_REWRITING_PROMPT_VERSION,
    QueryRewriteCache,
    generate_hyde,
    multi_query,
    query_rewrite_cache_path,
    retrieve_query_variants,
    rewrite_query,
)
from ._shared import compute_answer_overlap_metrics, parallel_map_ordered
from .hybrid import best_hybrid_config


def sample_query_rewrite_targets(
    targets: Sequence[QuestionTarget],
    *,
    sample_size: int,
    seed: int,
    full_run: bool = False,
) -> list[QuestionTarget]:
    """Deterministic 30-question pilot sampler for query rewriting."""
    if full_run or len(targets) <= sample_size:
        return list(targets)
    rng = random.Random(seed)
    return rng.sample(list(targets), sample_size)


def build_query_rewrite_caches(
    *,
    strategies: Sequence[str],
    cache_root: Path,
    model: str,
    prompt_version: str = QUERY_REWRITING_PROMPT_VERSION,
) -> dict[str, QueryRewriteCache]:
    """Build one cache per strategy (excluding 'none')."""
    return {
        strategy: QueryRewriteCache(
            query_rewrite_cache_path(cache_root, strategy=strategy, model=model, prompt_version=prompt_version)
        )
        for strategy in strategies
        if strategy != "none"
    }


def run_query_rewriting_experiment(
    *,
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
    direct_summary_df: pd.DataFrame,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    index_manifest: dict[str, Any],
    llm_client: StructuredChatClient,
    rewrite_caches: Mapping[str, QueryRewriteCache],
    rewrite_model: str,
    timeout_seconds: int,
    strategies: Sequence[str],
    question_sample: int,
    sample_seed: int,
    multi_query_n: int = 3,
    full_run: bool = False,
    prompt_version: str = QUERY_REWRITING_PROMPT_VERSION,
    show_progress: bool = True,
    max_workers: int = 1,
    request_delay_seconds: float = 0.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Run the query rewriting pilot.

    Returns (rows_df, failures, examples, cache_stats).
    """
    work: list[dict[str, Any]] = []
    for dataset in targets_by_dataset:
        base_config = best_hybrid_config(direct_summary_df, dataset=dataset)
        if base_config is None:
            continue
        base_top_k, base_rrf_k = base_config
        selected_targets = sample_query_rewrite_targets(
            targets_by_dataset[dataset],
            sample_size=question_sample,
            seed=sample_seed,
            full_run=full_run,
        )
        for target in selected_targets:
            for strategy in strategies:
                work.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "strategy": strategy,
                        "base_top_k": base_top_k,
                        "base_rrf_k": base_rrf_k,
                    }
                )

    def call_strategy(item: dict[str, Any]) -> dict[str, Any]:
        try:
            transformed_queries, cache_hit, result_model, result_prompt_version = _generate_queries(
                strategy=item["strategy"],
                question=item["target"].question,
                llm_client=llm_client,
                rewrite_caches=rewrite_caches,
                model=rewrite_model,
                timeout_seconds=timeout_seconds,
                multi_query_n=multi_query_n,
                prompt_version=prompt_version,
            )
            return {
                **item,
                "transformed_queries": transformed_queries,
                "cache_hit": cache_hit,
                "result_model": result_model,
                "result_prompt_version": result_prompt_version,
                "error": None,
            }
        except Exception as exc:
            return {**item, "error": f"{type(exc).__name__}: {exc}"}

    results = parallel_map_ordered(
        work,
        call_strategy,
        max_workers=max_workers,
        description="query rewriting calls",
        show_progress=show_progress,
        request_delay_seconds=request_delay_seconds,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    for result in results:
        if result["error"] is not None:
            failures.append(
                {
                    "dataset": result["dataset"],
                    "qid": result["target"].qid,
                    "strategy": result["strategy"],
                    "error": result["error"],
                }
            )
            continue
        candidates = retrieve_query_variants(
            client=qdrant_client,
            collection_name=collection_name,
            embedder=embedder,
            query_texts=result["transformed_queries"],
            limit=result["base_top_k"],
            retrieval_mode="hybrid",
            static_filters={},
            rrf_k=result["base_rrf_k"],
            index_manifest=index_manifest,
        )
        row = evaluate_query_rewrite(
            target=result["target"],
            retrieved=candidates,
            retrieval_mode="hybrid",
            top_k=result["base_top_k"],
            rrf_k=result["base_rrf_k"],
            filter_name="none",
            metadata_filters={},
            strategy=result["strategy"],
            rewritten_queries=result["transformed_queries"],
            query_rewriting_model=result["result_model"],
            query_rewriting_prompt_version=result["result_prompt_version"],
            cache_hit=result["cache_hit"],
        ).to_json_record()
        row["dataset"] = result["dataset"]
        row.update(compute_answer_overlap_metrics(result["target"], candidates, prefix="direct"))
        rows.append(row)
        if result["strategy"] != "none":
            cache_hits += int(result["cache_hit"])
            cache_misses += int(not result["cache_hit"])
            if len(examples) < 3:
                examples.append(
                    {
                        "dataset": result["dataset"],
                        "qid": result["target"].qid,
                        "strategy": result["strategy"],
                        "question": result["target"].question,
                        "transformed_queries": result["transformed_queries"],
                    }
                )

    stats = {
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "failures": len(failures),
        "rows": len(rows),
    }
    return pd.DataFrame(rows), failures, examples, stats


def summarize_query_rewriting(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate query rewriting rows by (dataset, strategy, top_k, rrf_k)."""
    if df.empty:
        return df
    return (
        df.groupby(["dataset", "strategy", "retrieval_mode", "top_k", "rrf_k"], dropna=False)
        .agg(
            questions=("qid", "nunique"),
            article_hit=("direct_article_hit", "mean"),
            law_hit=("direct_law_hit", "mean"),
            article_mrr=("direct_article_mrr", "mean"),
            cache_hit_rate=("cache_hit", "mean"),
            avg_transformed_queries=("transformed_query_count", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "article_hit", "article_mrr"], ascending=[True, False, False])
    )


def failure_rate(failures: Sequence[Mapping[str, Any]], *, cache_stats: Mapping[str, int]) -> float:
    """Failure ratio relative to attempted query rewriting calls."""
    attempts = len(failures) + int(cache_stats.get("cache_hits", 0)) + int(cache_stats.get("cache_misses", 0))
    return 0.0 if attempts == 0 else len(failures) / attempts


def _generate_queries(
    *,
    strategy: str,
    question: str,
    llm_client: StructuredChatClient,
    rewrite_caches: Mapping[str, QueryRewriteCache],
    model: str,
    timeout_seconds: int,
    multi_query_n: int,
    prompt_version: str,
) -> tuple[list[str], bool, str | None, str | None]:
    """Dispatch to the matching strategy and return queries + caching metadata."""
    if strategy == "none":
        return [question], True, None, None
    if strategy == "rewrite":
        result = rewrite_query(
            llm_client=llm_client,
            question=question,
            model=model,
            timeout_seconds=timeout_seconds,
            cache=rewrite_caches[strategy],
            prompt_version=prompt_version,
        )
        return list(result.queries), result.cache_hit, result.model, result.prompt_version
    if strategy == "hyde":
        result = generate_hyde(
            llm_client=llm_client,
            question=question,
            model=model,
            timeout_seconds=timeout_seconds,
            cache=rewrite_caches[strategy],
            prompt_version=prompt_version,
        )
        return list(result.queries), result.cache_hit, result.model, result.prompt_version
    if strategy == "multi_query":
        result = multi_query(
            llm_client=llm_client,
            question=question,
            model=model,
            timeout_seconds=timeout_seconds,
            cache=rewrite_caches[strategy],
            n=multi_query_n,
            prompt_version=prompt_version,
        )
        return list(result.queries), result.cache_hit, result.model, result.prompt_version
    raise ValueError(f"Unsupported query rewriting strategy: {strategy}")

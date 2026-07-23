"""Per-experiment engines for the retrieval-diagnostics notebook (06b)."""

from __future__ import annotations

from ._shared import compute_answer_overlap_metrics
from .direct import (
    DirectExperimentCache,
    build_collection_identity,
    run_direct_experiment,
    summarize_direct,
    summarize_direct_by_level,
)
from .graph import (
    best_graph_by_dataset,
    graph_scenario_config,
    run_graph_experiment,
    select_graph_base_configs,
    summarize_graph,
)
from .hybrid import best_hybrid_config, best_hybrid_rrf_k, summarize_hybrid
from .query_rewriting import (
    build_query_rewrite_caches,
    failure_rate as query_rewriting_failure_rate,
    run_query_rewriting_experiment,
    sample_query_rewrite_targets,
    summarize_query_rewriting,
)
from .rerank import (
    default_cache_path as default_rerank_cache_path,
    failure_rate as rerank_failure_rate,
    rerank_best_impact,
    run_rerank_experiment,
    sample_rerank_targets,
    summarize_rerank,
)

__all__ = [
    "DirectExperimentCache",
    "build_collection_identity",
    "best_graph_by_dataset",
    "best_hybrid_config",
    "best_hybrid_rrf_k",
    "build_query_rewrite_caches",
    "compute_answer_overlap_metrics",
    "default_rerank_cache_path",
    "graph_scenario_config",
    "query_rewriting_failure_rate",
    "rerank_best_impact",
    "rerank_failure_rate",
    "run_direct_experiment",
    "run_graph_experiment",
    "run_query_rewriting_experiment",
    "run_rerank_experiment",
    "sample_query_rewrite_targets",
    "sample_rerank_targets",
    "select_graph_base_configs",
    "summarize_direct",
    "summarize_direct_by_level",
    "summarize_graph",
    "summarize_hybrid",
    "summarize_query_rewriting",
    "summarize_rerank",
]

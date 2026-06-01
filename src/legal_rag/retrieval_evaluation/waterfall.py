"""Waterfall summarization and AdvancedRagConfig recommendation.

Folds the per-experiment row-level diagnostics into a single
`scenarios_df` (the table that the notebook displays and the spec mandates),
selects the winning scenario per dataset, and translates the winner into the
payload that `notebooks/06_advanced_graph_rag.ipynb` consumes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from legal_rag.advanced_graph_rag.prompts import RERANK_PROMPT_VERSION

from .evaluator import summarize_scenario
from .experiments.graph import graph_scenario_config
from .models import RETRIEVAL_EVALUATION_SCHEMA_VERSION
from .profiles import FILTER_VARIANTS, RELATION_TYPE_VARIANTS, DiagnosticProfile
from .query_rewriting import QUERY_REWRITING_PROMPT_VERSION


def summarize_experiment(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_name: str,
    scenario_name: str,
    dataset: str,
    stage: str,
    config: Mapping[str, Any],
    article_hit_key: str,
    law_hit_key: str,
    article_mrr_key: str,
    baseline_pct: float | None,
    filter_excluded_key: str | None = "filter_excluded",
    status: str = "run",
    skip_reason: str = "",
) -> dict[str, Any]:
    """Build one waterfall row, returning a skipped placeholder when status != 'run'."""
    if status != "run":
        return {
            "experiment_name": experiment_name,
            "scenario_name": scenario_name,
            "dataset": dataset,
            "stage": stage,
            "status": status,
            "skip_reason": skip_reason,
            "article_hit_pct": None,
            "law_hit_pct": None,
            "article_mrr": None,
            "n_questions": 0,
            "n_filter_excluded": 0,
            "config": dict(config),
            "delta_vs_baseline": None,
        }
    record = summarize_scenario(
        rows,
        scenario_name=scenario_name,
        dataset=dataset,
        stage=stage,
        config=config,
        article_hit_key=article_hit_key,
        law_hit_key=law_hit_key,
        article_mrr_key=article_mrr_key,
        filter_excluded_key=filter_excluded_key,
        baseline_pct=baseline_pct,
    ).to_json_record()
    record.update({"experiment_name": experiment_name, "status": status, "skip_reason": skip_reason})
    return record


def build_waterfall(
    *,
    datasets: Sequence[str],
    profile: DiagnosticProfile,
    direct_df: pd.DataFrame,
    direct_summary_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    graph_summary_df: pd.DataFrame,
    best_graph_by_dataset_df: pd.DataFrame,
    rerank_df: pd.DataFrame,
    query_rewrite_df: pd.DataFrame,
    hybrid_available: bool,
    graph_enabled: bool,
    rerank_enabled: bool,
    query_rewriting_enabled: bool,
) -> pd.DataFrame:
    """Assemble the scenarios_df mandated by the 06b Contract."""
    rows: list[dict[str, Any]] = []
    baseline_top_k = profile.baseline_top_k
    budget_top_k = profile.budget_top_k
    low_noise_threshold = profile.low_noise_threshold
    primary_current_filter = "law_status_current"

    for dataset in datasets:
        direct_subset = direct_df[direct_df["dataset"] == dataset] if not direct_df.empty else pd.DataFrame()
        graph_subset = graph_df[graph_df["dataset"] == dataset] if not graph_df.empty else pd.DataFrame()
        rerank_subset = rerank_df[rerank_df["dataset"] == dataset] if not rerank_df.empty else pd.DataFrame()
        query_rewrite_subset = (
            query_rewrite_df[query_rewrite_df["dataset"] == dataset] if not query_rewrite_df.empty else pd.DataFrame()
        )

        # --- baseline -----------------------------------------------------------------
        baseline_rows = (
            direct_subset[
                (direct_subset["retrieval_mode"] == "dense")
                & (direct_subset["top_k"] == baseline_top_k)
                & (direct_subset["filter_name"] == "none")
            ].to_dict("records")
            if not direct_subset.empty
            else []
        )
        baseline_record = summarize_experiment(
            baseline_rows,
            experiment_name="dense_baseline_top10",
            scenario_name=f"Baseline dense@{baseline_top_k} (no filter)",
            dataset=dataset,
            stage="direct",
            config={"retrieval_mode": "dense", "top_k": baseline_top_k, "filter": "none"},
            article_hit_key="direct_article_hit",
            law_hit_key="direct_law_hit",
            article_mrr_key="direct_article_mrr",
            baseline_pct=None,
        )
        baseline_pct = baseline_record["article_hit_pct"]
        rows.append(baseline_record)

        # --- best dense top-k ---------------------------------------------------------
        dense_candidates = direct_summary_df[
            (direct_summary_df["dataset"] == dataset)
            & (direct_summary_df["retrieval_mode"] == "dense")
            & (direct_summary_df["filter_name"] == "none")
        ].sort_values(["article_hit", "article_mrr", "top_k"], ascending=[False, False, True])
        if not dense_candidates.empty:
            best_dense = dense_candidates.iloc[0]
            rows.append(
                summarize_experiment(
                    _direct_rows_for_summary(direct_subset, best_dense),
                    experiment_name="dense_topk_curve",
                    scenario_name="Best dense top-k (no filter)",
                    dataset=dataset,
                    stage="direct",
                    config={"retrieval_mode": "dense", "top_k": int(best_dense["top_k"]), "filter": "none"},
                    article_hit_key="direct_article_hit",
                    law_hit_key="direct_law_hit",
                    article_mrr_key="direct_article_mrr",
                    baseline_pct=baseline_pct,
                )
            )

        # --- best dense within budget -------------------------------------------------
        budget_candidates = direct_summary_df[
            (direct_summary_df["dataset"] == dataset)
            & (direct_summary_df["retrieval_mode"] == "dense")
            & (direct_summary_df["top_k"] <= budget_top_k)
        ].sort_values(
            ["article_hit", "filter_excluded_rate", "article_mrr", "law_hit", "top_k"],
            ascending=[False, True, False, False, True],
        )
        if not budget_candidates.empty:
            best_budget = budget_candidates.iloc[0]
            rows.append(
                summarize_experiment(
                    _direct_rows_for_summary(direct_subset, best_budget),
                    experiment_name="best_dense_budget_top20",
                    scenario_name=f"Best direct budget top{budget_top_k}",
                    dataset=dataset,
                    stage="direct",
                    config={
                        "retrieval_mode": "dense",
                        "top_k": int(best_budget["top_k"]),
                        "filter": str(best_budget["filter_name"]),
                    },
                    article_hit_key="direct_article_hit",
                    law_hit_key="direct_law_hit",
                    article_mrr_key="direct_article_mrr",
                    baseline_pct=baseline_pct,
                )
            )

        # --- law_status=current filter ------------------------------------------------
        current_candidates = direct_summary_df[
            (direct_summary_df["dataset"] == dataset)
            & (direct_summary_df["retrieval_mode"] == "dense")
            & (direct_summary_df["top_k"] == baseline_top_k)
            & (direct_summary_df["filter_name"] == primary_current_filter)
        ]
        if not current_candidates.empty:
            current_row = current_candidates.iloc[0]
            rows.append(
                summarize_experiment(
                    _direct_rows_for_summary(direct_subset, current_row),
                    experiment_name="current_law_filter",
                    scenario_name="+ Filter law_status=current",
                    dataset=dataset,
                    stage="direct",
                    config={
                        "retrieval_mode": "dense",
                        "top_k": baseline_top_k,
                        "filter": primary_current_filter,
                    },
                    article_hit_key="direct_article_hit",
                    law_hit_key="direct_law_hit",
                    article_mrr_key="direct_article_mrr",
                    baseline_pct=baseline_pct,
                )
            )

        # --- best filter from sweep ---------------------------------------------------
        best_filter_candidates = direct_summary_df[
            (direct_summary_df["dataset"] == dataset)
            & (direct_summary_df["retrieval_mode"] == "dense")
            & (direct_summary_df["top_k"] == baseline_top_k)
        ].sort_values(
            ["article_hit", "filter_excluded_rate", "article_mrr", "law_hit"],
            ascending=[False, True, False, False],
        )
        if not best_filter_candidates.empty:
            best_filter = best_filter_candidates.iloc[0]
            rows.append(
                summarize_experiment(
                    _direct_rows_for_summary(direct_subset, best_filter),
                    experiment_name="best_filter_from_sweep",
                    scenario_name="+ Best filter from sweep",
                    dataset=dataset,
                    stage="direct",
                    config={
                        "retrieval_mode": "dense",
                        "top_k": baseline_top_k,
                        "filter": str(best_filter["filter_name"]),
                    },
                    article_hit_key="direct_article_hit",
                    law_hit_key="direct_law_hit",
                    article_mrr_key="direct_article_mrr",
                    baseline_pct=baseline_pct,
                )
            )

        # --- graph scenarios ----------------------------------------------------------
        graph_status = "run" if graph_enabled else "skipped"
        graph_reason = "" if graph_enabled else "Graph sweep disabled by profile"
        rows.append(
            summarize_experiment(
                _graph_baseline_subset(graph_subset, baseline_top_k=baseline_top_k, relation_set="default"),
                experiment_name="graph_default_seed3",
                scenario_name="+ Graph default seed=3",
                dataset=dataset,
                stage="graph",
                config={
                    "top_k": baseline_top_k,
                    "filter": "none",
                    "seed_k": 3,
                    "max_chunks_per_law": 2,
                    "min_confidence": 0.45,
                    "relations": "default",
                },
                article_hit_key="post_article_hit",
                law_hit_key="post_law_hit",
                article_mrr_key="post_article_mrr",
                baseline_pct=baseline_pct,
                status=graph_status,
                skip_reason=graph_reason,
            )
        )
        rows.append(
            summarize_experiment(
                _graph_baseline_subset(graph_subset, baseline_top_k=baseline_top_k, relation_set="references_only"),
                experiment_name="graph_references_only",
                scenario_name="+ Graph REFERENCES only",
                dataset=dataset,
                stage="graph",
                config={
                    "top_k": baseline_top_k,
                    "filter": "none",
                    "seed_k": 3,
                    "max_chunks_per_law": 2,
                    "min_confidence": 0.45,
                    "relations": "references_only",
                },
                article_hit_key="post_article_hit",
                law_hit_key="post_law_hit",
                article_mrr_key="post_article_mrr",
                baseline_pct=baseline_pct,
                status=graph_status,
                skip_reason=graph_reason,
            )
        )

        best_for_dataset = (
            best_graph_by_dataset_df[best_graph_by_dataset_df["dataset"] == dataset]
            if not best_graph_by_dataset_df.empty
            else pd.DataFrame()
        )
        if not best_for_dataset.empty:
            best_row_meta = best_for_dataset.iloc[0]
            rows.append(
                summarize_experiment(
                    _graph_rows_for_summary(graph_subset, best_row_meta),
                    experiment_name="best_graph_from_sweep",
                    scenario_name="+ Best graph from sweep",
                    dataset=dataset,
                    stage="graph",
                    config=graph_scenario_config(best_row_meta),
                    article_hit_key="post_article_hit",
                    law_hit_key="post_law_hit",
                    article_mrr_key="post_article_mrr",
                    baseline_pct=baseline_pct,
                )
            )

        low_noise_candidates = (
            graph_summary_df[
                (graph_summary_df["dataset"] == dataset)
                & (graph_summary_df["expansion_noise_ratio"].notna())
                & (graph_summary_df["expansion_noise_ratio"] <= low_noise_threshold)
            ].sort_values(
                ["post_article_hit", "delta_vs_direct", "graph_incremental_hits", "expansion_noise_ratio"],
                ascending=[False, False, False, True],
            )
            if not graph_summary_df.empty
            else pd.DataFrame()
        )
        if low_noise_candidates.empty:
            rows.append(
                summarize_experiment(
                    [],
                    experiment_name="best_graph_low_noise",
                    scenario_name="+ Best graph low-noise skipped",
                    dataset=dataset,
                    stage="graph",
                    config={"max_expansion_noise_ratio": low_noise_threshold},
                    article_hit_key="post_article_hit",
                    law_hit_key="post_law_hit",
                    article_mrr_key="post_article_mrr",
                    baseline_pct=baseline_pct,
                    status="skipped",
                    skip_reason=f"No graph configuration with expansion_noise_ratio <= {low_noise_threshold}",
                )
            )
        else:
            low_noise_row = low_noise_candidates.iloc[0]
            rows.append(
                summarize_experiment(
                    _graph_rows_for_summary(graph_subset, low_noise_row),
                    experiment_name="best_graph_low_noise",
                    scenario_name="+ Best graph low-noise",
                    dataset=dataset,
                    stage="graph",
                    config=graph_scenario_config(low_noise_row),
                    article_hit_key="post_article_hit",
                    law_hit_key="post_law_hit",
                    article_mrr_key="post_article_mrr",
                    baseline_pct=baseline_pct,
                )
            )

        # --- hybrid + downstream stages ----------------------------------------------
        if hybrid_available:
            hybrid_candidates = direct_summary_df[
                (direct_summary_df["dataset"] == dataset) & (direct_summary_df["retrieval_mode"] == "hybrid")
            ].sort_values(
                ["article_hit", "article_mrr", "top_k", "rrf_k"], ascending=[False, False, True, True]
            )
            if not hybrid_candidates.empty:
                hybrid_row = hybrid_candidates.iloc[0]
                rows.append(
                    summarize_experiment(
                        _direct_rows_for_summary(direct_subset, hybrid_row),
                        experiment_name="hybrid_if_available",
                        scenario_name="Hybrid best available",
                        dataset=dataset,
                        stage="direct",
                        config={
                            "retrieval_mode": "hybrid",
                            "top_k": int(hybrid_row["top_k"]),
                            "rrf_k": int(hybrid_row["rrf_k"]),
                            "filter": str(hybrid_row["filter_name"]),
                        },
                        article_hit_key="direct_article_hit",
                        law_hit_key="direct_law_hit",
                        article_mrr_key="direct_article_mrr",
                        baseline_pct=baseline_pct,
                    )
                )

            rows.append(
                _make_query_rewriting_scenario(
                    query_rewrite_subset=query_rewrite_subset,
                    dataset=dataset,
                    baseline_pct=baseline_pct,
                    query_rewriting_enabled=query_rewriting_enabled,
                    strategies=profile.query_rewriting_strategies,
                )
            )
            rows.append(
                _make_rerank_scenario(
                    rerank_subset=rerank_subset,
                    dataset=dataset,
                    baseline_pct=baseline_pct,
                    rerank_enabled=rerank_enabled,
                    input_k_values=profile.rerank_input_k_values,
                    output_k_values=profile.rerank_output_k_values,
                )
            )
        else:
            rows.append(
                summarize_experiment(
                    [],
                    experiment_name="hybrid_if_available",
                    scenario_name="Hybrid retrieval skipped",
                    dataset=dataset,
                    stage="direct",
                    config={"retrieval_mode": "hybrid"},
                    article_hit_key="direct_article_hit",
                    law_hit_key="direct_law_hit",
                    article_mrr_key="direct_article_mrr",
                    baseline_pct=baseline_pct,
                    status="skipped",
                    skip_reason="Index dense-only: sparse vector missing or embedder lacks sparse embeddings",
                )
            )
            rows.append(
                summarize_experiment(
                    [],
                    experiment_name="query_rewriting",
                    scenario_name="Query rewriting skipped",
                    dataset=dataset,
                    stage="query_rewriting",
                    config={
                        "prompt_version": QUERY_REWRITING_PROMPT_VERSION,
                        "strategies": list(profile.query_rewriting_strategies),
                    },
                    article_hit_key="direct_article_hit",
                    law_hit_key="direct_law_hit",
                    article_mrr_key="direct_article_mrr",
                    filter_excluded_key=None,
                    baseline_pct=baseline_pct,
                    status="skipped",
                    skip_reason="Hybrid retrieval unavailable; query rewriting H depends on best hybrid",
                )
            )
            rows.append(
                summarize_experiment(
                    [],
                    experiment_name="llm_rerank",
                    scenario_name="LLM rerank skipped",
                    dataset=dataset,
                    stage="rerank",
                    config={
                        "prompt_version": RERANK_PROMPT_VERSION,
                        "input_k": list(profile.rerank_input_k_values),
                        "output_k": list(profile.rerank_output_k_values),
                    },
                    article_hit_key="reranked_article_hit",
                    law_hit_key="reranked_law_hit",
                    article_mrr_key="reranked_article_mrr",
                    filter_excluded_key=None,
                    baseline_pct=baseline_pct,
                    status="skipped",
                    skip_reason="Hybrid retrieval unavailable; rerank G depends on best hybrid",
                )
            )

    return pd.DataFrame(rows)


def select_best_scenario(scenarios_df: pd.DataFrame, *, dataset: str | None = None) -> pd.Series | None:
    """Return the winning scenario row (article_hit then MRR then law_hit)."""
    if scenarios_df.empty:
        return None
    subset = scenarios_df[
        (scenarios_df["status"] == "run") & (scenarios_df["article_hit_pct"].notna())
    ]
    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]
    if subset.empty:
        return None
    return subset.sort_values(
        ["article_hit_pct", "article_mrr", "law_hit_pct"], ascending=[False, False, False]
    ).iloc[0]


def to_advanced_config_recommendation(
    *,
    scenarios_df: pd.DataFrame,
    datasets: Sequence[str],
    profile: DiagnosticProfile,
    advanced_rag_defaults: Mapping[str, Any],
    rerank_impact_df: pd.DataFrame | None = None,
    rerank_failure_rate: float = 0.0,
    query_rewriting_failure_rate: float = 0.0,
) -> dict[str, Any] | None:
    """Translate the waterfall winners into the JSON consumed by notebook 06.

    Returns None when no scenarios have been executed.

    `advanced_rag_defaults` provides default values from `AdvancedRagConfig`
    (top_k, rrf_k, graph_expansion_seed_k, ...). Only the picks that pass the
    promotion threshold are flipped on in the recommended config.
    """
    rows: list[dict[str, Any]] = []
    min_gain = profile.min_promotion_gain_pp
    for dataset in datasets:
        dense_best = _scenario(scenarios_df, dataset=dataset, experiment_name="dense_topk_curve")
        hybrid_best = _scenario(scenarios_df, dataset=dataset, experiment_name="hybrid_if_available")
        base_retrieval = (
            hybrid_best
            if hybrid_best is not None
            and (dense_best is None or float(hybrid_best["article_hit_pct"]) >= float(dense_best["article_hit_pct"]))
            else dense_best
        )
        base_cfg = _cfg(base_retrieval)
        base_hit = float(base_retrieval["article_hit_pct"]) if base_retrieval is not None else None

        filter_best = _scenario(scenarios_df, dataset=dataset, experiment_name="best_filter_from_sweep")
        graph_best = _scenario(scenarios_df, dataset=dataset, experiment_name="best_graph_low_noise")
        query_best = _scenario(scenarios_df, dataset=dataset, experiment_name="query_rewriting")
        rerank_best = _scenario(scenarios_df, dataset=dataset, experiment_name="llm_rerank")

        graph_gain = _gain(graph_best, base_retrieval)
        query_gain = _gain(query_best, base_retrieval)
        rerank_gain = _gain(rerank_best, base_retrieval)

        graph_cfg = _cfg(graph_best)
        query_cfg = _cfg(query_best)
        rerank_cfg = _cfg(rerank_best)
        filter_cfg = _cfg(filter_best)

        promote_graph = bool(
            graph_best is not None
            and graph_gain is not None
            and graph_gain >= min_gain
            and int(graph_best.get("n_filter_excluded") or 0) == 0
        )
        promote_query = bool(
            query_best is not None
            and query_gain is not None
            and query_gain >= min_gain
            and query_cfg.get("strategy") not in {None, "none"}
            and query_rewriting_failure_rate <= profile.query_rewriting_failure_warn_threshold
        )

        rerank_impact = (
            rerank_impact_df[rerank_impact_df["dataset"] == dataset]
            if rerank_impact_df is not None and not rerank_impact_df.empty
            else pd.DataFrame()
        )
        rerank_is_stable = bool(
            not rerank_impact.empty
            and float(rerank_impact.iloc[0].get("delta_vs_pre_article_hit", 0.0)) >= 0.0
            and int(rerank_impact.iloc[0].get("demoted", 0)) == 0
            and rerank_failure_rate <= profile.rerank_failure_warn_threshold
        )
        promote_rerank = bool(rerank_best is not None and rerank_is_stable)

        baseline_scenario = _scenario(scenarios_df, dataset=dataset, experiment_name="dense_baseline_top10")
        use_filter = bool(
            filter_best is not None
            and filter_cfg.get("filter") not in {None, "none"}
            and _gain(filter_best, baseline_scenario) is not None
            and _gain(filter_best, baseline_scenario) >= min_gain
            and int(filter_best.get("n_filter_excluded") or 0) == 0
        )

        config = {
            "metadata_filters_enabled": use_filter,
            "static_filters": FILTER_VARIANTS.get(str(filter_cfg.get("filter")), {}) if use_filter else {},
            "hybrid_enabled": base_cfg.get("retrieval_mode") == "hybrid",
            "top_k": int(base_cfg.get("top_k", advanced_rag_defaults.get("top_k", 10))),
            "rrf_k": int(
                base_cfg.get("rrf_k", advanced_rag_defaults.get("rrf_k", 60))
                or advanced_rag_defaults.get("rrf_k", 60)
            ),
            "graph_expansion_enabled": promote_graph,
            "graph_expansion_seed_k": int(
                graph_cfg.get("seed_k", advanced_rag_defaults.get("graph_expansion_seed_k", 3))
            ),
            "graph_expansion_relation_types": RELATION_TYPE_VARIANTS.get(
                str(graph_cfg.get("relations", "default")),
                advanced_rag_defaults.get("graph_expansion_relation_types", RELATION_TYPE_VARIANTS["default"]),
            ),
            "max_chunks_per_expanded_law": int(
                graph_cfg.get("max_chunks_per_law", advanced_rag_defaults.get("max_chunks_per_expanded_law", 2))
            ),
            "max_expanded_chunks_total": advanced_rag_defaults.get("max_expanded_chunks_total", 15),
            "min_edge_confidence": float(
                graph_cfg.get("min_confidence", advanced_rag_defaults.get("min_edge_confidence", 0.45))
            ),
            "rerank_enabled": promote_rerank,
            "rerank_input_k": int(
                rerank_cfg.get(
                    "rerank_input_k",
                    min(int(base_cfg.get("top_k", advanced_rag_defaults.get("top_k", 10))), 50),
                )
            ),
            "rerank_output_k": int(
                rerank_cfg.get("rerank_output_k", advanced_rag_defaults.get("rerank_output_k", 5))
            ),
        }
        rows.append(
            {
                "dataset": dataset,
                "base_retrieval_scenario": None if base_retrieval is None else str(base_retrieval["scenario_name"]),
                "base_article_hit_pct": base_hit,
                "graph_gain_pp": graph_gain,
                "query_rewriting_gain_pp": query_gain,
                "rerank_gain_pp_vs_dense10_baseline": rerank_gain,
                "rerank_failure_rate": rerank_failure_rate,
                "query_rewriting_failure_rate": query_rewriting_failure_rate,
                "promote_graph": promote_graph,
                "promote_query_rewriting": promote_query,
                "promote_rerank": promote_rerank,
                "promote_filter": use_filter,
                "query_rewriting_strategy": str(query_cfg.get("strategy", "none")) if promote_query else "none",
                "query_rewriting_note": (
                    "Not included in recommended_advanced_config because AdvancedRagConfig "
                    "does not expose query rewriting fields yet."
                ),
                "recommended_config": config,
            }
        )

    if not rows:
        return None

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": RETRIEVAL_EVALUATION_SCHEMA_VERSION,
        "created_at": created_at,
        "diagnostic_profile": profile.name,
        "min_promotion_gain_pp": profile.min_promotion_gain_pp,
        "dataset_recommendations": rows,
        "recommended_advanced_config": rows[0]["recommended_config"],
        "query_rewriting_recommendation": {
            "enabled": bool(rows[0]["promote_query_rewriting"]),
            "strategy": rows[0]["query_rewriting_strategy"],
            "note": "Diagnostic evidence only; add AdvancedRagConfig/runner support before applying to notebook 06.",
        },
        "selection_note": "Retrieval-only recommendation. Run notebook 06 smoke before full final analysis.",
    }


def _scenario(scenarios_df: pd.DataFrame, *, dataset: str, experiment_name: str) -> pd.Series | None:
    subset = scenarios_df[
        (scenarios_df["dataset"] == dataset)
        & (scenarios_df["experiment_name"] == experiment_name)
        & (scenarios_df["status"] == "run")
        & (scenarios_df["article_hit_pct"].notna())
    ]
    if subset.empty:
        return None
    return subset.sort_values(
        ["article_hit_pct", "article_mrr", "law_hit_pct"], ascending=[False, False, False]
    ).iloc[0]


def _cfg(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    value = row.get("config")
    return value if isinstance(value, dict) else {}


def _gain(candidate: pd.Series | None, baseline: pd.Series | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate["article_hit_pct"]) - float(baseline["article_hit_pct"])


def _direct_rows_for_summary(direct_subset: pd.DataFrame, row: pd.Series) -> list[dict[str, Any]]:
    if direct_subset.empty:
        return []
    mask = (
        (direct_subset["retrieval_mode"] == row["retrieval_mode"])
        & (direct_subset["filter_name"] == row["filter_name"])
        & (direct_subset["top_k"] == row["top_k"])
    )
    if "rrf_k" in direct_subset.columns and "rrf_k" in row and not pd.isna(row["rrf_k"]):
        mask &= direct_subset["rrf_k"] == row["rrf_k"]
    return direct_subset[mask].to_dict("records")


def _graph_rows_for_summary(graph_subset: pd.DataFrame, row: pd.Series) -> list[dict[str, Any]]:
    if graph_subset.empty:
        return []
    mask = (
        (graph_subset["retrieval_mode"] == row["retrieval_mode"])
        & (graph_subset["filter_name"] == row["filter_name"])
        & (graph_subset["top_k"] == row["top_k"])
        & (graph_subset["graph_expansion_seed_k"] == row["graph_expansion_seed_k"])
        & (graph_subset["max_chunks_per_expanded_law"] == row["max_chunks_per_expanded_law"])
        & (graph_subset["min_edge_confidence"] == row["min_edge_confidence"])
        & (graph_subset["relation_set_name"] == row["relation_set_name"])
    )
    if "rrf_k" in graph_subset.columns and "rrf_k" in row:
        if pd.isna(row["rrf_k"]):
            mask &= graph_subset["rrf_k"].isna()
        else:
            mask &= graph_subset["rrf_k"] == row["rrf_k"]
    if "graph_base_config_name" in graph_subset.columns and "graph_base_config_name" in row:
        mask &= graph_subset["graph_base_config_name"] == row["graph_base_config_name"]
    return graph_subset[mask].to_dict("records")


def _graph_baseline_subset(
    graph_subset: pd.DataFrame,
    *,
    baseline_top_k: int,
    relation_set: str,
) -> list[dict[str, Any]]:
    if graph_subset.empty:
        return []
    mask = (
        (graph_subset["retrieval_mode"] == "dense")
        & (graph_subset["top_k"] == baseline_top_k)
        & (graph_subset["filter_name"] == "none")
        & (graph_subset["graph_expansion_seed_k"] == 3)
        & (graph_subset["max_chunks_per_expanded_law"] == 2)
        & (graph_subset["min_edge_confidence"] == 0.45)
        & (graph_subset["relation_set_name"] == relation_set)
    )
    return graph_subset[mask].to_dict("records")


def _make_query_rewriting_scenario(
    *,
    query_rewrite_subset: pd.DataFrame,
    dataset: str,
    baseline_pct: float | None,
    query_rewriting_enabled: bool,
    strategies: Sequence[str],
) -> dict[str, Any]:
    if not query_rewriting_enabled:
        return summarize_experiment(
            [],
            experiment_name="query_rewriting",
            scenario_name="Query rewriting skipped",
            dataset=dataset,
            stage="query_rewriting",
            config={"prompt_version": QUERY_REWRITING_PROMPT_VERSION, "strategies": list(strategies)},
            article_hit_key="direct_article_hit",
            law_hit_key="direct_law_hit",
            article_mrr_key="direct_article_mrr",
            filter_excluded_key=None,
            baseline_pct=baseline_pct,
            status="skipped",
            skip_reason="Query rewriting disabled by profile",
        )
    if query_rewrite_subset.empty:
        return summarize_experiment(
            [],
            experiment_name="query_rewriting",
            scenario_name="Query rewriting skipped",
            dataset=dataset,
            stage="query_rewriting",
            config={"prompt_version": QUERY_REWRITING_PROMPT_VERSION, "strategies": list(strategies)},
            article_hit_key="direct_article_hit",
            law_hit_key="direct_law_hit",
            article_mrr_key="direct_article_mrr",
            filter_excluded_key=None,
            baseline_pct=baseline_pct,
            status="skipped",
            skip_reason="No query rewriting rows produced",
        )
    candidates = (
        query_rewrite_subset.groupby(["strategy", "top_k", "rrf_k"], dropna=False)
        .agg(
            article_hit=("direct_article_hit", "mean"),
            article_mrr=("direct_article_mrr", "mean"),
            law_hit=("direct_law_hit", "mean"),
        )
        .reset_index()
        .sort_values(["article_hit", "article_mrr", "strategy"], ascending=[False, False, True])
    )
    best = candidates.iloc[0]
    rows = query_rewrite_subset[
        (query_rewrite_subset["strategy"] == best["strategy"])
        & (query_rewrite_subset["top_k"] == best["top_k"])
        & (query_rewrite_subset["rrf_k"] == best["rrf_k"])
    ].to_dict("records")
    return summarize_experiment(
        rows,
        experiment_name="query_rewriting",
        scenario_name="+ Query rewriting",
        dataset=dataset,
        stage="query_rewriting",
        config={
            "retrieval_mode": "hybrid",
            "top_k": int(best["top_k"]),
            "rrf_k": int(best["rrf_k"]),
            "strategy": str(best["strategy"]),
            "prompt_version": QUERY_REWRITING_PROMPT_VERSION,
        },
        article_hit_key="direct_article_hit",
        law_hit_key="direct_law_hit",
        article_mrr_key="direct_article_mrr",
        filter_excluded_key=None,
        baseline_pct=baseline_pct,
    )


def _make_rerank_scenario(
    *,
    rerank_subset: pd.DataFrame,
    dataset: str,
    baseline_pct: float | None,
    rerank_enabled: bool,
    input_k_values: Sequence[int],
    output_k_values: Sequence[int],
) -> dict[str, Any]:
    if not rerank_enabled:
        return summarize_experiment(
            [],
            experiment_name="llm_rerank",
            scenario_name="LLM rerank skipped",
            dataset=dataset,
            stage="rerank",
            config={
                "prompt_version": RERANK_PROMPT_VERSION,
                "input_k": list(input_k_values),
                "output_k": list(output_k_values),
            },
            article_hit_key="reranked_article_hit",
            law_hit_key="reranked_law_hit",
            article_mrr_key="reranked_article_mrr",
            filter_excluded_key=None,
            baseline_pct=baseline_pct,
            status="skipped",
            skip_reason="Rerank disabled by profile",
        )
    if rerank_subset.empty:
        return summarize_experiment(
            [],
            experiment_name="llm_rerank",
            scenario_name="LLM rerank skipped",
            dataset=dataset,
            stage="rerank",
            config={
                "prompt_version": RERANK_PROMPT_VERSION,
                "input_k": list(input_k_values),
                "output_k": list(output_k_values),
            },
            article_hit_key="reranked_article_hit",
            law_hit_key="reranked_law_hit",
            article_mrr_key="reranked_article_mrr",
            filter_excluded_key=None,
            baseline_pct=baseline_pct,
            status="skipped",
            skip_reason="No rerank rows produced",
        )
    candidates = (
        rerank_subset.groupby(["top_k", "rrf_k", "rerank_input_k", "rerank_output_k"], dropna=False)
        .agg(
            article_hit=("reranked_article_hit", "mean"),
            article_mrr=("reranked_article_mrr", "mean"),
            law_hit=("reranked_law_hit", "mean"),
        )
        .reset_index()
        .sort_values(
            ["article_hit", "article_mrr", "rerank_input_k", "rerank_output_k"],
            ascending=[False, False, True, True],
        )
    )
    best = candidates.iloc[0]
    rows = rerank_subset[
        (rerank_subset["rerank_input_k"] == best["rerank_input_k"])
        & (rerank_subset["rerank_output_k"] == best["rerank_output_k"])
        & (rerank_subset["rrf_k"] == best["rrf_k"])
    ].to_dict("records")
    return summarize_experiment(
        rows,
        experiment_name="llm_rerank",
        scenario_name="+ LLM rerank",
        dataset=dataset,
        stage="rerank",
        config={
            "retrieval_mode": "hybrid",
            "top_k": int(best["top_k"]),
            "rrf_k": int(best["rrf_k"]),
            "rerank_input_k": int(best["rerank_input_k"]),
            "rerank_output_k": int(best["rerank_output_k"]),
            "prompt_version": RERANK_PROMPT_VERSION,
        },
        article_hit_key="reranked_article_hit",
        law_hit_key="reranked_law_hit",
        article_mrr_key="reranked_article_mrr",
        filter_excluded_key=None,
        baseline_pct=baseline_pct,
    )

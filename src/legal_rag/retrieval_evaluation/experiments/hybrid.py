"""Hybrid retrieval summary helpers.

Hybrid retrieval is executed by `run_direct_experiment` with `retrieval_mode="hybrid"`,
so this module only provides selection/summary helpers used by the waterfall and the
recommendation builder.
"""

from __future__ import annotations

import pandas as pd


def summarize_hybrid(direct_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the direct summary to dense+hybrid rows for the F-experiment view."""
    if direct_summary_df.empty:
        return direct_summary_df
    return direct_summary_df[direct_summary_df["retrieval_mode"].isin(["dense", "hybrid"])].copy()


def best_hybrid_rrf_k(direct_summary_df: pd.DataFrame, *, dataset: str) -> int | None:
    """Pick the winning hybrid `rrf_k` for a dataset, or None when hybrid is missing."""
    subset = direct_summary_df[
        (direct_summary_df["dataset"] == dataset)
        & (direct_summary_df["retrieval_mode"] == "hybrid")
        & (direct_summary_df["filter_name"] == "none")
    ].copy()
    if subset.empty:
        return None
    winner = subset.sort_values(
        ["article_hit", "article_mrr", "top_k", "rrf_k"],
        ascending=[False, False, True, True],
    ).iloc[0]
    rrf = winner.get("rrf_k")
    return None if pd.isna(rrf) else int(rrf)


def best_hybrid_config(direct_summary_df: pd.DataFrame, *, dataset: str) -> tuple[int, int] | None:
    """Pick the best (top_k, rrf_k) for hybrid, or None when hybrid is missing."""
    subset = direct_summary_df[
        (direct_summary_df["dataset"] == dataset)
        & (direct_summary_df["retrieval_mode"] == "hybrid")
        & (direct_summary_df["filter_name"] == "none")
    ].copy()
    if subset.empty:
        return None
    winner = subset.sort_values(
        ["article_hit", "article_mrr", "top_k", "rrf_k"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return int(winner["top_k"]), int(winner["rrf_k"])

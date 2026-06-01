"""Plot helpers for the retrieval-diagnostics notebook (06b).

Each `plot_*` builds one focused figure for one experiment section, returning
`(fig, ax)`. Functions accept an optional `ax` so notebooks can compose them in
a multi-axes grid when needed; the default style matches `03_indexing_contract`
(figsize ~ (8, 4), grid on).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Stable palette per experiment family — keeps the same color in every figure.
TECHNIQUE_COLORS: dict[str, str] = {
    "dense_baseline_top10": "#64748b",
    "dense_topk_curve": "#2563eb",
    "best_dense_budget_top20": "#0ea5e9",
    "current_law_filter": "#16a34a",
    "best_filter_from_sweep": "#22c55e",
    "graph_default_seed3": "#f97316",
    "graph_references_only": "#a855f7",
    "best_graph_from_sweep": "#dc2626",
    "best_graph_low_noise": "#7c2d12",
    "hybrid_if_available": "#0891b2",
    "llm_rerank": "#9333ea",
    "query_rewriting": "#ec4899",
}

RELATION_MARKERS: dict[str, str] = {
    "references_only": "s",
    "modified_by_only": "^",
    "inserted_by_only": "v",
    "replacement_only": "D",
    "amendment_only": "P",
    "default": "o",
}


def pct(value: float | int | None) -> str:
    """Format a fraction (0..1) or percentage (>1) as a human-readable string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value) * 100:.1f}%" if abs(float(value)) <= 1 else f"{float(value):.1f}%"


def _resolve_axes(ax: plt.Axes | None, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_direct_topk_curves(
    direct_summary_df: pd.DataFrame,
    *,
    dataset: str,
    ax: plt.Axes | None = None,
    budget_top_k: int | None = 20,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of article_hit@k per retrieval_mode + filter for one dataset."""
    fig, ax = _resolve_axes(ax, figsize=(8, 4))
    subset = direct_summary_df[direct_summary_df["dataset"] == dataset].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no direct data", ha="center", va="center")
        return fig, ax
    for (mode, filter_name, rrf_k), group in subset.groupby(
        ["retrieval_mode", "filter_name", "rrf_k"], dropna=False
    ):
        rrf_label = "" if pd.isna(rrf_k) else f" / rrf={int(rrf_k)}"
        label = f"{mode} / {filter_name}{rrf_label}"
        group = group.sort_values("top_k")
        linewidth = 2.5 if filter_name == "none" else 1.6
        ax.plot(group["top_k"], group["article_hit"] * 100, marker="o", linewidth=linewidth, label=label)
    if budget_top_k is not None:
        ax.axvline(budget_top_k, color="#94a3b8", linestyle=":", linewidth=1)
    ax.set_title(f"Direct retrieval — article_hit@k ({dataset})")
    ax.set_xlabel("top_k")
    ax.set_ylabel("article_hit %")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_filter_comparison(
    direct_summary_df: pd.DataFrame,
    *,
    dataset: str,
    top_k: int = 10,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of dense article_hit per filter at a fixed top_k."""
    fig, ax = _resolve_axes(ax, figsize=(8, 4))
    subset = direct_summary_df[
        (direct_summary_df["dataset"] == dataset)
        & (direct_summary_df["retrieval_mode"] == "dense")
        & (direct_summary_df["top_k"] == top_k)
    ].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no filter data", ha="center", va="center")
        return fig, ax
    subset = subset.sort_values("article_hit", ascending=False)
    colors = ["#22c55e" if name == "none" else "#0ea5e9" for name in subset["filter_name"]]
    ax.bar(subset["filter_name"], subset["article_hit"] * 100, color=colors)
    ax.set_title(f"Filter sweep dense@{top_k} — article_hit % ({dataset})")
    ax.set_ylabel("article_hit %")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for index, value in enumerate(subset["article_hit"] * 100):
        ax.text(index, float(value) + 0.8, f"{float(value):.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_hybrid_sweep(
    direct_summary_df: pd.DataFrame,
    *,
    dataset: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of hybrid article_hit vs top_k for each rrf_k."""
    fig, ax = _resolve_axes(ax, figsize=(8, 4))
    subset = direct_summary_df[
        (direct_summary_df["dataset"] == dataset)
        & (direct_summary_df["retrieval_mode"] == "hybrid")
        & (direct_summary_df["filter_name"] == "none")
    ].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no hybrid data", ha="center", va="center")
        return fig, ax
    for rrf_k, group in subset.groupby("rrf_k", dropna=False):
        group = group.sort_values("top_k")
        ax.plot(
            group["top_k"],
            group["article_hit"] * 100,
            marker="o",
            label=f"rrf_k={int(rrf_k) if not pd.isna(rrf_k) else 'n/a'}",
        )
    ax.set_title(f"Hybrid sweep — article_hit@k ({dataset})")
    ax.set_xlabel("top_k")
    ax.set_ylabel("article_hit %")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_graph_topk_configs(
    graph_summary_df: pd.DataFrame,
    *,
    dataset: str,
    top_n: int = 8,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of the top-N graph configurations colored by noise ratio."""
    fig, ax = _resolve_axes(ax, figsize=(9, 4.5))
    subset = graph_summary_df[graph_summary_df["dataset"] == dataset].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no graph data", ha="center", va="center")
        return fig, ax
    subset = subset.sort_values(
        ["post_article_hit", "delta_vs_direct", "graph_incremental_hits", "expansion_noise_ratio"],
        ascending=[False, False, False, True],
    ).head(top_n).copy()
    subset["label"] = subset.apply(
        lambda row: (
            f"{row['relation_set_name']} | f={row['filter_name']} k={int(row['top_k'])} "
            f"s={int(row['graph_expansion_seed_k'])} c={int(row['max_chunks_per_expanded_law'])} "
            f"conf={float(row['min_edge_confidence']):.2f}"
        ),
        axis=1,
    )
    noise = subset["expansion_noise_ratio"].fillna(0.0).clip(0, 1)
    colors = plt.cm.OrRd(0.25 + 0.65 * noise)
    ax.barh(subset["label"], subset["post_article_hit"] * 100, color=colors)
    ax.set_title(f"Top graph configs ({dataset})")
    ax.set_xlabel("post article_hit %")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    for index, row in enumerate(subset.itertuples(index=False)):
        ax.text(
            float(row.post_article_hit) * 100 + 0.7,
            index,
            f"+{float(row.delta_vs_direct) * 100:.1f}pp n={float(row.expansion_noise_ratio or 0):.2f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    return fig, ax


def plot_graph_tradeoff(
    graph_summary_df: pd.DataFrame,
    *,
    dataset: str,
    low_noise_threshold: float = 0.95,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter of (expansion_noise_ratio, delta_article_hit) per relation_set."""
    fig, ax = _resolve_axes(ax, figsize=(8, 4.5))
    subset = graph_summary_df[graph_summary_df["dataset"] == dataset].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no graph data", ha="center", va="center")
        return fig, ax
    for relation_set, group in subset.groupby("relation_set_name"):
        marker = RELATION_MARKERS.get(str(relation_set), "o")
        sizes = 30 + group["graph_incremental_hits"].fillna(0).astype(float) * 35
        ax.scatter(
            group["expansion_noise_ratio"].fillna(0.0),
            group["delta_vs_direct"] * 100,
            s=sizes,
            alpha=0.55,
            marker=marker,
            label=str(relation_set),
        )
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.axvline(low_noise_threshold, color="#dc2626", linestyle=":", linewidth=1)
    ax.set_title(f"Graph tradeoff — noise vs gain ({dataset})")
    ax.set_xlabel("expansion_noise_ratio")
    ax.set_ylabel("delta article_hit pp")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_rerank_impact(
    rerank_summary_df: pd.DataFrame,
    *,
    dataset: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of pre vs post rerank article_hit % per (input_k, output_k)."""
    fig, ax = _resolve_axes(ax, figsize=(9, 4.5))
    subset = rerank_summary_df[rerank_summary_df["dataset"] == dataset].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no rerank data", ha="center", va="center")
        return fig, ax
    subset = subset.sort_values(
        ["article_hit", "delta_vs_pre_article_hit"], ascending=[False, False]
    ).head(8)
    labels = [
        f"in={int(row.rerank_input_k)}->out={int(row.rerank_output_k)} rrf={int(row.rrf_k)}"
        for row in subset.itertuples(index=False)
    ]
    x = range(len(labels))
    width = 0.4
    ax.bar([i - width / 2 for i in x], subset["pre_article_hit"] * 100, width=width, label="pre", color="#94a3b8")
    ax.bar([i + width / 2 for i in x], subset["article_hit"] * 100, width=width, label="post rerank", color="#9333ea")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"LLM rerank impact ({dataset})")
    ax.set_ylabel("article_hit %")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_query_rewriting_strategies(
    query_rewriting_summary_df: pd.DataFrame,
    *,
    dataset: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of article_hit % per query rewriting strategy."""
    fig, ax = _resolve_axes(ax, figsize=(8, 4))
    subset = query_rewriting_summary_df[query_rewriting_summary_df["dataset"] == dataset].copy()
    if subset.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no query rewriting data", ha="center", va="center")
        return fig, ax
    subset = subset.sort_values(["strategy", "article_hit"], ascending=[True, False])
    subset = subset.drop_duplicates(subset=["strategy"], keep="first").set_index("strategy")
    colors = {
        "none": "#94a3b8",
        "rewrite": "#2563eb",
        "hyde": "#f97316",
        "multi_query": "#ec4899",
    }
    bar_colors = [colors.get(str(name), "#64748b") for name in subset.index]
    ax.bar(subset.index, subset["article_hit"] * 100, color=bar_colors)
    ax.set_title(f"Query rewriting strategies — article_hit % ({dataset})")
    ax.set_ylabel("article_hit %")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for index, value in enumerate(subset["article_hit"] * 100):
        ax.text(index, float(value) + 0.8, f"{float(value):.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_waterfall(
    scenarios_df: pd.DataFrame,
    *,
    dataset: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar plot of all the executed scenarios for one dataset."""
    subset = scenarios_df[
        (scenarios_df["dataset"] == dataset)
        & (scenarios_df["status"] == "run")
        & (scenarios_df["article_hit_pct"].notna())
    ].copy()
    if subset.empty:
        fig, ax = _resolve_axes(ax, figsize=(9, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no scenarios executed", ha="center", va="center")
        return fig, ax
    height = max(4.0, 0.45 * len(subset))
    fig, ax = _resolve_axes(ax, figsize=(9, height))
    labels = subset.apply(_short_scenario_label, axis=1)
    colors = [TECHNIQUE_COLORS.get(str(name), "#3b82b6") for name in subset["experiment_name"]]
    ax.barh(labels, subset["article_hit_pct"], color=colors)
    baseline_values = subset.loc[subset["experiment_name"] == "dense_baseline_top10", "article_hit_pct"]
    if not baseline_values.empty:
        ax.axvline(float(baseline_values.iloc[0]), color="#475569", linestyle="--", linewidth=1, label="baseline")
        ax.legend(fontsize=8)
    ax.set_title(f"Retrieval waterfall ({dataset})")
    ax.set_xlabel("article_hit %")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    for index, value in enumerate(subset["article_hit_pct"]):
        ax.text(float(value) + 0.7, index, f"{float(value):.1f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig, ax


def _short_scenario_label(row: pd.Series) -> str:
    name = str(row["experiment_name"])
    config: Mapping[str, Any] = row.get("config") or {}
    if name == "dense_baseline_top10":
        return "dense@10"
    if name == "dense_topk_curve":
        return f"best dense@{config.get('top_k', '?')}"
    if name == "best_dense_budget_top20":
        return f"budget@{config.get('top_k', '?')}\n{config.get('filter', '?')}"
    if name == "current_law_filter":
        return "law_status current"
    if name == "best_filter_from_sweep":
        return f"best filter\n{config.get('filter', '?')}"
    if name == "graph_default_seed3":
        return "graph default"
    if name == "graph_references_only":
        return "graph refs"
    if name == "best_graph_from_sweep":
        return f"best graph\n{config.get('relations', '?')} k={config.get('top_k', '?')}"
    if name == "best_graph_low_noise":
        return f"low-noise graph\n{config.get('relations', '?')} k={config.get('top_k', '?')}"
    if name == "hybrid_if_available":
        return "hybrid" if row.get("status") == "run" else "hybrid skipped"
    if name == "llm_rerank":
        return f"rerank {config.get('rerank_input_k', '?')}->{config.get('rerank_output_k', '?')}"
    if name == "query_rewriting":
        return f"qr {config.get('strategy', '?')}"
    return str(row["scenario_name"])

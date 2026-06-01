"""Plot helpers for the Advanced Graph RAG notebook (06).

Each `plot_*` builds one focused figure and returns `(fig, ax)` so the notebook
can call it in one line. Functions accept optional `ax` to compose grids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PIPELINE_STEPS: list[str] = [
    "Domanda",
    "Query rewriting\n(multi-query)",
    "Metadata\nfilters",
    "Qdrant retrieval\ndense / hybrid",
    "Graph expansion\nhop=1",
    "Deduplica\nchunk",
    "LLM rerank\nscore 0-2",
    "Contesto\nfinale",
    "Risposta + citazioni\n+ judge",
]

PHASE_COLORS = {
    "io": "#4e79a7",
    "transform": "#9c755f",
    "retrieval": "#f28e2b",
    "fusion": "#59a14f",
    "answer": "#e15759",
}

SIMPLE_COLOR = "#9aa0a6"
ADVANCED_COLOR = "#2f6f9f"
MCQ_COLOR = "#2f6f9f"
NO_HINT_COLOR = "#2a9d62"


def _resolve_axes(ax: plt.Axes | None, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _metric(summary: Mapping[str, Any] | None, dataset: str, key: str) -> float:
    if not summary:
        return 0.0
    block = summary.get(dataset) or {}
    value = block.get(key)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def plot_pipeline_diagram(
    steps: Sequence[str] | None = None,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Render the high-level Advanced Graph RAG pipeline as a horizontal flow."""
    steps = list(steps or DEFAULT_PIPELINE_STEPS)
    fig, ax = _resolve_axes(ax, figsize=(max(12, 1.4 * len(steps)), 2.6))
    ax.axis("off")
    box_w, box_h, y = 0.86, 0.34, 0.5
    palette = ["#4e79a7", "#9c755f", "#f28e2b", "#f28e2b", "#59a14f", "#59a14f", "#bc5090", "#59a14f", "#e15759"]
    colors = [palette[i % len(palette)] if i < len(palette) else PHASE_COLORS["fusion"] for i in range(len(steps))]
    for idx, (label, color) in enumerate(zip(steps, colors)):
        ax.add_patch(plt.Rectangle((idx - box_w / 2, y - box_h / 2), box_w, box_h,
                                    facecolor=color, edgecolor="#222", linewidth=1.0, alpha=0.92))
        ax.text(idx, y, label, ha="center", va="center", color="white", fontsize=9, weight="bold")
        if idx < len(steps) - 1:
            ax.annotate("", xy=(idx + 1 - box_w / 2 - 0.05, y),
                        xytext=(idx + box_w / 2 + 0.05, y),
                        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333"})
    ax.set_xlim(-0.65, len(steps) - 0.35)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, ax


def plot_simple_vs_advanced(
    simple_summary: Mapping[str, Any] | None,
    advanced_summary: Mapping[str, Any] | None,
    *,
    advanced_label: str = "Advanced RAG",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar of strict_accuracy MCQ + no_hint, Simple vs Advanced."""
    fig, ax = _resolve_axes(ax, figsize=(8, 4))
    if advanced_summary is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no advanced summary", ha="center", va="center")
        return fig, ax
    labels = ["MCQ", "No-hint"]
    advanced_values = [_metric(advanced_summary, "mcq", "strict_accuracy"),
                       _metric(advanced_summary, "no_hint", "strict_accuracy")]
    x = list(range(len(labels)))
    if simple_summary:
        simple_values = [_metric(simple_summary, "mcq", "strict_accuracy"),
                         _metric(simple_summary, "no_hint", "strict_accuracy")]
        ax.bar([i - 0.18 for i in x], simple_values, width=0.36, label="Simple RAG", color=SIMPLE_COLOR)
        ax.bar([i + 0.18 for i in x], advanced_values, width=0.36, label=advanced_label, color=ADVANCED_COLOR)
        ax.legend()
    else:
        ax.bar(labels, advanced_values, color=ADVANCED_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("strict accuracy")
    ax.set_title(f"Simple RAG vs {advanced_label} - strict accuracy")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    fig.tight_layout()
    return fig, ax


def plot_accuracy_by_level(
    summary: Mapping[str, Any] | None,
    *,
    datasets: Sequence[str] = ("mcq", "no_hint"),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of strict accuracy per difficulty level (L1..Ln) for each dataset."""
    if summary is None:
        fig, ax = _resolve_axes(ax, figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no summary", ha="center", va="center")
        return fig, ax
    level_set: set[str] = set()
    for dataset in datasets:
        level_set.update((summary.get(dataset) or {}).get("by_level", {}).keys())
    levels = sorted(level_set)
    if not levels:
        fig, ax = _resolve_axes(ax, figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no per-level data", ha="center", va="center")
        return fig, ax
    fig, ax = _resolve_axes(ax, figsize=(max(8, len(levels) * 1.2), 4))
    x = list(range(len(levels)))
    width = 0.36
    colors = {"mcq": MCQ_COLOR, "no_hint": NO_HINT_COLOR}
    for offset, dataset in zip([-width / 2, width / 2], datasets):
        values = [float(((summary.get(dataset) or {}).get("by_level", {}).get(level, {}) or {}).get("strict_accuracy") or 0)
                  for level in levels]
        ax.bar([i + offset for i in x], values, width=width, label=dataset, color=colors.get(dataset, "#888"))
    ax.set_xticks(x, levels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("strict accuracy")
    ax.set_title("Advanced RAG - strict accuracy per livello")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_run_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Two-panel: feature adoption counts + failure category distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not diagnostics:
        for ax in axes:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no diagnostics", ha="center", va="center")
        return fig, list(axes)
    features = {
        "metadata": int(diagnostics.get("metadata_filtered_rows") or 0),
        "hybrid": int(diagnostics.get("hybrid_rows") or 0),
        "graph": int(diagnostics.get("graph_expanded_rows") or 0),
        "rerank": int(diagnostics.get("reranked_rows") or 0),
        "ref hit": int(diagnostics.get("reference_law_hits") or 0),
    }
    axes[0].bar(list(features.keys()), list(features.values()), color="#4c78a8")
    axes[0].set_title("Feature attivate e reference hit")
    axes[0].tick_params(axis="x", rotation=20)
    if axes[0].containers:
        axes[0].bar_label(axes[0].containers[0])
    failures = {str(key): int(value) for key, value in (diagnostics.get("failure_category_counts") or {}).items()}
    if failures:
        axes[1].bar(list(failures.keys()), list(failures.values()), color="#e15759")
        axes[1].set_title("Failure categories")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].bar_label(axes[1].containers[0])
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "no failure categories", ha="center", va="center")
    fig.tight_layout()
    return fig, list(axes)


def plot_context_and_rerank(
    diagnostics: Mapping[str, Any] | None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Two-panel: context_sufficient_counts + rerank_score_distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if not diagnostics:
        for ax in axes:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no diagnostics", ha="center", va="center")
        return fig, list(axes)
    context_sufficient = {str(key): int(value) for key, value in (diagnostics.get("context_sufficient_counts") or {}).items()}
    if context_sufficient:
        axes[0].bar(list(context_sufficient.keys()), list(context_sufficient.values()),
                    color=["#59a14f", "#f0c14b", "#e15759"][: len(context_sufficient)])
        axes[0].set_title("Context sufficiency (no_hint)")
        axes[0].bar_label(axes[0].containers[0])
    else:
        axes[0].set_axis_off()
        axes[0].text(0.5, 0.5, "no context_sufficient data", ha="center", va="center")
    rerank_dist = {str(key): int(value) for key, value in (diagnostics.get("rerank_score_distribution") or {}).items()}
    if rerank_dist:
        axes[1].bar(list(rerank_dist.keys()), list(rerank_dist.values()), color="#9c6ade")
        axes[1].set_title("Rerank score distribution (0/1/2)")
        axes[1].bar_label(axes[1].containers[0])
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "rerank disattivato o nessuno score", ha="center", va="center")
    fig.tight_layout()
    return fig, list(axes)


def plot_process_counts(
    advanced_row: Mapping[str, Any] | None,
    *,
    simple_row: Mapping[str, Any] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of chunks-per-phase for a single advanced row; optional simple overlay."""
    fig, ax = _resolve_axes(ax, figsize=(9, 4))
    if not advanced_row:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no advanced row", ha="center", va="center")
        return fig, ax
    direct = len(set(advanced_row.get("retrieved_chunk_ids", [])) - set(advanced_row.get("graph_expanded_chunk_ids", [])))
    graph = len(advanced_row.get("graph_expanded_chunk_ids", []))
    rerank = len(advanced_row.get("reranked_chunk_ids", []))
    context = len(advanced_row.get("context_chunk_ids", []))
    advanced_counts = {"retrieval": direct, "graph expansion": graph, "rerank": rerank, "context finale": context}
    x = list(range(len(advanced_counts)))
    if simple_row:
        simple_counts = {
            "retrieval": len(simple_row.get("retrieved_chunk_ids", [])),
            "graph expansion": 0,
            "rerank": 0,
            "context finale": len(simple_row.get("context_chunk_ids", []) or simple_row.get("retrieved_chunk_ids", [])),
        }
        ax.bar([i - 0.18 for i in x], list(simple_counts.values()), width=0.36, label="simple-like", color=SIMPLE_COLOR)
        ax.bar([i + 0.18 for i in x], list(advanced_counts.values()), width=0.36, label="advanced", color=ADVANCED_COLOR)
        ax.legend()
    else:
        ax.bar(list(advanced_counts.keys()), list(advanced_counts.values()),
               color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
    ax.set_xticks(x, list(advanced_counts.keys()))
    ax.set_ylabel("# chunk")
    ax.set_title("Chunk per fase pipeline")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    return fig, ax


__all__ = [
    "DEFAULT_PIPELINE_STEPS",
    "plot_accuracy_by_level",
    "plot_context_and_rerank",
    "plot_pipeline_diagram",
    "plot_process_counts",
    "plot_run_diagnostics",
    "plot_simple_vs_advanced",
]

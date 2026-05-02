"""Scoring and aggregation helpers for oracle-context evaluation."""

from __future__ import annotations

import re
from typing import Any

from .models import MCQ_LABELS


def score_mcq_label(predicted_label: str | None, correct_label: str) -> tuple[int | None, str | None]:
    """Score an MCQ label deterministically."""
    label = str(predicted_label or "").strip().upper()
    if label not in MCQ_LABELS:
        return None, f"invalid_mcq_label: {predicted_label!r}"
    return (1 if label == correct_label else 0), None


def level_sort_key(level_name: str) -> tuple[int, Any]:
    """Sort L1-L4 levels numerically and unknown labels last."""
    match = re.match(r"^L(\d+)$", str(level_name).strip().upper())
    return (0, int(match.group(1))) if match else (1, str(level_name))


def aggregate_results(
    name: str,
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    max_score_per_row: int,
) -> dict[str, Any]:
    """Aggregate global and by-level metrics for a result set."""
    valid_scores = [int(row[score_key]) for row in rows if row.get(score_key) is not None]
    processed = len(rows)
    judged = len(valid_scores)
    score_sum = int(sum(valid_scores))
    max_score_sum = int(max_score_per_row * judged)
    out = {
        "dataset": name,
        "processed": processed,
        "judged": judged,
        "score_sum": score_sum,
        "max_score_sum": max_score_sum,
        "accuracy": (score_sum / max_score_sum) if max_score_sum else None,
        "mean_score": (score_sum / judged) if judged else None,
        "coverage": (judged / processed) if processed else None,
        "strict_accuracy": (score_sum / (processed * max_score_per_row)) if processed else None,
        "errors": sum(1 for row in rows if row.get("error")),
        "by_level": {},
    }
    by_level: dict[str, dict[str, Any]] = {}
    for row in rows:
        level = str(row.get("level") or "UNKNOWN")
        stats = by_level.setdefault(level, {"processed": 0, "judged": 0, "score_sum": 0, "errors": 0})
        stats["processed"] += 1
        if row.get("error"):
            stats["errors"] += 1
        if row.get(score_key) is not None:
            stats["judged"] += 1
            stats["score_sum"] += int(row[score_key])
    for level, stats in by_level.items():
        level_max = max_score_per_row * int(stats["judged"])
        level_processed_max = max_score_per_row * int(stats["processed"])
        stats["max_score_sum"] = level_max
        stats["accuracy"] = (stats["score_sum"] / level_max) if level_max else None
        stats["mean_score"] = (stats["score_sum"] / stats["judged"]) if stats["judged"] else None
        stats["coverage"] = (stats["judged"] / stats["processed"]) if stats["processed"] else None
        stats["strict_accuracy"] = (stats["score_sum"] / level_processed_max) if level_processed_max else None
        by_level[level] = stats
    out["by_level"] = dict(sorted(by_level.items(), key=lambda item: level_sort_key(item[0])))
    return out


def add_delta(no_context: dict[str, Any], oracle_context: dict[str, Any]) -> dict[str, Any]:
    """Compute oracle-minus-no-context deltas globally and by level."""
    delta = {
        "accuracy": _delta(oracle_context.get("accuracy"), no_context.get("accuracy")),
        "mean_score": _delta(oracle_context.get("mean_score"), no_context.get("mean_score")),
        "strict_accuracy": _delta(oracle_context.get("strict_accuracy"), no_context.get("strict_accuracy")),
        "coverage": _delta(oracle_context.get("coverage"), no_context.get("coverage")),
        "by_level": {},
    }
    levels = sorted(
        set((no_context.get("by_level") or {}).keys()) | set((oracle_context.get("by_level") or {}).keys()),
        key=level_sort_key,
    )
    for level in levels:
        left = (no_context.get("by_level") or {}).get(level, {})
        right = (oracle_context.get("by_level") or {}).get(level, {})
        delta["by_level"][level] = {
            "accuracy": _delta(right.get("accuracy"), left.get("accuracy")),
            "mean_score": _delta(right.get("mean_score"), left.get("mean_score")),
            "strict_accuracy": _delta(right.get("strict_accuracy"), left.get("strict_accuracy")),
            "coverage": _delta(right.get("coverage"), left.get("coverage")),
        }
    return delta


def _delta(right: Any, left: Any) -> float | None:
    if right is None or left is None:
        return None
    return float(right) - float(left)

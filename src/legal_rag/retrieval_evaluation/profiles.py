"""Diagnostic profile configuration for retrieval evaluation runs.

The notebook 06b dispatches each experiment based on the active profile, so the
profile drives both *what* runs and *how big* the sweeps are. Only two profiles
are exposed by default: a `light` profile that finishes quickly for iteration,
and a `full` profile that covers the matrices required by the spec's Quality
Gates for the definitive thesis run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ExperimentName = Literal["direct", "graph", "hybrid", "rerank", "query_rewriting"]

# Filter variants — name -> filter dict for Qdrant. Kept module-level because
# the notebook maps the winner scenario_name back to the Qdrant filter payload.
FILTER_VARIANTS: dict[str, dict[str, Any]] = {
    "none": {},
    "law_status_current": {"law_status": "current"},
    "index_views_current": {"index_views": "current"},
    "article_status_current": {"article_status": "current"},
    "law_article_status_current": {
        "law_status": "current",
        "article_status": "current",
    },
    "law_status_active": {"law_status": ["current", "partial"]},
    "article_status_active": {"article_status": ["current", "partial"]},
    "passage_status_active": {"passage_status": ["current", "partial"]},
    "index_views_not_explicitly_past": {"index_views": "not_explicitly_past"},
}

FILTER_AUDIT_FILTER_NAMES: tuple[str, ...] = (
    "none",
    "law_status_current",
    "article_status_current",
    "law_article_status_current",
    "law_status_active",
    "article_status_active",
    "passage_status_active",
    "index_views_current",
    "index_views_not_explicitly_past",
)
FILTER_AUDIT_EXACT_FILTER_NAMES: tuple[str, ...] = (
    "none",
    "index_views_current",
    "index_views_not_explicitly_past",
)

# Graph relation type variants — name -> ordered list of relation types.
RELATION_TYPE_VARIANTS: dict[str, list[str]] = {
    "references_only": ["REFERENCES"],
    "modified_by_only": ["MODIFIED_BY"],
    "inserted_by_only": ["INSERTED_BY"],
    "replacement_only": ["REPLACED_BY", "REPLACES"],
    "amendment_only": ["AMENDS", "INSERTS", "MODIFIED_BY", "INSERTED_BY"],
    "default": ["REFERENCES", "AMENDS", "INSERTS", "MODIFIED_BY", "INSERTED_BY"],
}

# Evaluation dataset specs: name -> {filename, question_key}.
DATASET_SPECS: dict[str, dict[str, str]] = {
    "no_hint": {"filename": "questions_no_hint.jsonl", "question_key": "question"},
    "mcq": {"filename": "questions_mcq.jsonl", "question_key": "question_stem"},
}


class DiagnosticProfile(BaseModel):
    """Sweep parameters for one retrieval-diagnostic run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str

    # Dataset selection
    datasets: list[str] = Field(default_factory=lambda: ["mcq", "no_hint"])
    benchmark_size: int | None = None

    # Experiment switches
    enabled_experiments: set[ExperimentName] = Field(
        default_factory=lambda: {"direct", "graph", "hybrid", "rerank", "query_rewriting"}
    )

    # Direct/hybrid sweeps
    top_k_values: list[int] = Field(default_factory=lambda: [10, 20, 50, 100])
    hybrid_top_k_values: list[int] = Field(default_factory=lambda: [10, 20, 50, 100])
    hybrid_rrf_k_values: list[int] = Field(default_factory=lambda: [30, 60, 90])
    filter_variants: list[str] = Field(default_factory=lambda: ["none", "law_status_current"])
    hybrid_filters_enabled: bool = False

    # Validity-filter audit
    exact_control_enabled: bool = False
    exact_control_top_k_values: list[int] = Field(default_factory=lambda: [10, 50])
    exact_control_filter_variants: list[str] = Field(
        default_factory=lambda: list(FILTER_AUDIT_EXACT_FILTER_NAMES)
    )
    bootstrap_resamples: int = Field(default=10_000, gt=0)
    bootstrap_seed: int = 42

    # Graph sweeps
    graph_base_configs: list[str] = Field(default_factory=lambda: ["dense@10", "hybrid_best"])
    graph_modes: list[str] = Field(default_factory=lambda: ["dense", "hybrid"])
    graph_top_k_values: list[int] = Field(default_factory=lambda: [10, 100])
    graph_seed_k_values: list[int] = Field(default_factory=lambda: [1, 3])
    graph_max_chunks_per_law_values: list[int] = Field(default_factory=lambda: [1, 2])
    graph_min_edge_confidence_values: list[float] = Field(default_factory=lambda: [0.45])
    graph_relation_set_names: list[str] = Field(default_factory=lambda: ["references_only", "default"])
    graph_filter_names: list[str] = Field(default_factory=lambda: ["none"])
    graph_expand_only_problem_cases: bool = True

    # Rerank
    rerank_input_k_values: list[int] = Field(default_factory=lambda: [50])
    rerank_output_k_values: list[int] = Field(default_factory=lambda: [3, 5, 10])
    rerank_question_sample: int = 30
    rerank_sample_seed: int = 42
    rerank_full_run: bool = False
    rerank_failure_warn_threshold: float = 0.05

    # Query rewriting
    query_rewriting_strategies: list[str] = Field(
        default_factory=lambda: ["none", "rewrite", "hyde", "multi_query"]
    )
    query_rewriting_question_sample: int = 30
    query_rewriting_sample_seed: int = 43
    query_rewriting_full_run: bool = False
    multi_query_n: int = 3
    query_rewriting_failure_warn_threshold: float = 0.05

    # Promotion thresholds (used by waterfall + recommendation)
    baseline_top_k: int = 10
    budget_top_k: int = 20
    low_noise_threshold: float = 0.95
    min_promotion_gain_pp: float = 1.0


PROFILES: dict[str, DiagnosticProfile] = {
    "light": DiagnosticProfile(
        name="light",
        description="Run rapida con sweep ridotti per iterazione quotidiana.",
        datasets=["no_hint"],
        enabled_experiments={"direct", "graph", "hybrid", "rerank", "query_rewriting"},
        top_k_values=[10, 50],
        hybrid_top_k_values=[10, 50],
        hybrid_rrf_k_values=[60],
        filter_variants=["none", "law_status_current"],
        graph_base_configs=["dense@10"],
        graph_modes=["dense"],
        graph_top_k_values=[10],
        graph_seed_k_values=[3],
        graph_max_chunks_per_law_values=[2],
        graph_min_edge_confidence_values=[0.45],
        graph_relation_set_names=["references_only", "default"],
        rerank_input_k_values=[50],
        rerank_output_k_values=[3, 5],
        rerank_question_sample=30,
        rerank_full_run=False,
        query_rewriting_strategies=["none", "rewrite", "hyde", "multi_query"],
        query_rewriting_question_sample=30,
        query_rewriting_full_run=False,
    ),
    "full": DiagnosticProfile(
        name="full",
        description="Run completa che copre i Quality Gates dello spec 06b.",
        datasets=["mcq", "no_hint"],
        enabled_experiments={"direct", "graph", "hybrid", "rerank", "query_rewriting"},
        top_k_values=[10, 20, 50, 100],
        hybrid_top_k_values=[10, 20, 50, 100],
        hybrid_rrf_k_values=[30, 60, 90],
        filter_variants=["none", "law_status_current", "index_views_current", "article_status_current"],
        graph_base_configs=["dense@10", "hybrid_best"],
        graph_modes=["dense", "hybrid"],
        graph_top_k_values=[10, 100],
        graph_seed_k_values=[1, 3],
        graph_max_chunks_per_law_values=[1, 2],
        graph_min_edge_confidence_values=[0.45],
        graph_relation_set_names=["references_only", "default"],
        rerank_input_k_values=[20, 50, 100],
        rerank_output_k_values=[3, 5, 10],
        rerank_question_sample=30,
        rerank_full_run=True,
        query_rewriting_strategies=["none", "rewrite", "hyde", "multi_query"],
        query_rewriting_question_sample=30,
        query_rewriting_full_run=True,
    ),
    "filter_audit": DiagnosticProfile(
        name="filter_audit",
        description="Deterministic validity-filter coverage and retrieval audit.",
        datasets=["mcq", "no_hint"],
        enabled_experiments={"direct", "hybrid"},
        top_k_values=[5, 10, 20, 50, 100],
        hybrid_top_k_values=[5, 10, 20, 50, 100],
        hybrid_rrf_k_values=[30],
        filter_variants=list(FILTER_AUDIT_FILTER_NAMES),
        hybrid_filters_enabled=True,
        exact_control_enabled=True,
        exact_control_top_k_values=[10, 50],
        exact_control_filter_variants=list(FILTER_AUDIT_EXACT_FILTER_NAMES),
        bootstrap_resamples=10_000,
        bootstrap_seed=42,
    ),
}


def env_flag(name: str, default: bool) -> bool:
    """Return a boolean env var, accepting truthy strings like 1/true/yes/on."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    """Return an int env var, falling back to default when unset or empty."""
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def env_optional_int(name: str, default: int | None) -> int | None:
    """Like env_int, but accepts the literal 'all' as None."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return None if value.strip().lower() == "all" else int(value)


def env_float(name: str, default: float) -> float:
    """Return a float env var, falling back to default when unset or empty."""
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def env_int_list(name: str, default: list[int]) -> list[int]:
    """Return a comma-separated int list env var."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def env_str_list(name: str, default: list[str]) -> list[str]:
    """Return a comma-separated string list env var."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def env_float_list(name: str, default: list[float]) -> list[float]:
    """Return a comma-separated float list env var."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def resolve_profile(name: str | None = None) -> DiagnosticProfile:
    """Pick the active diagnostic profile, env override `RETRIEVAL_DIAGNOSTIC_PROFILE`."""
    resolved = (
        name
        or os.environ.get("RETRIEVAL_DIAGNOSTIC_PROFILE")
        or os.environ.get("RETRIEVAL_DIAGNOSTICS_PROFILE")
        or "light"
    ).strip() or "light"
    if resolved not in PROFILES:
        raise ValueError(f"Unknown diagnostic profile {resolved!r}. Use one of {sorted(PROFILES)}")
    return PROFILES[resolved]


def unique_output_dir(
    base: Path,
    *,
    run_name: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Build a deduplicated timestamped output directory: base/run_name__timestamp."""
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = base / f"{run_name}__{timestamp}" if run_name else base
    if not target.exists():
        return target
    counter = 2
    while True:
        candidate = target.with_name(f"{target.name}__{counter}")
        if not candidate.exists():
            return candidate
        counter += 1

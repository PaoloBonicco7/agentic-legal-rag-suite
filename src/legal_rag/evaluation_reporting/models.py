"""Typed contracts for the evaluation reporting step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EVALUATION_REPORTING_SCHEMA_VERSION = "evaluation-reporting-v1"

METHODS = ("no_rag", "simple_rag", "advanced_rag")
DATASETS = ("mcq", "no_hint")
FAILURE_CATEGORIES = (
    "retrieval_miss",
    "context_noise",
    "abstention",
    "contradiction",
    "generation_error",
    "judge_error",
    "unknown",
)
REQUIRED_METRIC_FIELDS = (
    "processed",
    "judged",
    "score_sum",
    "max_score_sum",
    "accuracy",
    "mean_score",
    "coverage",
    "strict_accuracy",
    "errors",
    "by_level",
)

MethodName = Literal["no_rag", "simple_rag", "advanced_rag"]
DatasetName = Literal["mcq", "no_hint"]
FailureCategory = Literal[
    "retrieval_miss",
    "context_noise",
    "abstention",
    "contradiction",
    "generation_error",
    "judge_error",
    "unknown",
]


class EvaluationReportingConfig(BaseModel):
    """Runtime configuration for comparison report generation."""

    model_config = ConfigDict(extra="forbid")

    no_rag_dir: str = "data/baseline_runs/no_rag"
    simple_rag_dir: str = "data/rag_runs/simple"
    advanced_rag_dir: str = "data/rag_runs/advanced/default"
    evaluation_manifest_path: str = "data/evaluation_clean/evaluation_manifest.json"
    output_dir: str = "data/reports"
    allow_partial: bool = False
    max_examples_per_category: int = Field(default=3, ge=0)


class _Record(BaseModel):
    """Strict base model for exported JSON records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize records with JSON-compatible values and explicit nulls."""
        return self.model_dump(mode="json", exclude_none=False)


class ReportOutputFiles(_Record):
    """Generated artifact names for evaluation reporting."""

    comparison_summary: str = "comparison_summary.json"
    comparison_by_level: str = "comparison_by_level.json"
    failure_analysis: str = "failure_analysis.json"
    thesis_tables: str = "thesis_tables.md"
    quality_report: str = "quality_report.md"
    report_manifest: str = "report_manifest.json"


class SourceRunRecord(_Record):
    """Traceable reference to one compared run."""

    method: MethodName
    run_dir: str
    manifest_path: str
    summary_path: str
    mcq_results_path: str
    no_hint_results_path: str
    diagnostics_path: str | None
    manifest_hash: str | None
    summary_hash: str | None
    mcq_results_hash: str | None
    no_hint_results_hash: str | None
    diagnostics_hash: str | None
    run_name: str | None
    present: bool


class MetricDeltaRecord(_Record):
    """Delta between two metric dictionaries."""

    accuracy: float | None
    mean_score: float | None
    strict_accuracy: float | None
    coverage: float | None


class FailureExampleRecord(_Record):
    """Traceable representative failure row."""

    qid: str
    method: MethodName
    dataset: DatasetName
    level: str
    score: int | None
    judge_score: int | None
    error: str | None
    failure_category: FailureCategory
    run_name: str | None


class SafeReportingConfigRecord(_Record):
    """Run config exported to report manifests."""

    no_rag_dir: str
    simple_rag_dir: str
    advanced_rag_dir: str
    evaluation_manifest_path: str
    output_dir: str
    allow_partial: bool
    max_examples_per_category: int


class EvaluationReportingManifest(_Record):
    """Complete manifest for generated comparison reports."""

    schema_version: str
    created_at: str
    config: SafeReportingConfigRecord
    complete: bool
    evaluation_dataset_hash: str
    source_runs: dict[str, SourceRunRecord]
    source_hashes: dict[str, str]
    quality_issues: list[str]
    outputs: ReportOutputFiles
    output_hashes: dict[str, str]
    manifest_hash_note: str

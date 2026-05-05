"""Public API for 07 evaluation reporting."""

from __future__ import annotations

from .models import (
    EVALUATION_REPORTING_SCHEMA_VERSION,
    FAILURE_CATEGORIES,
    METHODS,
    REQUIRED_METRIC_FIELDS,
    EvaluationReportingConfig,
    EvaluationReportingManifest,
    FailureExampleRecord,
    ReportOutputFiles,
    SourceRunRecord,
)
from .runner import (
    build_comparison_by_level,
    build_comparison_summary,
    build_failure_analysis,
    build_thesis_tables,
    classify_failure,
    load_run_artifacts,
    run_evaluation_reporting,
)

__all__ = [
    "EVALUATION_REPORTING_SCHEMA_VERSION",
    "FAILURE_CATEGORIES",
    "METHODS",
    "REQUIRED_METRIC_FIELDS",
    "EvaluationReportingConfig",
    "EvaluationReportingManifest",
    "FailureExampleRecord",
    "ReportOutputFiles",
    "SourceRunRecord",
    "build_comparison_by_level",
    "build_comparison_summary",
    "build_failure_analysis",
    "build_thesis_tables",
    "classify_failure",
    "load_run_artifacts",
    "run_evaluation_reporting",
]

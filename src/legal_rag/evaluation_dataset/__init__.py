"""Public API for phase 2 evaluation dataset normalization."""

from __future__ import annotations

from .export import run_evaluation_dataset
from .models import (
    EVALUATION_SCHEMA_VERSION,
    EXPECTED_OPTION_LABELS,
    EvaluationDatasetConfig,
    McqQuestionRecord,
    NoHintQuestionRecord,
)
from .parsing import (
    build_mcq_record,
    build_no_hint_record,
    normalize_references,
    parse_mcq_question,
    validate_alignment,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EXPECTED_OPTION_LABELS",
    "EvaluationDatasetConfig",
    "McqQuestionRecord",
    "NoHintQuestionRecord",
    "build_mcq_record",
    "build_no_hint_record",
    "normalize_references",
    "parse_mcq_question",
    "run_evaluation_dataset",
    "validate_alignment",
]

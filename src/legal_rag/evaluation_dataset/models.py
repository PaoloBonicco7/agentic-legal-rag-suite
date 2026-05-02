"""Typed contracts for the clean evaluation question dataset."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

EVALUATION_SCHEMA_VERSION = "evaluation-dataset-v1"
EXPECTED_OPTION_LABELS = ("A", "B", "C", "D", "E", "F")


class EvaluationDatasetConfig(BaseModel):
    """Runtime configuration for the deterministic evaluation dataset build."""

    mcq_source: str = "data/evaluation/questions.csv"
    no_hint_source: str = "data/evaluation/questions_no_hint.csv"
    output_dir: str = "data/evaluation_clean"
    expected_records: int = Field(default=100, gt=0)


class _Record(BaseModel):
    """Strict Pydantic base model for exported evaluation records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize records with JSON-compatible values and explicit nulls."""
        return self.model_dump(mode="json", exclude_none=False)


class McqQuestionRecord(_Record):
    """One normalized multiple-choice benchmark question."""

    qid: str = Field(min_length=1)
    source_position: int = Field(gt=0)
    level: str = Field(min_length=1)
    question_stem: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=1)
    correct_label: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    expected_references: list[str] = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: dict[str, str]) -> dict[str, str]:
        """Ensure MCQ options are complete, ordered and non-empty."""
        labels = tuple(value.keys())
        if labels != EXPECTED_OPTION_LABELS:
            raise ValueError(f"options must contain ordered labels {EXPECTED_OPTION_LABELS!r}")
        if any(not text.strip() for text in value.values()):
            raise ValueError("option texts must be non-empty")
        return value

    @field_validator("correct_label")
    @classmethod
    def _validate_correct_label(cls, value: str) -> str:
        """Ensure the correct label belongs to the fixed MCQ label set."""
        label = value.strip().upper()
        if label not in EXPECTED_OPTION_LABELS:
            raise ValueError(f"correct_label must be one of {EXPECTED_OPTION_LABELS!r}")
        return label


class NoHintQuestionRecord(_Record):
    """One normalized open-answer benchmark question linked to its MCQ source."""

    qid: str = Field(min_length=1)
    source_position: int = Field(gt=0)
    level: str = Field(min_length=1)
    question: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    expected_references: list[str] = Field(min_length=1)
    linked_mcq_qid: str = Field(min_length=1)


def mcq_question_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize an MCQ question record."""
    return McqQuestionRecord.model_validate(data).to_json_record()


def no_hint_question_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a no-hint question record."""
    return NoHintQuestionRecord.model_validate(data).to_json_record()

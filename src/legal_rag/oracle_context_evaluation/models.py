"""Typed contracts for the oracle-context evaluation step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ORACLE_CONTEXT_SCHEMA_VERSION = "oracle-context-evaluation-v1"
ORACLE_PROMPT_VERSION = "oracle-context-prompts-v1"
DEFAULT_CHAT_MODEL = "SLURM.gpt-oss:120b"
MCQ_LABELS = ("A", "B", "C", "D", "E", "F")
RunName = Literal[
    "mcq_no_context",
    "mcq_oracle_context",
    "no_hint_no_context",
    "no_hint_oracle_context",
]


class OracleEvaluationConfig(BaseModel):
    """Runtime configuration for controlled oracle-context evaluation."""

    evaluation_dir: str = "data/evaluation_clean"
    laws_dir: str = "data/laws_dataset_clean"
    output_dir: str = "data/evaluation_runs/oracle_context"
    env_file: str | None = ".env"
    api_url: str | None = None
    api_key: str | None = None
    base_url: str = "https://utopia.hpc4ai.unito.it/api"
    api_mode: Literal["openai", "ollama"] = "ollama"
    chat_model: str = DEFAULT_CHAT_MODEL
    judge_model: str | None = None
    timeout_seconds: int = Field(default=120, gt=0)
    start: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, gt=0)
    smoke: bool = False
    retry_attempts: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    prompt_version: str = ORACLE_PROMPT_VERSION

    @field_validator("judge_model")
    @classmethod
    def _empty_judge_model_to_none(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @property
    def resolved_judge_model(self) -> str:
        """Return the judge model, falling back to the answer model."""
        return self.judge_model or self.chat_model

    @property
    def effective_limit(self) -> int | None:
        """Return the selected row limit, using smoke mode as a one-row run."""
        return 1 if self.smoke else self.limit


class _Record(BaseModel):
    """Strict base model for exported JSON records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize records with JSON-compatible values and explicit nulls."""
        return self.model_dump(mode="json", exclude_none=False)


class ResolvedReference(_Record):
    """One source-of-truth legal article resolved from a dataset reference."""

    reference_text: str
    law_id: str
    law_title: str
    article_id: str
    article_label_norm: str
    article_text: str


class OracleContextRecord(_Record):
    """Oracle legal context for one evaluation question."""

    qid: str
    level: str
    expected_references: list[str]
    resolved_references: list[ResolvedReference]
    context_article_ids: list[str]
    context_text: str
    context_hash: str | None
    error: str | None


class McqAnswerOutput(_Record):
    """Structured MCQ output produced by the answer model."""

    answer_label: str
    short_rationale: str | None = None

    @field_validator("answer_label")
    @classmethod
    def _validate_answer_label(cls, value: str) -> str:
        label = str(value or "").strip().upper()
        if label not in MCQ_LABELS:
            raise ValueError(f"answer_label must be one of {MCQ_LABELS!r}")
        return label


class NoHintAnswerOutput(_Record):
    """Structured open answer produced by the answer model."""

    answer_text: str = ""
    short_rationale: str | None = None

    @field_validator("answer_text")
    @classmethod
    def _normalize_answer_text(cls, value: str) -> str:
        return str(value or "").strip()


class JudgeOutput(_Record):
    """Structured no-hint semantic judge result on a 0-2 scale."""

    score: Literal[0, 1, 2]
    explanation: str = Field(min_length=1)

    @field_validator("explanation")
    @classmethod
    def _normalize_explanation(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("explanation must be non-empty")
        return text


class McqResultRow(_Record):
    """Row-level result for one MCQ run."""

    qid: str
    level: str
    question: str
    options: dict[str, str]
    correct_label: str
    predicted_label: str | None
    score: int | None
    context_article_ids: list[str]
    error: str | None


class NoHintResultRow(_Record):
    """Row-level result for one no-hint run."""

    qid: str
    level: str
    question: str
    predicted_answer: str | None
    correct_answer: str
    judge_score: int | None
    judge_explanation: str | None
    context_article_ids: list[str]
    error: str | None

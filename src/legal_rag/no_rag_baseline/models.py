"""Typed contracts for the no-RAG baseline step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL

NO_RAG_SCHEMA_VERSION = "no-rag-baseline-v1"
NO_RAG_PROMPT_VERSION = "no-rag-prompts-v1"


class NoRagConfig(BaseModel):
    """Runtime configuration for model-only baseline evaluation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_dir: str = "data/evaluation_clean"
    output_dir: str = "data/baseline_runs/no_rag"
    env_file: str | None = ".env"
    api_url: str | None = None
    api_key: str | None = None
    base_url: str = "https://utopia.hpc4ai.unito.it/api"
    chat_model: str = DEFAULT_CHAT_MODEL
    judge_model: str | None = None
    timeout_seconds: int = Field(default=120, gt=0)
    start: int = Field(default=0, ge=0)
    benchmark_size: int | None = Field(default=None, gt=0)
    smoke: bool = False
    retry_attempts: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    random_seed: int | None = None
    prompt_version: str = NO_RAG_PROMPT_VERSION

    @field_validator("judge_model")
    @classmethod
    def _empty_judge_model_to_none(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @property
    def resolved_judge_model(self) -> str:
        """Return the judge model, falling back to the answer model."""
        return self.judge_model or self.chat_model

    @property
    def effective_benchmark_size(self) -> int | None:
        """Return the row limit, using smoke mode as a one-row run."""
        return 1 if self.smoke else self.benchmark_size


class _Record(BaseModel):
    """Strict base model for exported JSON records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize records with JSON-compatible values and explicit nulls."""
        return self.model_dump(mode="json", exclude_none=False)


class McqResultRow(_Record):
    """Row-level result for one no-RAG MCQ question."""

    qid: str
    level: str
    question: str
    options: dict[str, str]
    predicted_label: str | None
    correct_label: str
    score: int | None
    error: str | None


class NoHintResultRow(_Record):
    """Row-level result for one no-RAG open-answer question."""

    qid: str
    level: str
    question: str
    predicted_answer: str | None
    correct_answer: str
    judge_score: int | None
    judge_explanation: str | None
    error: str | None


class SafeNoRagConfigRecord(_Record):
    """Run config exported to manifests without secret values."""

    evaluation_dir: str
    output_dir: str
    env_file: str | None
    api_url: str | None
    api_key_present: bool
    base_url: str
    chat_model: str
    judge_model: str | None
    timeout_seconds: int
    start: int
    benchmark_size: int | None
    smoke: bool
    retry_attempts: int
    max_concurrency: int
    random_seed: int | None
    prompt_version: str


class NoRagModelIdentities(_Record):
    """Model names used by a no-RAG run."""

    answer_model: str
    judge_model: str


class NoRagConnectionRecord(_Record):
    """Connection metadata exported without secret values."""

    client: Literal["utopia", "injected"]
    api_url: str | None = None
    base_url: str | None = None
    api_key_present: bool = False
    env_file: str | None = None
    env_file_loaded: bool = False
    env_keys_loaded: list[str] = Field(default_factory=list)


class NoRagOutputFiles(_Record):
    """Generated artifact names for a no-RAG run."""

    mcq_results: str = "mcq_results.jsonl"
    no_hint_results: str = "no_hint_results.jsonl"
    no_rag_summary: str = "no_rag_summary.json"
    quality_report: str = "quality_report.md"
    no_rag_manifest: str = "no_rag_manifest.json"


class NoRagRunCounts(_Record):
    """Row and error counts for a no-RAG run."""

    mcq: int
    no_hint: int
    mcq_errors: int
    no_hint_errors: int


class NoRagLevelMetrics(_Record):
    """Aggregate metrics for one difficulty level."""

    processed: int
    judged: int
    score_sum: int
    errors: int
    max_score_sum: int
    accuracy: float | None
    mean_score: float | None
    coverage: float | None
    strict_accuracy: float | None


class NoRagMetricSummary(_Record):
    """Aggregate metrics for one no-RAG result family."""

    dataset: Literal["mcq", "no_hint"]
    processed: int
    judged: int
    score_sum: int
    max_score_sum: int
    accuracy: float | None
    mean_score: float | None
    coverage: float | None
    strict_accuracy: float | None
    errors: int
    by_level: dict[str, NoRagLevelMetrics]


class NoRagSummary(_Record):
    """Complete aggregate metric contract for the no-RAG baseline."""

    mcq: NoRagMetricSummary
    no_hint: NoRagMetricSummary


class NoRagManifest(_Record):
    """Complete manifest for a no-RAG baseline run."""

    schema_version: str
    created_at: str
    config: SafeNoRagConfigRecord
    models: NoRagModelIdentities
    connection: NoRagConnectionRecord
    prompt_version: str
    random_seed: int | None
    source_hashes: dict[str, str]
    upstream_manifests: dict[str, Any]
    counts: NoRagRunCounts
    outputs: NoRagOutputFiles
    output_hashes: dict[str, str]
    summary: NoRagSummary
    manifest_hash_note: str

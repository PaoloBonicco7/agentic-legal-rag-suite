"""Typed contracts for the simple RAG baseline step."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL, MCQ_LABELS

SIMPLE_RAG_SCHEMA_VERSION = "simple-rag-v1"
SIMPLE_RAG_PROMPT_VERSION = "simple-rag-prompts-v1"


class SimpleRagConfig(BaseModel):
    """Runtime configuration for dense-retrieval RAG evaluation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_dir: str = "data/evaluation_clean"
    index_dir: str = "data/indexes/qdrant"
    index_manifest_path: str = "data/indexing_runs/<latest>/index_manifest.json"
    output_dir: str = "data/rag_runs/simple"
    collection_name: str = "legal_chunks"
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
    show_progress: bool = False
    progress_interval: int = Field(default=5, ge=1)
    prompt_version: str = SIMPLE_RAG_PROMPT_VERSION
    top_k: int = Field(default=5, gt=0)
    max_context_chunks: int = Field(default=5, gt=0)
    max_context_chars: int = Field(default=16000, gt=0)
    static_filters: dict[str, Any] = Field(default_factory=dict)

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


class Citation(_Record):
    """One grounded citation resolved from a retrieved context chunk."""

    law_id: str
    article_id: str
    chunk_id: str
    chunk_text: str


class RetrievedChunkRecord(_Record):
    """One retrieved chunk persisted in row-level traces."""

    chunk_id: str
    score: float
    text: str
    payload: dict[str, Any]


class SimpleMcqAnswerOutput(_Record):
    """Structured MCQ output produced by the answer model."""

    answer_label: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    short_rationale: str | None = None

    @field_validator("answer_label")
    @classmethod
    def _validate_answer_label(cls, value: str) -> str:
        label = str(value or "").strip().upper()
        if label not in MCQ_LABELS:
            raise ValueError(f"answer_label must be one of {MCQ_LABELS!r}")
        return label

    @field_validator("citation_chunk_ids")
    @classmethod
    def _normalize_citation_ids(cls, value: list[str]) -> list[str]:
        return _unique_non_empty(value)


class SimpleNoHintAnswerOutput(_Record):
    """Structured open answer produced by the answer model."""

    answer_text: str = ""
    citation_chunk_ids: list[str] = Field(default_factory=list)
    short_rationale: str | None = None

    @field_validator("answer_text")
    @classmethod
    def _normalize_answer_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("citation_chunk_ids")
    @classmethod
    def _normalize_citation_ids(cls, value: list[str]) -> list[str]:
        return _unique_non_empty(value)


class McqResultRow(_Record):
    """Row-level result for one simple-RAG MCQ question."""

    qid: str
    level: str
    question: str
    retrieved_chunk_ids: list[str]
    retrieved_law_ids: list[str]
    context_chunk_ids: list[str]
    retrieved_count: int
    context_count: int
    answer: str | None
    citations: list[Citation]
    options: dict[str, str]
    correct_label: str
    predicted_label: str | None
    score: int | None
    error: str | None


class NoHintResultRow(_Record):
    """Row-level result for one simple-RAG open-answer question."""

    qid: str
    level: str
    question: str
    retrieved_chunk_ids: list[str]
    retrieved_law_ids: list[str]
    context_chunk_ids: list[str]
    retrieved_count: int
    context_count: int
    answer: str | None
    citations: list[Citation]
    predicted_answer: str | None
    correct_answer: str
    judge_score: int | None
    judge_explanation: str | None
    error: str | None


def _unique_non_empty(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out

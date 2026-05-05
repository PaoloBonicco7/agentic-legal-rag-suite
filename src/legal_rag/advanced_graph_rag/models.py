"""Typed contracts for the advanced graph-aware RAG step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legal_rag.laws_preprocessing.models import ALLOWED_RELATION_TYPES
from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL, MCQ_LABELS
from legal_rag.simple_rag.models import Citation, RetrievedChunkRecord

ADVANCED_RAG_SCHEMA_VERSION = "advanced-graph-rag-v1"
ADVANCED_RAG_PROMPT_VERSION = "advanced-rag-prompts-v1"

RetrievalMode = Literal["dense", "hybrid"]
FailureCategory = Literal[
    "retrieval_miss",
    "context_noise",
    "abstention",
    "contradiction",
    "generation_error",
    "judge_error",
    "unknown",
]


class AdvancedRagConfig(BaseModel):
    """Runtime configuration for advanced graph-aware RAG evaluation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_dir: str = "data/evaluation_clean"
    laws_dir: str = "data/laws_dataset_clean"
    index_dir: str = "data/indexes/qdrant"
    index_manifest_path: str = "data/indexing_runs/<latest>/index_manifest.json"
    simple_rag_manifest_path: str = "data/rag_runs/simple/simple_rag_manifest.json"
    output_root: str = "data/rag_runs/advanced"
    collection_name: str = "legal_chunks"
    run_name: str = "default"
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
    prompt_version: str = ADVANCED_RAG_PROMPT_VERSION

    metadata_filters_enabled: bool = True
    hybrid_enabled: bool = True
    graph_expansion_enabled: bool = True
    rerank_enabled: bool = True

    static_filters: dict[str, Any] = Field(default_factory=lambda: {"law_status": "current"})
    top_k: int = Field(default=10, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    graph_expansion_seed_k: int = Field(default=5, gt=0)
    graph_expansion_relation_types: list[str] = Field(default_factory=lambda: sorted(ALLOWED_RELATION_TYPES))
    max_chunks_per_expanded_law: int = Field(default=2, gt=0)
    graph_expansion_hops: int = Field(default=1, ge=1)
    rerank_input_k: int = Field(default=20, gt=0)
    rerank_output_k: int = Field(default=5, gt=0)
    max_context_chars: int = Field(default=16000, gt=0)

    @field_validator("judge_model")
    @classmethod
    def _empty_judge_model_to_none(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("run_name")
    @classmethod
    def _non_empty_run_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("run_name must be non-empty")
        return text

    @field_validator("graph_expansion_relation_types")
    @classmethod
    def _valid_relation_types(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            relation_type = str(value or "").strip().upper()
            if relation_type not in ALLOWED_RELATION_TYPES:
                raise ValueError(f"Unsupported relation_type: {value!r}")
            if relation_type not in seen:
                out.append(relation_type)
                seen.add(relation_type)
        return out

    @model_validator(mode="after")
    def _poc_hops_only(self) -> "AdvancedRagConfig":
        if self.graph_expansion_hops != 1:
            raise ValueError("graph_expansion_hops must be 1 for the PoC")
        return self

    @property
    def resolved_judge_model(self) -> str:
        """Return the judge model, falling back to the answer model."""
        return self.judge_model or self.chat_model

    @property
    def effective_benchmark_size(self) -> int | None:
        """Return the row limit, using smoke mode as a one-row run."""
        return 1 if self.smoke else self.benchmark_size

    @property
    def output_dir(self) -> str:
        """Return the concrete run output directory."""
        return f"{self.output_root.rstrip('/')}/{self.run_name}"

    @property
    def active_static_filters(self) -> dict[str, Any]:
        """Return metadata filters actually applied to retrieval."""
        return dict(self.static_filters) if self.metadata_filters_enabled else {}


class InteractiveRagConfig(AdvancedRagConfig):
    """Runtime configuration for one-off interactive advanced RAG questions."""

    run_name: str = "interactive"
    hybrid_enabled: bool = False
    max_concurrency: int = Field(default=1, ge=1)


class _Record(BaseModel):
    """Strict base model for exported JSON records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize records with JSON-compatible values and explicit nulls."""
        return self.model_dump(mode="json", exclude_none=False)


class GraphRelationUsed(_Record):
    """One explicit legal graph edge consumed during expansion."""

    source_law_id: str
    target_law_id: str
    relation_type: str


class RerankScore(_Record):
    """One LLM relevance score for a retrieved candidate."""

    chunk_id: str
    score: Literal[0, 1, 2]


class RerankOutput(_Record):
    """Structured reranker output."""

    scores: list[RerankScore] = Field(default_factory=list)


class AdvancedMcqAnswerOutput(_Record):
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


class AdvancedNoHintAnswerOutput(_Record):
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


class _AdvancedTrace(_Record):
    """Advanced row-level retrieval and explanation trace."""

    retrieved_chunk_ids: list[str]
    retrieved_law_ids: list[str]
    context_chunk_ids: list[str]
    retrieved_count: int
    context_count: int
    metadata_filters: dict[str, Any]
    retrieval_mode: RetrievalMode
    graph_expanded_law_ids: list[str]
    graph_expanded_chunk_ids: list[str]
    graph_relations_used: list[GraphRelationUsed]
    reranked_chunk_ids: list[str]
    rerank_scores: list[int]
    context_included_count: int
    reference_law_hit: bool
    failure_category: FailureCategory | None


class AdvancedMcqResultRow(_AdvancedTrace):
    """Row-level result for one advanced-RAG MCQ question."""

    qid: str
    level: str
    question: str
    answer: str | None
    citations: list[Citation]
    options: dict[str, str]
    correct_label: str
    predicted_label: str | None
    score: int | None
    error: str | None


class AdvancedNoHintResultRow(_AdvancedTrace):
    """Row-level result for one advanced-RAG open-answer question."""

    qid: str
    level: str
    question: str
    answer: str | None
    citations: list[Citation]
    predicted_answer: str | None
    correct_answer: str
    judge_score: int | None
    judge_explanation: str | None
    error: str | None


class RetrievalTrace(_Record):
    """Candidate set produced before answer generation."""

    retrieved: list[RetrievedChunkRecord]
    expanded: list[RetrievedChunkRecord]
    graph_relations_used: list[GraphRelationUsed]
    reranked: list[RetrievedChunkRecord]
    rerank_scores: list[int]
    retrieval_mode: RetrievalMode
    metadata_filters: dict[str, Any]


class InteractiveStepTiming(_Record):
    """Wall-clock timings for a single interactive question."""

    retrieval_seconds: float = Field(default=0.0, ge=0.0)
    graph_expansion_seconds: float = Field(default=0.0, ge=0.0)
    rerank_seconds: float = Field(default=0.0, ge=0.0)
    context_seconds: float = Field(default=0.0, ge=0.0)
    answer_seconds: float = Field(default=0.0, ge=0.0)
    total_seconds: float = Field(default=0.0, ge=0.0)


class InteractiveRagResult(_Record):
    """Trace and answer for one interactive advanced RAG question."""

    question: str
    answer: str | None = None
    answer_rationale: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    invalid_citation_chunk_ids: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunkRecord] = Field(default_factory=list)
    expanded: list[RetrievedChunkRecord] = Field(default_factory=list)
    graph_relations_used: list[GraphRelationUsed] = Field(default_factory=list)
    reranked: list[RetrievedChunkRecord] = Field(default_factory=list)
    rerank_scores: list[int] = Field(default_factory=list)
    context_chunks: list[RetrievedChunkRecord] = Field(default_factory=list)
    context_text: str = ""
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_mode: RetrievalMode = "dense"
    hybrid_available: bool = False
    hybrid_unavailable_reason: str | None = None
    dense_vector_name: str | None = None
    sparse_vector_name: str | None = None
    collection_name: str
    flags: dict[str, bool] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    timing: InteractiveStepTiming = Field(default_factory=InteractiveStepTiming)
    error: str | None = None


def _unique_non_empty(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out

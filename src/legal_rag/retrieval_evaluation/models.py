"""Typed contracts for retrieval-only evaluation diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RETRIEVAL_EVALUATION_SCHEMA_VERSION = "retrieval-evaluation-v1"


class _Record(BaseModel):
    """Strict base model for JSON-friendly diagnostics."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize the record with JSON-compatible values."""
        return self.model_dump(mode="json", exclude_none=False)


class ReferenceTarget(_Record):
    """One expected legal reference resolved to article identity."""

    reference_text: str
    law_id: str
    article_id: str
    article_label_norm: str


class QuestionTarget(_Record):
    """One evaluation question with its expected retrieval targets."""

    qid: str
    level: str
    question: str
    correct_answer: str
    references: list[ReferenceTarget] = Field(default_factory=list)
    expected_law_ids: list[str] = Field(default_factory=list)
    expected_article_ids: list[str] = Field(default_factory=list)
    expected_law_chunk_count: int = Field(ge=0)
    expected_article_chunk_count: int = Field(ge=0)


class CandidateMetrics(_Record):
    """Hit and rank metrics for one ordered candidate list."""

    law_hit: bool
    article_hit: bool
    all_expected_articles_hit: bool
    first_law_rank: int | None
    first_article_rank: int | None
    article_mrr: float
    law_only_false_positive: bool


class RetrievalEvaluationRow(_Record):
    """One retrieval-only evaluation row for a question/configuration pair."""

    qid: str
    level: str
    question: str
    retrieval_mode: Literal["dense", "hybrid"]
    top_k: int = Field(gt=0)
    filter_name: str
    metadata_filters: dict[str, Any]
    graph_expansion_enabled: bool
    graph_expansion_seed_k: int | None
    max_chunks_per_expanded_law: int | None
    min_edge_confidence: float | None
    expected_law_ids: list[str]
    expected_article_ids: list[str]
    expected_law_chunk_count: int = Field(ge=0)
    expected_article_chunk_count: int = Field(ge=0)
    expected_article_filtered_chunk_count: int = Field(ge=0)
    filter_excluded: bool
    retrieved_count: int = Field(ge=0)
    expanded_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    direct_law_hit: bool
    direct_article_hit: bool
    direct_all_expected_articles_hit: bool
    direct_first_law_rank: int | None
    direct_first_article_rank: int | None
    direct_article_mrr: float
    law_only_false_positive: bool
    post_law_hit: bool
    post_article_hit: bool
    post_all_expected_articles_hit: bool
    post_first_law_rank: int | None
    post_first_article_rank: int | None
    post_article_mrr: float
    graph_incremental_hit: bool
    expanded_expected_article_hits: int = Field(ge=0)
    expansion_noise_ratio: float | None
    retrieved_chunk_ids: list[str]
    expanded_chunk_ids: list[str]


class RerankEvaluationRow(_Record):
    """One rerank-stage evaluation row for a question/configuration pair."""

    qid: str
    level: str
    question: str
    retrieval_mode: Literal["dense", "hybrid"]
    top_k: int = Field(gt=0)
    filter_name: str
    metadata_filters: dict[str, Any]
    base_scenario: str
    rerank_model: str
    rerank_input_k: int = Field(gt=0)
    rerank_output_k: int = Field(gt=0)
    expected_law_ids: list[str]
    expected_article_ids: list[str]
    candidate_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    reranked_law_hit: bool
    reranked_article_hit: bool
    reranked_all_expected_articles_hit: bool
    reranked_first_law_rank: int | None
    reranked_first_article_rank: int | None
    reranked_article_mrr: float
    pre_rerank_article_hit: bool
    pre_rerank_first_article_rank: int | None
    rerank_recovered_article: bool
    rerank_demoted_article: bool
    rerank_scores: list[int]
    cache_hit: bool
    reranked_chunk_ids: list[str]


class RetrievalScenarioSummary(_Record):
    """Aggregated metrics for one named scenario in the waterfall table."""

    scenario_name: str
    dataset: str
    stage: Literal["direct", "graph", "rerank"]
    article_hit_pct: float = Field(ge=0.0, le=100.0)
    law_hit_pct: float = Field(ge=0.0, le=100.0)
    article_mrr: float = Field(ge=0.0, le=1.0)
    n_questions: int = Field(ge=0)
    n_filter_excluded: int = Field(ge=0)
    config: dict[str, Any]
    delta_vs_baseline: float | None

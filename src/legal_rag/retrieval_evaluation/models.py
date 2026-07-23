"""Typed contracts for retrieval-only evaluation diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RETRIEVAL_EVALUATION_SCHEMA_VERSION = "retrieval-evaluation-v5"
FILTER_AUDIT_SCHEMA_VERSION = "filter-audit-v1"
FILTER_AUDIT_PROMPT_VERSION = "none-v1"


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


ARTICLE_HIT_K_VALUES: tuple[int, ...] = (5, 10, 20)


class CandidateMetrics(_Record):
    """Hit and rank metrics for one ordered candidate list."""

    law_hit: bool
    article_hit: bool
    all_expected_articles_hit: bool
    first_law_rank: int | None
    first_article_rank: int | None
    article_mrr: float
    law_only_false_positive: bool
    article_hit_at_k: dict[str, bool] = Field(default_factory=dict)


class RetrievalEvaluationRow(_Record):
    """One retrieval-only evaluation row for a question/configuration pair."""

    qid: str
    level: str
    question: str
    retrieval_mode: Literal["dense", "hybrid"]
    top_k: int = Field(gt=0)
    rrf_k: int | None = Field(default=None, gt=0)
    filter_name: str
    metadata_filters: dict[str, Any]
    exact: bool = False
    collection_identity: str = ""
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
    direct_article_hit_at_k: dict[str, bool] = Field(default_factory=dict)
    law_only_false_positive: bool
    post_law_hit: bool
    post_article_hit: bool
    post_all_expected_articles_hit: bool
    post_first_law_rank: int | None
    post_first_article_rank: int | None
    post_article_mrr: float
    post_article_hit_at_k: dict[str, bool] = Field(default_factory=dict)
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
    rrf_k: int | None = Field(default=None, gt=0)
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


class QueryRewriteEvaluationRow(_Record):
    """One query-rewriting evaluation row for a question/configuration pair."""

    qid: str
    level: str
    question: str
    retrieval_mode: Literal["dense", "hybrid"]
    top_k: int = Field(gt=0)
    rrf_k: int | None = Field(default=None, gt=0)
    filter_name: str
    metadata_filters: dict[str, Any]
    strategy: Literal["none", "rewrite", "hyde", "multi_query"]
    query_rewriting_model: str | None
    query_rewriting_prompt_version: str | None
    cache_hit: bool
    rewritten_queries: list[str]
    transformed_query_count: int = Field(ge=0)
    expected_law_ids: list[str]
    expected_article_ids: list[str]
    expected_law_chunk_count: int = Field(ge=0)
    expected_article_chunk_count: int = Field(ge=0)
    retrieved_count: int = Field(ge=0)
    direct_law_hit: bool
    direct_article_hit: bool
    direct_all_expected_articles_hit: bool
    direct_first_law_rank: int | None
    direct_first_article_rank: int | None
    direct_article_mrr: float
    law_only_false_positive: bool
    retrieved_chunk_ids: list[str]


class RetrievalScenarioSummary(_Record):
    """Aggregated metrics for one named scenario in the waterfall table."""

    scenario_name: str
    dataset: str
    stage: Literal["direct", "graph", "rerank", "query_rewriting"]
    article_hit_pct: float = Field(ge=0.0, le=100.0)
    law_hit_pct: float = Field(ge=0.0, le=100.0)
    article_mrr: float = Field(ge=0.0, le=1.0)
    n_questions: int = Field(ge=0)
    n_filter_excluded: int = Field(ge=0)
    config: dict[str, Any]
    delta_vs_baseline: float | None


class FilterReferenceAuditRow(_Record):
    """Static coverage of one expected article under one metadata filter."""

    schema_version: Literal["filter-audit-v1"] = FILTER_AUDIT_SCHEMA_VERSION
    qid: str
    datasets: list[str]
    level: str
    reference_text: str
    law_id: str
    article_id: str
    filter_name: str
    metadata_filters: dict[str, Any]
    exact: bool = False
    collection_identity: str
    total_target_chunks: int = Field(ge=0)
    retained_target_chunks: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    coverage_status: Literal["fully_eligible", "partially_eligible", "fully_excluded"]
    active_target_chunks: int = Field(ge=0)
    retained_active_target_chunks: int = Field(ge=0)
    active_slice: bool
    unknown_status_present: bool
    unknown_status_excluded: bool
    law_statuses: list[str]
    article_statuses: list[str]
    passage_statuses: list[str]
    content_availability: list[str]
    status_event_ids: list[str]
    status_rule_ids: list[str]
    expected_reference_validity: str | None = None
    answer_support_relation: str | None = None
    answer_supporting_law_id: str | None = None
    answer_supporting_article_id: str | None = None
    answer_supporting_passage_id: str | None = None
    supporting_passage_retained: bool | None = None
    temporal_scope_flag: str | None = None
    review_rationale: str | None = None


class FilterImpactRow(_Record):
    """Paired retrieval impact and independent filter decision axes."""

    schema_version: Literal["filter-audit-v1"] = FILTER_AUDIT_SCHEMA_VERSION
    dataset: str
    retrieval_mode: Literal["dense", "hybrid"]
    top_k: int = Field(gt=0)
    rrf_k: int | None = Field(default=None, gt=0)
    filter_name: str
    metadata_filters: dict[str, Any]
    exact: bool = False
    n_questions: int = Field(ge=0)
    article_success_pct: float = Field(ge=0.0, le=100.0)
    baseline_article_success_pct: float = Field(ge=0.0, le=100.0)
    article_success_delta_pp: float
    article_success_ci_low_pp: float
    article_success_ci_high_pp: float
    article_success_ci_level: float = Field(gt=0.0, lt=1.0)
    law_success_pct: float = Field(ge=0.0, le=100.0)
    baseline_law_success_pct: float = Field(ge=0.0, le=100.0)
    law_success_delta_pp: float
    article_mrr: float = Field(ge=0.0, le=1.0)
    baseline_article_mrr: float = Field(ge=0.0, le=1.0)
    article_mrr_delta: float
    article_mrr_ci_low: float
    article_mrr_ci_high: float
    gains: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    total_reference_targets: int = Field(ge=0)
    fully_eligible_targets: int = Field(ge=0)
    partially_eligible_targets: int = Field(ge=0)
    fully_excluded_targets: int = Field(ge=0)
    benchmark_full_coverage: bool
    active_slice_safety: Literal["safe", "unsafe", "unresolved"]
    retrieval_effect: Literal["beneficial", "harmful", "inconclusive"]
    bootstrap_supported: bool


class FilterExactControlRow(_Record):
    """Aggregate ANN-versus-exact dense-search control."""

    schema_version: Literal["filter-audit-v1"] = FILTER_AUDIT_SCHEMA_VERSION
    dataset: str
    top_k: int = Field(gt=0)
    filter_name: str
    metadata_filters: dict[str, Any]
    n_questions: int = Field(ge=0)
    mean_chunk_overlap: float = Field(ge=0.0, le=1.0)
    ann_article_success_pct: float = Field(ge=0.0, le=100.0)
    exact_article_success_pct: float = Field(ge=0.0, le=100.0)
    article_success_delta_pp: float
    ann_law_success_pct: float = Field(ge=0.0, le=100.0)
    exact_law_success_pct: float = Field(ge=0.0, le=100.0)
    law_success_delta_pp: float
    ann_article_mrr: float = Field(ge=0.0, le=1.0)
    exact_article_mrr: float = Field(ge=0.0, le=1.0)
    article_mrr_delta: float


class StatusTransitionRow(_Record):
    """One comparable law or article status transition from v1 to v2."""

    schema_version: Literal["filter-audit-v1"] = FILTER_AUDIT_SCHEMA_VERSION
    entity_type: Literal["law", "article"]
    entity_id: str
    law_id: str
    status_field: Literal["law_status", "article_status"]
    old_present: bool
    new_present: bool
    old_status: str | None = None
    new_status: str | None = None
    transition: str

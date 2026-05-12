"""Utilities for fast retrieval-only RAG diagnostics."""

from __future__ import annotations

from .evaluator import (
    CachedEmbedder,
    ChunkAvailabilityIndex,
    RerankCache,
    answer_overlap,
    candidate_metrics,
    dedupe_chunks,
    evaluate_candidate_set,
    evaluate_with_rerank,
    payload_matches_filters,
    resolve_question_targets,
    retrieve_direct,
    summarize_scenario,
    write_run_artifacts,
)
from .models import (
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    CandidateMetrics,
    QuestionTarget,
    ReferenceTarget,
    RerankEvaluationRow,
    RetrievalEvaluationRow,
    RetrievalScenarioSummary,
)

__all__ = [
    "RETRIEVAL_EVALUATION_SCHEMA_VERSION",
    "CachedEmbedder",
    "CandidateMetrics",
    "ChunkAvailabilityIndex",
    "QuestionTarget",
    "ReferenceTarget",
    "RerankCache",
    "RerankEvaluationRow",
    "RetrievalEvaluationRow",
    "RetrievalScenarioSummary",
    "answer_overlap",
    "candidate_metrics",
    "dedupe_chunks",
    "evaluate_candidate_set",
    "evaluate_with_rerank",
    "payload_matches_filters",
    "resolve_question_targets",
    "retrieve_direct",
    "summarize_scenario",
    "write_run_artifacts",
]

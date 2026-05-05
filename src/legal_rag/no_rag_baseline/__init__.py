"""Public API for 04 no-RAG baseline."""

from __future__ import annotations

from legal_rag.oracle_context_evaluation.llm import UtopiaStructuredChatClient, resolve_ollama_chat_url
from legal_rag.oracle_context_evaluation.models import JudgeOutput, McqAnswerOutput, NoHintAnswerOutput
from legal_rag.oracle_context_evaluation.scoring import aggregate_results, score_mcq_label

from .models import (
    NO_RAG_PROMPT_VERSION,
    NO_RAG_SCHEMA_VERSION,
    McqResultRow,
    NoHintResultRow,
    NoRagConfig,
    NoRagConnectionRecord,
    NoRagLevelMetrics,
    NoRagManifest,
    NoRagMetricSummary,
    NoRagModelIdentities,
    NoRagOutputFiles,
    NoRagRunCounts,
    NoRagSummary,
    SafeNoRagConfigRecord,
)
from .runner import (
    build_summary,
    resolve_answer_model,
    resolve_judge_model,
    resolve_utopia_runtime,
    run_mcq,
    run_no_hint,
    run_no_rag_baseline,
)

__all__ = [
    "NO_RAG_PROMPT_VERSION",
    "NO_RAG_SCHEMA_VERSION",
    "JudgeOutput",
    "McqAnswerOutput",
    "McqResultRow",
    "NoHintAnswerOutput",
    "NoHintResultRow",
    "NoRagConfig",
    "NoRagConnectionRecord",
    "NoRagLevelMetrics",
    "NoRagManifest",
    "NoRagMetricSummary",
    "NoRagModelIdentities",
    "NoRagOutputFiles",
    "NoRagRunCounts",
    "NoRagSummary",
    "SafeNoRagConfigRecord",
    "UtopiaStructuredChatClient",
    "aggregate_results",
    "build_summary",
    "resolve_ollama_chat_url",
    "resolve_answer_model",
    "resolve_judge_model",
    "resolve_utopia_runtime",
    "run_mcq",
    "run_no_hint",
    "run_no_rag_baseline",
    "score_mcq_label",
]

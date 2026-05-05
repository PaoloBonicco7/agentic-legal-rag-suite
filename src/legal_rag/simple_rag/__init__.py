"""Public API for 05 simple RAG baseline."""

from __future__ import annotations

from legal_rag.oracle_context_evaluation.llm import UtopiaStructuredChatClient, resolve_ollama_chat_url
from legal_rag.oracle_context_evaluation.scoring import aggregate_results, score_mcq_label

from .models import (
    SIMPLE_RAG_PROMPT_VERSION,
    SIMPLE_RAG_SCHEMA_VERSION,
    Citation,
    McqResultRow,
    NoHintResultRow,
    RetrievedChunkRecord,
    SimpleMcqAnswerOutput,
    SimpleNoHintAnswerOutput,
    SimpleRagConfig,
)
from .retrieval import build_static_filter, load_index_manifest, resolve_collection_name, resolve_index_manifest_path, search_dense
from .runner import (
    build_context,
    build_query_embedder,
    build_summary,
    resolve_answer_model,
    resolve_judge_model,
    resolve_utopia_runtime,
    run_mcq,
    run_no_hint,
    run_simple_rag,
)

__all__ = [
    "SIMPLE_RAG_PROMPT_VERSION",
    "SIMPLE_RAG_SCHEMA_VERSION",
    "Citation",
    "McqResultRow",
    "NoHintResultRow",
    "RetrievedChunkRecord",
    "SimpleMcqAnswerOutput",
    "SimpleNoHintAnswerOutput",
    "SimpleRagConfig",
    "UtopiaStructuredChatClient",
    "aggregate_results",
    "build_context",
    "build_query_embedder",
    "build_static_filter",
    "build_summary",
    "load_index_manifest",
    "resolve_collection_name",
    "resolve_index_manifest_path",
    "resolve_ollama_chat_url",
    "resolve_answer_model",
    "resolve_judge_model",
    "resolve_utopia_runtime",
    "run_mcq",
    "run_no_hint",
    "run_simple_rag",
    "score_mcq_label",
    "search_dense",
]

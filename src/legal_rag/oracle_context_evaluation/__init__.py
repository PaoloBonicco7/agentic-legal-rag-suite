"""Public API for 02b oracle-context evaluation."""

from __future__ import annotations

from .llm import (
    UtopiaOpenAIChatClient,
    UtopiaStructuredChatClient,
    discover_utopia_api_models,
    discover_utopia_models,
    resolve_ollama_chat_url,
    resolve_openai_chat_completions_url,
)
from .models import (
    ORACLE_CONTEXT_SCHEMA_VERSION,
    ORACLE_PROMPT_VERSION,
    JudgeOutput,
    McqAnswerOutput,
    NoHintAnswerOutput,
    OracleEvaluationConfig,
)
from .references import OracleReferenceResolver, build_context_text, parse_reference, split_reference_values
from .runner import (
    build_oracle_contexts,
    create_default_client,
    resolve_answer_model,
    resolve_judge_model,
    resolve_utopia_runtime,
    run_oracle_context_evaluation,
)
from .scoring import aggregate_results, score_mcq_label

__all__ = [
    "ORACLE_CONTEXT_SCHEMA_VERSION",
    "ORACLE_PROMPT_VERSION",
    "JudgeOutput",
    "McqAnswerOutput",
    "NoHintAnswerOutput",
    "OracleEvaluationConfig",
    "OracleReferenceResolver",
    "UtopiaOpenAIChatClient",
    "UtopiaStructuredChatClient",
    "aggregate_results",
    "build_context_text",
    "build_oracle_contexts",
    "create_default_client",
    "discover_utopia_api_models",
    "discover_utopia_models",
    "parse_reference",
    "resolve_ollama_chat_url",
    "resolve_openai_chat_completions_url",
    "resolve_answer_model",
    "resolve_judge_model",
    "resolve_utopia_runtime",
    "run_oracle_context_evaluation",
    "score_mcq_label",
    "split_reference_values",
]

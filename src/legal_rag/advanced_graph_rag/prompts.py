"""Prompt builders for advanced graph-aware RAG runs."""

from __future__ import annotations

import json
from typing import Any

from legal_rag.simple_rag.models import RetrievedChunkRecord
from legal_rag.simple_rag.prompts import format_context_chunks, format_options

from .models import AdvancedMcqAnswerOutput, AdvancedNoHintAnswerOutput, RerankOutput


def schema_dict(model_cls: type[AdvancedMcqAnswerOutput] | type[AdvancedNoHintAnswerOutput] | type[RerankOutput]) -> dict[str, Any]:
    """Return a JSON schema payload accepted by Ollama-compatible structured chat."""
    return model_cls.model_json_schema()


def build_rerank_prompt(question: str, candidates: list[RetrievedChunkRecord]) -> str:
    """Build a chunk relevance reranking prompt."""
    blocks = []
    for idx, chunk in enumerate(candidates, start=1):
        payload = chunk.payload
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] chunk_id: {chunk.chunk_id}",
                    f"law_id: {payload.get('law_id', '')}",
                    f"article_id: {payload.get('article_id', '')}",
                    "text:",
                    chunk.text,
                ]
            )
        )
    return (
        "You are reranking Italian legal context chunks for a legal QA task.\n"
        "Score every candidate against the question with this rubric:\n"
        "- 2: directly answers or strongly supports the answer.\n"
        "- 1: partially related or useful background.\n"
        "- 0: irrelevant or misleading.\n"
        "Return one score for each input chunk_id. Use only integer scores 0, 1, or 2.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(RerankOutput), ensure_ascii=False)}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Candidate chunks:\n"
        f"{'\n\n'.join(blocks)}"
    )


def build_mcq_prompt(record: dict[str, Any], context_text: str) -> str:
    """Build an MCQ answer prompt with advanced retrieved context."""
    return (
        "You answer Italian legal multiple-choice questions using only the provided legal context.\n"
        "Choose exactly one label among A, B, C, D, E, F from the provided options.\n"
        "Return citation_chunk_ids containing only chunk_id values present in the context.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(AdvancedMcqAnswerOutput), ensure_ascii=False)}\n\n"
        "Legal context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{record['question_stem']}\n\n"
        "Options:\n"
        f"{format_options(record['options'])}"
    )


def build_no_hint_prompt(record: dict[str, Any], context_text: str) -> str:
    """Build an open-answer prompt with advanced retrieved context."""
    return (
        "You answer Italian legal questions precisely and concisely using only the provided legal context.\n"
        "Do not mention multiple-choice labels or options.\n"
        "Return citation_chunk_ids containing only chunk_id values present in the context.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(AdvancedNoHintAnswerOutput), ensure_ascii=False)}\n\n"
        "Legal context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{record['question']}"
    )


__all__ = [
    "build_mcq_prompt",
    "build_no_hint_prompt",
    "build_rerank_prompt",
    "format_context_chunks",
    "schema_dict",
]

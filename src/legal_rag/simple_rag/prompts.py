"""Prompt builders for simple RAG baseline runs."""

from __future__ import annotations

import json
from typing import Any

from .models import RetrievedChunkRecord, SimpleMcqAnswerOutput, SimpleNoHintAnswerOutput


def schema_dict(model_cls: type[SimpleMcqAnswerOutput] | type[SimpleNoHintAnswerOutput]) -> dict[str, Any]:
    """Return a JSON schema payload accepted by Ollama-compatible structured chat."""
    return model_cls.model_json_schema()


def format_options(options: dict[str, str]) -> str:
    """Render MCQ options in stable A-F order."""
    return "\n".join(f"{label}) {options[label]}" for label in ("A", "B", "C", "D", "E", "F"))


def format_context_chunks(chunks: list[RetrievedChunkRecord], *, text_overrides: dict[str, str] | None = None) -> str:
    """Render retrieved chunks as citation-ready context blocks."""
    overrides = text_overrides or {}
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        text = overrides.get(chunk.chunk_id, chunk.text)
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] chunk_id: {chunk.chunk_id}",
                    f"law_id: {payload.get('law_id', '')}",
                    f"article_id: {payload.get('article_id', '')}",
                    f"law_title: {payload.get('law_title', '')}",
                    "text:",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_mcq_prompt(record: dict[str, Any], context_text: str) -> str:
    """Build an MCQ answer prompt with retrieved legal context."""
    return (
        "You answer Italian legal multiple-choice questions using only the provided legal context.\n"
        "Choose exactly one label among A, B, C, D, E, F from the provided options.\n"
        "Return citation_chunk_ids containing only chunk_id values present in the context.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(SimpleMcqAnswerOutput), ensure_ascii=False)}\n\n"
        "Legal context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{record['question_stem']}\n\n"
        "Options:\n"
        f"{format_options(record['options'])}"
    )


def build_no_hint_prompt(record: dict[str, Any], context_text: str) -> str:
    """Build an open-answer prompt with retrieved legal context."""
    return (
        "You answer Italian legal questions precisely and concisely using only the provided legal context.\n"
        "Do not mention multiple-choice labels or options.\n"
        "Return citation_chunk_ids containing only chunk_id values present in the context.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(SimpleNoHintAnswerOutput), ensure_ascii=False)}\n\n"
        "Legal context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{record['question']}"
    )

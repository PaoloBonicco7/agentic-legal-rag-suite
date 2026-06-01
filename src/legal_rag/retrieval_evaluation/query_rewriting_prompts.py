"""Prompt builders for query rewriting diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

QUERY_REWRITING_PROMPT_VERSION = "query-rewriting-v1"


def build_rewrite_prompt(question: str, *, payload_schema: Mapping[str, Any]) -> str:
    """Build a legal query rewrite prompt."""
    return (
        "You rewrite Italian legal QA questions into concise retrieval queries for a statutory corpus.\n"
        "Preserve the user's legal intent, named entities, deadlines, competent bodies, sanctions, conditions, and exceptions.\n"
        "Use legal terminology when it is implied by the question, but do not add facts or cite laws that are not present.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(dict(payload_schema), ensure_ascii=False)}\n\n"
        "Question:\n"
        f"{question}"
    )


def build_hyde_prompt(question: str, *, payload_schema: Mapping[str, Any]) -> str:
    """Build a HyDE prompt for legal retrieval."""
    return (
        "You generate a short hypothetical statutory passage in Italian to improve retrieval.\n"
        "Write what a relevant legal provision might say if it answered the question directly.\n"
        "Do not invent article numbers, law numbers, dates, or citations. Keep it factual, compact, and retrieval-oriented.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(dict(payload_schema), ensure_ascii=False)}\n\n"
        "Question:\n"
        f"{question}"
    )


def build_multi_query_prompt(question: str, *, n: int, payload_schema: Mapping[str, Any]) -> str:
    """Build a multi-query prompt for legal retrieval."""
    return (
        "You generate alternative Italian legal retrieval queries for the same question.\n"
        f"Return exactly {n} distinct queries.\n"
        "Each query must be concise, non-empty, and focused on a different legal wording of the same information need.\n"
        "Do not add facts, law numbers, article numbers, or dates that are not present in the question.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(dict(payload_schema), ensure_ascii=False)}\n\n"
        "Question:\n"
        f"{question}"
    )


__all__ = [
    "QUERY_REWRITING_PROMPT_VERSION",
    "build_hyde_prompt",
    "build_multi_query_prompt",
    "build_rewrite_prompt",
]

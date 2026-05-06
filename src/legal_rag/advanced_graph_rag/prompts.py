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
        "Score every candidate against the question with this legal-domain rubric:\n"
        "- 2: the chunk states the rule, definition, list, condition, sanction, deadline, competent body, or exception needed to answer.\n"
        "- 1: the chunk is related background, cites the relevant norm, or points to another act, but does not itself state the answer.\n"
        "- 0: the chunk is irrelevant, obsolete for the question, about a different legal institute, or likely to mislead the answer.\n"
        "Prefer specific operative provisions over titles, recitals, financial clauses, or generic cross-references.\n"
        "If a chunk explicitly lists the requested items, score it 2. If it only mentions the article that may contain them, score it 1.\n"
        "Examples:\n"
        "- Question: Quali sono gli organi dell'azienda USL? Chunk lists 'direttore generale' and 'collegio sindacale' -> score 2.\n"
        "- Question: A quali condizioni e' ammessa la pubblicita' per le sale da gioco? Chunk says advertising is prohibited -> score 2.\n"
        "- Question: Chi adotta il piano? Chunk only says the plan is governed by a later deliberation -> score 1 unless it names the adopting body.\n"
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
        "You answer Italian legal multiple-choice questions using the provided legal context as the primary source.\n"
        "Choose exactly one label among A, B, C, D, E, F from the provided options.\n"
        "Use context chunks that state the decisive rule over chunks that only mention related topics.\n"
        "If the context is not decisive, choose the option most consistent with the legal question and do not overfit unrelated context wording.\n"
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
        "You answer Italian legal questions precisely and concisely using the provided legal context as the primary source.\n"
        "Instructions:\n"
        "1. Read all chunks and identify the chunks that explicitly contain the answer.\n"
        "2. Extract the answer from those chunks. If the question asks 'quali sono' or 'quali', list the items; do not merely cite the article.\n"
        "3. If the context states a prohibition, exception, condition, sanction, deadline, competent body, or legal consequence, state that implication directly.\n"
        "4. Do not write that the context lacks information when a chunk answers the question explicitly or by clear implication.\n"
        "5. Set context_sufficient='yes' when the context answers the question, 'partial' when it only supports an incomplete answer, and 'no' when it is not relevant.\n"
        "6. If context_sufficient='partial', give the partial answer and say what is missing. If it is 'no', say that the context does not answer.\n"
        "Do not mention multiple-choice labels or options.\n"
        "Return citation_chunk_ids containing only chunk_id values that you actually used and that are present in the context.\n"
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

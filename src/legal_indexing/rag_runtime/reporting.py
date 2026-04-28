from __future__ import annotations

from typing import Any

from .qdrant_retrieval import RetrievedChunk


def retrieval_preview(
    retrieved: list[RetrievedChunk], *, limit: int = 10, excerpt_chars: int = 220
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, doc in enumerate(retrieved[: max(1, int(limit))], start=1):
        excerpt = (doc.text or "").strip().replace("\n", " ")
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[:excerpt_chars].rstrip() + "..."
        rows.append(
            {
                "rank": rank,
                "chunk_id": doc.chunk_id,
                "law_id": doc.law_id,
                "article_id": doc.article_id,
                "law_status": doc.law_status,
                "law_date": doc.law_date,
                "score": round(float(doc.score), 6),
                "source_passage_ids": list(doc.source_passage_ids),
                "source_chunk_ids": list(doc.source_chunk_ids),
                "excerpt": excerpt,
            }
        )
    return rows


def summarize_filters(state: dict[str, Any]) -> dict[str, Any]:
    filters = state.get("filters_used", {}) or {}
    metadata_decision = state.get("metadata_filter_decision") or {}
    return {
        "view_filter": filters.get("view", "none"),
        "law_status_filter": filters.get("law_status"),
        "law_ids_filter": list(filters.get("law_ids") or []),
        "metadata_seed_law_ids": list(filters.get("metadata_seed_law_ids") or []),
        "relation_types_filter": list(filters.get("relation_types") or []),
        "metadata_seed_article_ids": list(filters.get("metadata_seed_article_ids") or []),
        "year_from": filters.get("year_from"),
        "year_to": filters.get("year_to"),
        "metadata_mode": filters.get("metadata_mode"),
        "relation_query": bool(filters.get("relation_query")),
        "metadata_hard_law_filter_applied": bool(filters.get("metadata_hard_law_filter_applied")),
        "metadata_hard_article_filter_applied": bool(
            filters.get("metadata_hard_article_filter_applied")
        ),
        "metadata_heuristics": list(metadata_decision.get("applied_heuristics") or []),
        "raw_filter_present": bool(state.get("query_filter") is not None),
    }


def summarize_context(state: dict[str, Any]) -> dict[str, Any]:
    summary = state.get("context_summary")
    if isinstance(summary, dict):
        return summary
    context = str(state.get("context") or "")
    return {
        "included_count": len(list(state.get("retrieved") or [])),
        "total_chars": len(context),
    }


def summarize_answer(state: dict[str, Any]) -> dict[str, Any]:
    answer = state.get("answer")
    if not isinstance(answer, dict):
        return {
            "answer": "",
            "citations": [],
            "needs_more_context": True,
            "answer_source": "fallback",
            "was_empty_before_guard": True,
        }
    return {
        "answer": str(answer.get("answer") or ""),
        "citations": list(answer.get("citations") or []),
        "needs_more_context": bool(answer.get("needs_more_context")),
        "answer_source": str(answer.get("answer_source") or "model"),
        "was_empty_before_guard": bool(answer.get("was_empty_before_guard")),
    }


def provenance_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = state.get("provenance")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "chunk_id": row.get("chunk_id"),
                "law_id": row.get("law_id"),
                "article_id": row.get("article_id"),
                "source_chunk_ids": list(row.get("source_chunk_ids") or []),
                "source_passage_ids": list(row.get("source_passage_ids") or []),
                "score": row.get("score"),
                "cited": bool(row.get("cited")),
            }
        )
    return out

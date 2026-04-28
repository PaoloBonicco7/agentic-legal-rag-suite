from __future__ import annotations

from typing import Any

from .chunk_refinement import RefinedChunk


def refined_chunk_payload(
    chunk: RefinedChunk,
    *,
    dataset_hash: str,
    chunking_profile_id: str,
    embedding_model: str,
    content_hash: str,
    payload_hash: str | None = None,
) -> dict[str, Any]:
    law_year: int | None = None
    if isinstance(chunk.law_date, str) and len(chunk.law_date) >= 4 and chunk.law_date[:4].isdigit():
        law_year = int(chunk.law_date[:4])

    payload: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "law_id": chunk.law_id,
        "article_id": chunk.article_id,
        "source_passage_ids": list(chunk.source_passage_ids),
        "source_chunk_ids": list(chunk.source_chunk_ids),
        "source_passage_labels": list(chunk.source_passage_labels),
        "article_order_in_law": chunk.article_order_in_law,
        "passage_start_order": chunk.passage_start_order,
        "passage_end_order": chunk.passage_end_order,
        "article_chunk_order": chunk.article_chunk_order,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "index_views": list(chunk.index_views),
        "law_status": chunk.law_status,
        "article_is_abrogated": bool(chunk.article_is_abrogated),
        "related_law_ids": list(chunk.related_law_ids),
        "relation_types": list(chunk.relation_types),
        "inbound_law_ids": list(chunk.inbound_law_ids),
        "outbound_law_ids": list(chunk.outbound_law_ids),
        "law_date": chunk.law_date,
        "law_year": law_year,
        "law_number": chunk.law_number,
        "law_title": chunk.law_title,
        "status_confidence": chunk.status_confidence,
        "status_evidence": [dict(x) for x in chunk.status_evidence],
        "dataset_hash": dataset_hash,
        "chunking_profile_id": chunking_profile_id,
        "embedding_model": embedding_model,
        "content_hash": content_hash,
        "text": chunk.text,
        "text_for_embedding": chunk.text_for_embedding,
    }
    if payload_hash is not None:
        payload["payload_hash"] = payload_hash
    return payload


__all__ = ["refined_chunk_payload"]

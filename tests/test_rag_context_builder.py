from __future__ import annotations

from legal_indexing.rag_runtime.context_builder import build_context
from legal_indexing.rag_runtime.qdrant_retrieval import RetrievedChunk


def _chunk(*, chunk_id: str, law_id: str, article_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.5,
        text=f"Testo per {chunk_id}",
        point_id=chunk_id,
        payload={
            "chunk_id": chunk_id,
            "law_id": law_id,
            "article_id": article_id,
            "source_chunk_ids": [chunk_id],
            "source_passage_ids": [f"{article_id}#p:intro"],
        },
    )


def test_context_builder_backfills_after_caps_until_max_chunks() -> None:
    # First six candidates are mostly from one law: caps skip several of them.
    # Builder must continue scanning the remaining candidates to backfill to max_chunks.
    retrieved = [
        _chunk(chunk_id=f"law:a#art:{i}#chunk:0", law_id="law:a", article_id=f"law:a#art:{i}")
        for i in range(1, 7)
    ] + [
        _chunk(chunk_id=f"law:b#art:{i}#chunk:0", law_id="law:b", article_id=f"law:b#art:{i}")
        for i in range(1, 4)
    ] + [
        _chunk(chunk_id=f"law:c#art:{i}#chunk:0", law_id="law:c", article_id=f"law:c#art:{i}")
        for i in range(1, 4)
    ]

    out = build_context(
        retrieved,
        max_chunks=6,
        max_chars=20_000,
        per_chunk_max_chars=500,
    )
    assert out.to_dict()["included_count"] == 6
    assert len(out.included_chunk_ids) == 6

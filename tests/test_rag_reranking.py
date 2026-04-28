from __future__ import annotations

from legal_indexing.rag_runtime.config import AdvancedRerankConfig
from legal_indexing.rag_runtime.metadata_filters import MetadataFilterDecision
from legal_indexing.rag_runtime.qdrant_retrieval import RetrievedChunk
from legal_indexing.rag_runtime.reranking import rerank_candidates


def _chunk(
    *,
    chunk_id: str,
    score: float,
    text: str,
    law_status: str,
    relation_types: list[str],
    law_date: str = "2020-01-01",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        text=text,
        point_id=chunk_id,
        payload={
            "chunk_id": chunk_id,
            "law_status": law_status,
            "relation_types": relation_types,
            "law_date": law_date,
            "source_chunk_ids": [],
            "source_passage_ids": [],
        },
    )


def test_reranking_adds_graph_and_metadata_bonus_deterministically() -> None:
    chunks = [
        _chunk(
            chunk_id="c1",
            score=0.9,
            text="modifica della norma principale",
            law_status="current",
            relation_types=["AMENDS"],
        ),
        _chunk(
            chunk_id="c2",
            score=0.9,
            text="disposizione generale",
            law_status="current",
            relation_types=["REFERENCES"],
        ),
    ]
    decision = MetadataFilterDecision(
        view="current",
        law_status="current",
        law_ids=tuple(),
        relation_types=("AMENDS",),
        article_ids=tuple(),
        year_from=None,
        year_to=None,
        applied_heuristics=tuple(),
    )
    out = rerank_candidates(
        "chi modifica la norma",
        chunks,
        config=AdvancedRerankConfig(),
        source_tags_by_chunk={"c1": {"graph"}, "c2": {"primary"}},
        metadata_decision=decision,
    )
    ordered = [x.chunk_id for x in out.ordered_chunks]
    assert ordered[0] == "c1"
    assert len(out.rows) == 2
    assert out.rows[0].final_score >= out.rows[1].final_score


def test_reranking_tie_breaks_by_chunk_id() -> None:
    cfg = AdvancedRerankConfig(
        weight_retrieval_score=1.0,
        weight_graph_bonus=0.0,
        weight_metadata_bonus=0.0,
        weight_lexical_overlap=0.0,
        tie_breaker="chunk_id",
    )
    chunks = [
        _chunk(
            chunk_id="z_chunk",
            score=0.5,
            text="testo",
            law_status="current",
            relation_types=[],
        ),
        _chunk(
            chunk_id="a_chunk",
            score=0.5,
            text="testo",
            law_status="current",
            relation_types=[],
        ),
    ]
    out = rerank_candidates("domanda", chunks, config=cfg)
    assert [x.chunk_id for x in out.ordered_chunks] == ["a_chunk", "z_chunk"]

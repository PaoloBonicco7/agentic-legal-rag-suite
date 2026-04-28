from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .config import AdvancedRerankConfig
from .metadata_filters import MetadataFilterDecision, is_relation_query
from .qdrant_retrieval import RetrievedChunk


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class RerankRow:
    chunk_id: str
    retrieval_score: float
    lexical_overlap: float
    sparse_score: float
    graph_bonus: float
    metadata_bonus: float
    final_score: float
    source_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "retrieval_score": self.retrieval_score,
            "lexical_overlap": self.lexical_overlap,
            "sparse_score": self.sparse_score,
            "graph_bonus": self.graph_bonus,
            "metadata_bonus": self.metadata_bonus,
            "final_score": self.final_score,
            "source_tags": list(self.source_tags),
        }


@dataclass(frozen=True)
class RerankResult:
    ordered_chunks: tuple[RetrievedChunk, ...]
    rows: tuple[RerankRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "ordered_chunk_ids": [chunk.chunk_id for chunk in self.ordered_chunks],
        }


def _tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for match in _TOKEN_RE.finditer(str(text or "").lower()):
        token = match.group(0).strip()
        if token:
            out.add(token)
    return out


def _metadata_match_score(
    chunk: RetrievedChunk,
    *,
    decision: MetadataFilterDecision | None,
) -> float:
    if decision is None:
        return 0.0

    checks = 0
    score = 0.0

    if decision.law_status:
        checks += 1
        if (chunk.law_status or "").strip().lower() == decision.law_status.strip().lower():
            score += 1.0

    if decision.law_ids:
        checks += 1
        law_id = (chunk.law_id or "").strip()
        if law_id and law_id in set(decision.law_ids):
            score += 1.0

    if decision.relation_types:
        checks += 1
        rels = {str(x).strip().upper() for x in (chunk.payload.get("relation_types") or [])}
        wanted = {str(x).strip().upper() for x in decision.relation_types}
        if rels & wanted:
            score += 1.0

    if decision.article_ids:
        checks += 1
        article_id = (chunk.article_id or "").strip()
        if article_id and article_id in set(decision.article_ids):
            score += 1.0

    if decision.year_from is not None or decision.year_to is not None:
        checks += 1
        law_date = str(chunk.payload.get("law_date") or "").strip()
        year = None
        if len(law_date) >= 4 and law_date[:4].isdigit():
            year = int(law_date[:4])
        if year is not None:
            lower_ok = decision.year_from is None or year >= decision.year_from
            upper_ok = decision.year_to is None or year <= decision.year_to
            if lower_ok and upper_ok:
                score += 1.0

    if checks == 0:
        return 0.0
    return score / checks


def rerank_candidates(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    config: AdvancedRerankConfig,
    source_tags_by_chunk: dict[str, set[str]] | None = None,
    metadata_decision: MetadataFilterDecision | None = None,
    relation_query: bool = False,
) -> RerankResult:
    if not chunks:
        return RerankResult(ordered_chunks=tuple(), rows=tuple())

    source_tags_by_chunk = source_tags_by_chunk or {}
    query_tokens = _tokenize(question)
    query_low = str(question or "").lower()
    relation_query = bool(relation_query or is_relation_query(question, decision=metadata_decision))
    query_is_specific = any(
        k in query_low for k in ("art.", "articolo", "comma", "l.r.", "legge regionale", "d.lgs.", "law:", "#art:")
    )

    scored: list[tuple[RetrievedChunk, RerankRow]] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.text)
        if query_tokens:
            lexical_overlap = len(query_tokens & chunk_tokens) / float(len(query_tokens))
        else:
            lexical_overlap = 0.0

        source_tags = tuple(sorted(source_tags_by_chunk.get(chunk.chunk_id, set())))
        graph_bonus = 1.0 if "graph" in set(source_tags) else 0.0
        if query_is_specific and not relation_query:
            graph_bonus *= 0.35

        sparse_score_raw = chunk.payload.get("hybrid_sparse_score")
        try:
            sparse_score = float(sparse_score_raw) if sparse_score_raw is not None else 0.0
        except Exception:
            sparse_score = 0.0
        sparse_score = max(0.0, min(1.0, sparse_score))

        metadata_bonus = _metadata_match_score(chunk, decision=metadata_decision)

        final_score = (
            config.weight_retrieval_score * float(chunk.score)
            + config.weight_graph_bonus * graph_bonus
            + config.weight_metadata_bonus * metadata_bonus
            + config.weight_lexical_overlap * lexical_overlap
            + config.weight_sparse_score * sparse_score
        )
        row = RerankRow(
            chunk_id=chunk.chunk_id,
            retrieval_score=float(chunk.score),
            lexical_overlap=float(lexical_overlap),
            sparse_score=float(sparse_score),
            graph_bonus=float(graph_bonus),
            metadata_bonus=float(metadata_bonus),
            final_score=float(final_score),
            source_tags=source_tags,
        )
        scored.append((chunk, row))

    if config.tie_breaker == "retrieval_score":
        scored.sort(
            key=lambda item: (
                -item[1].final_score,
                -item[1].retrieval_score,
                item[1].chunk_id,
            )
        )
    else:
        scored.sort(key=lambda item: (-item[1].final_score, item[1].chunk_id))

    return RerankResult(
        ordered_chunks=tuple([item[0] for item in scored]),
        rows=tuple([item[1] for item in scored]),
    )

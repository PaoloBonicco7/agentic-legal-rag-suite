from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .qdrant_retrieval import RetrievedChunk


@dataclass(frozen=True)
class ContextBlock:
    rank: int
    chunk_id: str
    law_id: str | None
    article_id: str | None
    source_passage_ids: tuple[str, ...]
    score: float
    retrieval_source: str | None
    rerank_score: float | None
    text: str


@dataclass(frozen=True)
class ContextBuildResult:
    context: str
    blocks: tuple[ContextBlock, ...]
    included_chunk_ids: tuple[str, ...]
    total_candidates: int
    truncated_chunks: int
    total_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "included_chunk_ids": list(self.included_chunk_ids),
            "included_count": len(self.included_chunk_ids),
            "total_candidates": self.total_candidates,
            "truncated_chunks": self.truncated_chunks,
            "total_chars": self.total_chars,
        }


def build_context(
    retrieved: list[RetrievedChunk],
    *,
    max_chunks: int,
    max_chars: int,
    per_chunk_max_chars: int,
    provenance_map: dict[str, dict[str, Any]] | None = None,
) -> ContextBuildResult:
    if max_chunks <= 0:
        raise ValueError("max_chunks must be > 0")
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if per_chunk_max_chars <= 0:
        raise ValueError("per_chunk_max_chars must be > 0")

    selected: list[ContextBlock] = []
    used_chars = 0
    truncated = 0
    provenance_map = provenance_map or {}
    law_seen_counts: dict[str, int] = {}
    article_seen_counts: dict[str, int] = {}
    max_per_law = max(2, min(4, max_chunks // 2))
    max_per_article = 2

    for rank, doc in enumerate(retrieved[:max_chunks], start=1):
        law_key = str(doc.law_id or "").strip()
        article_key = str(doc.article_id or "").strip()
        if law_key and law_seen_counts.get(law_key, 0) >= max_per_law:
            continue
        if article_key and article_seen_counts.get(article_key, 0) >= max_per_article:
            continue

        provenance = provenance_map.get(doc.chunk_id) or {}
        retrieval_source = str(provenance.get("retrieval_source") or "").strip() or None
        rerank_score = provenance.get("rerank_score")
        if rerank_score is not None:
            try:
                rerank_score = float(rerank_score)
            except Exception:
                rerank_score = None

        body = (doc.text or "").strip()
        if len(body) > per_chunk_max_chars:
            body = body[:per_chunk_max_chars].rstrip()
            truncated += 1
        header = (
            f"[{rank}] chunk_id={doc.chunk_id} law_id={doc.law_id or ''} "
            f"article_id={doc.article_id or ''} score={doc.score:.4f}"
        )
        if retrieval_source:
            header = f"{header} source={retrieval_source}"
        if rerank_score is not None:
            header = f"{header} rerank={rerank_score:.4f}"
        if doc.source_passage_ids:
            header = f"{header} source_passages={','.join(doc.source_passage_ids)}"
        block_text = f"{header}\n{body}"
        if selected and (used_chars + len(block_text) + 2 > max_chars):
            break
        selected.append(
            ContextBlock(
                rank=rank,
                chunk_id=doc.chunk_id,
                law_id=doc.law_id,
                article_id=doc.article_id,
                source_passage_ids=doc.source_passage_ids,
                score=doc.score,
                retrieval_source=retrieval_source,
                rerank_score=rerank_score,
                text=body,
            )
        )
        if law_key:
            law_seen_counts[law_key] = law_seen_counts.get(law_key, 0) + 1
        if article_key:
            article_seen_counts[article_key] = article_seen_counts.get(article_key, 0) + 1
        used_chars += len(block_text) + 2

    if not selected and retrieved:
        doc = retrieved[0]
        body = (doc.text or "").strip()[: max(300, min(per_chunk_max_chars, 800))]
        selected.append(
            ContextBlock(
                rank=1,
                chunk_id=doc.chunk_id,
                law_id=doc.law_id,
                article_id=doc.article_id,
                source_passage_ids=doc.source_passage_ids,
                score=doc.score,
                retrieval_source=None,
                rerank_score=None,
                text=body,
            )
        )
        used_chars = len(body)

    blocks_text: list[str] = []
    for block in selected:
        header = (
            f"[{block.rank}] chunk_id={block.chunk_id} law_id={block.law_id or ''} "
            f"article_id={block.article_id or ''} score={block.score:.4f}"
        )
        if block.retrieval_source:
            header = f"{header} source={block.retrieval_source}"
        if block.rerank_score is not None:
            header = f"{header} rerank={block.rerank_score:.4f}"
        if block.source_passage_ids:
            header = f"{header} source_passages={','.join(block.source_passage_ids)}"
        blocks_text.append(f"{header}\n{block.text}")

    return ContextBuildResult(
        context="\n\n".join(blocks_text),
        blocks=tuple(selected),
        included_chunk_ids=tuple([b.chunk_id for b in selected]),
        total_candidates=len(retrieved),
        truncated_chunks=truncated,
        total_chars=used_chars,
    )

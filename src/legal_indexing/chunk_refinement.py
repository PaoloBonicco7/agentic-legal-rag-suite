from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from statistics import median
from typing import Any, Iterable

from .io import PassageRecord
from .settings import ChunkingProfile


_WORD_RE = re.compile(r"\S+")
_SUFFIX_ORDER: dict[str, int] = {
    "": 0,
    "bis": 1,
    "ter": 2,
    "quater": 3,
    "quinquies": 4,
    "sexies": 5,
    "septies": 6,
    "octies": 7,
    "novies": 8,
    "decies": 9,
    "undecies": 10,
    "duodecies": 11,
    "terdecies": 12,
    "quaterdecies": 13,
    "quinquiesdecies": 14,
}


@dataclass
class RefinedChunk:
    chunk_id: str
    law_id: str
    article_id: str

    text: str
    text_for_embedding: str

    source_passage_ids: tuple[str, ...]
    source_passage_labels: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]

    article_order_in_law: int
    passage_start_order: int
    passage_end_order: int
    article_chunk_order: int

    prev_chunk_id: str | None
    next_chunk_id: str | None

    index_views: tuple[str, ...]
    law_status: str
    article_is_abrogated: bool

    related_law_ids: tuple[str, ...]
    relation_types: tuple[str, ...]
    inbound_law_ids: tuple[str, ...]
    outbound_law_ids: tuple[str, ...]

    law_date: str | None
    law_number: int | None
    law_title: str | None
    status_confidence: float | None
    status_evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ChunkRefinementDiagnostics:
    profile_id: str
    profile: dict[str, int]
    input_passages: int
    output_chunks: int
    merged_units: int
    split_units: int
    input_word_stats: dict[str, float]
    output_word_stats: dict[str, float]
    merge_examples: tuple[dict[str, Any], ...]
    split_examples: tuple[dict[str, Any], ...]


def _word_list(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _word_count(text: str) -> int:
    return len(_word_list(text))


def _stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    vals = sorted(values)
    n = len(vals)

    def p(q: float) -> float:
        idx = max(0, min(n - 1, int(round((n - 1) * q))))
        return float(vals[idx])

    return {
        "count": float(n),
        "min": float(vals[0]),
        "p50": float(median(vals)),
        "p90": p(0.90),
        "p99": p(0.99),
        "max": float(vals[-1]),
    }


def passage_order_key(label: str) -> tuple[int, int, int, int, str]:
    value = (label or "").strip().lower()
    if value == "intro":
        return (0, 0, 0, 0, value)

    m = re.fullmatch(r"c(?P<num>\d+)(?P<suf>[a-z]*)(?:\.lit_(?P<lit>[a-z]))?", value)
    if m:
        num = int(m.group("num"))
        suffix = m.group("suf") or ""
        lit = m.group("lit")
        lit_ord = ord(lit) - 96 if lit else 0
        return (1, num, _SUFFIX_ORDER.get(suffix, 1000), lit_ord, value)

    m2 = re.fullmatch(r"lit_(?P<lit>[a-z])", value)
    if m2:
        lit_ord = ord(m2.group("lit")) - 96
        return (2, 0, 0, lit_ord, value)

    return (3, 0, 0, 0, value)


def _split_words(words: list[str], *, max_words: int, overlap_words: int) -> list[str]:
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def _article_label_from_article_id(article_id: str) -> str:
    if "#art:" in article_id:
        return article_id.split("#art:", 1)[1]
    return article_id


def _merge_unit_text(unit: list[PassageRecord]) -> tuple[str, list[str], list[int]]:
    texts: list[str] = []
    labels: list[str] = []
    counts: list[int] = []
    for p in unit:
        cur = (p.text or "").strip()
        if cur:
            texts.append(cur)
        labels.append(p.passage_label)
        counts.append(_word_count(cur))
    return "\n".join(texts).strip(), labels, counts


def _embedding_prefix(chunk: RefinedChunk) -> str:
    art = _article_label_from_article_id(chunk.article_id)
    date_part = chunk.law_date or "unknown-date"
    number_part = f"n.{chunk.law_number}" if chunk.law_number is not None else "n.?"
    title_part = chunk.law_title or ""
    labels = ", ".join(chunk.source_passage_labels)
    return (
        f"[LR {date_part} {number_part}] {title_part} | Art. {art} | passages: {labels}"
    ).strip()


def refine_chunks_with_diagnostics(
    passages: Iterable[PassageRecord],
    article_order_by_id: dict[str, int],
    chunking_profile: ChunkingProfile,
) -> tuple[list[RefinedChunk], ChunkRefinementDiagnostics]:
    chunking_profile.validate()

    grouped: dict[str, list[PassageRecord]] = defaultdict(list)
    for p in passages:
        if not p.article_id:
            continue
        grouped[p.article_id].append(p)

    merge_examples: list[dict[str, Any]] = []
    split_examples: list[dict[str, Any]] = []

    refined: list[RefinedChunk] = []
    before_sizes: list[int] = []
    after_sizes: list[int] = []
    merged_units = 0
    split_units = 0

    for article_id in sorted(grouped):
        article_passages = sorted(
            grouped[article_id], key=lambda p: (passage_order_key(p.passage_label), p.passage_id)
        )
        for p in article_passages:
            before_sizes.append(_word_count(p.text))

        article_chunks: list[RefinedChunk] = []
        unit_idx = 0
        i = 0
        while i < len(article_passages):
            unit = [article_passages[i]]
            total_words = _word_count(article_passages[i].text)
            j = i + 1
            if total_words < chunking_profile.min_words_merge:
                while j < len(article_passages) and total_words < chunking_profile.min_words_merge:
                    next_words = _word_count(article_passages[j].text)
                    if total_words + next_words > chunking_profile.max_words_split:
                        break
                    unit.append(article_passages[j])
                    total_words += next_words
                    j += 1
                i = j
            else:
                i += 1

            merged_text, unit_labels, unit_counts = _merge_unit_text(unit)
            if len(unit) > 1:
                merged_units += 1
                if len(merge_examples) < 20:
                    merge_examples.append(
                        {
                            "article_id": article_id,
                            "passage_labels": list(unit_labels),
                            "word_counts_before": unit_counts,
                            "word_count_after": _word_count(merged_text),
                        }
                    )

            unit_words = _word_list(merged_text)
            pieces = _split_words(
                unit_words,
                max_words=chunking_profile.max_words_split,
                overlap_words=chunking_profile.overlap_words_split,
            )
            if len(pieces) > 1:
                split_units += 1
                if len(split_examples) < 20:
                    split_examples.append(
                        {
                            "article_id": article_id,
                            "passage_labels": list(unit_labels),
                            "source_words": len(unit_words),
                            "piece_word_counts": [_word_count(x) for x in pieces],
                        }
                    )

            passage_orders = [passage_order_key(p.passage_label) for p in unit]
            start_order = min(passage_orders)
            end_order = max(passage_orders)
            start_ord_idx = start_order[0] * 1_000_000 + start_order[1] * 10_000 + start_order[2] * 100 + start_order[3]
            end_ord_idx = end_order[0] * 1_000_000 + end_order[1] * 10_000 + end_order[2] * 100 + end_order[3]

            source_passage_ids = tuple(p.passage_id for p in unit)
            source_passage_labels = tuple(p.passage_label for p in unit)
            source_chunk_ids = tuple(
                chunk_id
                for p in unit
                for chunk_id in p.source_chunk_ids
                if isinstance(chunk_id, str) and chunk_id.strip()
            )

            union_index_views = sorted({v for p in unit for v in p.index_views})
            union_related_law_ids = sorted({v for p in unit for v in p.related_law_ids})
            union_relation_types = sorted({v for p in unit for v in p.relation_types})
            union_inbound_law_ids = sorted({v for p in unit for v in p.inbound_law_ids})
            union_outbound_law_ids = sorted({v for p in unit for v in p.outbound_law_ids})

            seed = unit[0]
            article_order = int(article_order_by_id.get(article_id, -1))

            for split_idx, part in enumerate(pieces):
                chunk_id = (
                    f"{article_id}#rc:{start_ord_idx}-{end_ord_idx}"
                    f"#u:{unit_idx}#s:{split_idx}"
                )
                chunk = RefinedChunk(
                    chunk_id=chunk_id,
                    law_id=seed.law_id,
                    article_id=seed.article_id,
                    text=part,
                    text_for_embedding="",  # assigned after instantiation
                    source_passage_ids=source_passage_ids,
                    source_passage_labels=source_passage_labels,
                    source_chunk_ids=source_chunk_ids,
                    article_order_in_law=article_order,
                    passage_start_order=start_ord_idx,
                    passage_end_order=end_ord_idx,
                    article_chunk_order=-1,
                    prev_chunk_id=None,
                    next_chunk_id=None,
                    index_views=tuple(union_index_views),
                    law_status=seed.law_status,
                    article_is_abrogated=bool(seed.article_is_abrogated),
                    related_law_ids=tuple(union_related_law_ids),
                    relation_types=tuple(union_relation_types),
                    inbound_law_ids=tuple(union_inbound_law_ids),
                    outbound_law_ids=tuple(union_outbound_law_ids),
                    law_date=seed.law_date,
                    law_number=seed.law_number,
                    law_title=seed.law_title,
                    status_confidence=seed.status_confidence,
                    status_evidence=seed.status_evidence,
                )
                chunk.text_for_embedding = f"{_embedding_prefix(chunk)}\n\n{chunk.text}".strip()
                article_chunks.append(chunk)
                after_sizes.append(_word_count(part))

            unit_idx += 1

        article_chunks.sort(
            key=lambda c: (c.passage_start_order, c.passage_end_order, c.chunk_id)
        )
        for idx, c in enumerate(article_chunks):
            c.article_chunk_order = idx
            c.prev_chunk_id = article_chunks[idx - 1].chunk_id if idx > 0 else None
            c.next_chunk_id = article_chunks[idx + 1].chunk_id if idx + 1 < len(article_chunks) else None
            refined.append(c)

    refined.sort(
        key=lambda c: (
            c.law_id,
            c.article_order_in_law,
            c.passage_start_order,
            c.passage_end_order,
            c.article_chunk_order,
            c.chunk_id,
        )
    )

    diagnostics = ChunkRefinementDiagnostics(
        profile_id=chunking_profile.profile_id,
        profile={
            "min_words_merge": chunking_profile.min_words_merge,
            "max_words_split": chunking_profile.max_words_split,
            "overlap_words_split": chunking_profile.overlap_words_split,
        },
        input_passages=len(before_sizes),
        output_chunks=len(refined),
        merged_units=merged_units,
        split_units=split_units,
        input_word_stats=_stats(before_sizes),
        output_word_stats=_stats(after_sizes),
        merge_examples=tuple(merge_examples),
        split_examples=tuple(split_examples),
    )

    return refined, diagnostics


def refine_chunks(
    passages: Iterable[PassageRecord],
    article_order_by_id: dict[str, int],
    chunking_profile: ChunkingProfile,
) -> list[RefinedChunk]:
    refined, _ = refine_chunks_with_diagnostics(passages, article_order_by_id, chunking_profile)
    return refined


__all__ = [
    "RefinedChunk",
    "ChunkRefinementDiagnostics",
    "passage_order_key",
    "refine_chunks",
    "refine_chunks_with_diagnostics",
]

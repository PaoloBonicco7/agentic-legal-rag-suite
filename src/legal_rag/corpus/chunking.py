"""Source-exact article aggregation with a complete token budget."""

import re
from dataclasses import dataclass

from ..artifacts import digest
from ..contracts import Chunk, Document, SourceSpan, Unit
from ..tokens import TokenCounter
from .parser import substantive

CHUNKING_VERSION = "contextual-chunking-2"


@dataclass
class Piece:
    text: str
    spans: list[SourceSpan]
    unit_ids: list[str]
    status: str


def _slice_spans(unit: Unit, start: int, end: int) -> list[SourceSpan]:
    result, position = [], 0
    for span in unit.spans:
        length = span.end - span.start
        left, right = max(start, position), min(end, position + length)
        if left < right:
            result.append(SourceSpan(block_id=span.block_id, start=span.start + left - position, end=span.start + right - position))
        position += length + 1
    return result


def build_chunks(document: Document, units: list[Unit], counter: TokenCounter, target_tokens: int = 384, max_tokens: int = 512) -> list[Chunk]:
    if not 0 < target_tokens <= max_tokens:
        raise ValueError("Expected 0 < target_tokens <= max_tokens")
    by_id = {u.unit_id: u for u in units}
    groups: dict[str, list[Unit]] = {}
    for unit in units:
        if unit.kind == "article" or not substantive(unit.text):
            continue
        if unit.kind == "annex" and re.search(r"\bomissis\b", unit.text, re.I):
            continue
        key = unit.article_label or ("unstructured" if unit.kind == "unstructured" else unit.unit_id)
        groups.setdefault(key, []).append(unit)
    chunks = []
    for group in groups.values():
        first = group[0]
        article = next((u for u in units if u.kind == "article" and u.article_label == first.article_label), None) if first.article_label else None
        heading = article.heading if article else first.heading
        label = first.article_label
        prefix = f"{document.identity}\n{document.title}\n"
        if label:
            prefix += f"Articolo {label}. {heading}\n"
        elif heading:
            prefix += heading + "\n"
        if counter.count(prefix) >= max_tokens:
            raise ValueError(f"Metadata prefix exceeds token budget: {document.document_id}")
        pending: list[Piece] = []

        def text_of(pieces: list[Piece]) -> str:
            return "\n".join(piece.text for piece in pieces)

        def emit() -> None:
            if not pending:
                return
            text = text_of(pending)
            spans = [span for piece in pending for span in piece.spans]
            embedding_text = prefix + text
            if counter.count(embedding_text) > max_tokens:
                raise ValueError(f"Chunk exceeds token budget: {document.document_id}")
            status = "ceased_confirmed" if any(p.status == "ceased_confirmed" for p in pending) else "unknown" if document.status == "unknown" or any(p.status == "unknown" for p in pending) else "no_cessation_evidence"
            uid = digest(f"{document.snapshot_id}:{embedding_text}:{[(s.block_id, s.start, s.end) for s in spans]}")[:24]
            chunks.append(Chunk(chunk_id="chunk:" + uid, document_id=document.document_id, snapshot_id=document.snapshot_id, article_label=label, title=document.title, identity=document.identity, heading=heading, text=text, embedding_text=embedding_text, unit_ids=list(dict.fromkeys(uid for piece in pending for uid in piece.unit_ids)), spans=spans, source_url=document.source_url, status=status, warnings=list(document.warnings)))
            pending.clear()

        def pieces(unit: Unit) -> list[Piece]:
            ids = [unit.unit_id]
            parent = by_id.get(unit.parent_id)
            while parent:
                ids.append(parent.unit_id)
                parent = by_id.get(parent.parent_id)
            if counter.count(prefix + unit.text) <= max(target_tokens, counter.count(prefix) + 1):
                return [Piece(unit.text, unit.spans, ids, unit.status)]
            result, start = [], 0
            ends = [match.end() for match in re.finditer(r"\S+", unit.text)]
            budget = max(target_tokens, counter.count(prefix) + 1)
            while start < len(unit.text):
                while start < len(unit.text) and unit.text[start].isspace():
                    start += 1
                if start == len(unit.text):
                    break
                candidates = [end for end in ends if end > start]
                low, high, best = 0, len(candidates) - 1, None
                while low <= high:
                    middle = (low + high) // 2
                    end = candidates[middle]
                    if counter.count(prefix + unit.text[start:end]) <= budget:
                        best = end
                        low = middle + 1
                    else:
                        high = middle - 1
                if best is None:
                    low, high, best = start + 1, candidates[0], start
                    while low <= high:
                        middle = (low + high) // 2
                        if counter.count(prefix + unit.text[start:middle]) <= max_tokens:
                            best = middle
                            low = middle + 1
                        else:
                            high = middle - 1
                    if best == start:
                        raise ValueError(f"Cannot fit source text into token budget: {unit.unit_id}")
                tail = unit.text[best:].strip()
                if tail and not substantive(tail):
                    if counter.count(prefix + unit.text[start:]) <= max_tokens:
                        best = len(unit.text)
                    else:
                        earlier = [end for end in candidates if end < best and substantive(unit.text[end:].strip())]
                        if earlier:
                            best = earlier[-1]
                end = best
                while end > start and unit.text[end - 1].isspace():
                    end -= 1
                result.append(Piece(unit.text[start:end], _slice_spans(unit, start, end), ids, unit.status))
                start = best
            return result

        for unit in group:
            for piece in pieces(unit):
                changes_status = pending and (pending[-1].status == "ceased_confirmed") != (piece.status == "ceased_confirmed")
                exceeds_target = pending and counter.count(prefix + text_of(pending + [piece])) > target_tokens
                if changes_status or exceeds_target:
                    emit()
                if not pending:
                    parent = by_id.get(unit.parent_id) if unit.kind == "letter" else None
                    if parent and parent.kind == "paragraph" and substantive(parent.text) and (parent.status == "ceased_confirmed") == (piece.status == "ceased_confirmed"):
                        intro = Piece(parent.text, parent.spans, [parent.unit_id] + ([article.unit_id] if article else []), parent.status)
                        if counter.count(prefix + text_of([intro, piece])) <= max_tokens:
                            pending.append(intro)
                pending.append(piece)
                if counter.count(prefix + text_of(pending)) >= target_tokens:
                    emit()
        emit()
    return chunks

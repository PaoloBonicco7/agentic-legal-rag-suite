from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_ABROGATION_FULL_RE = re.compile(
    r"(legge\s+abrogata|\babrogat[oa]\b.*\bdall[ao]\b|\bnon\s+piu\s+in\s+vigore\b|\bnon\s+più\s+in\s+vigore\b|\bcessat\w*\s+efficac)",
    re.IGNORECASE,
)
_PARTIAL_EXCEPTION_RE = re.compile(r"\babrogat[oa]\b.{0,120}\bad\s+eccezione", re.IGNORECASE | re.DOTALL)
_INDEX_RE = re.compile(r"\bINDICE\b", re.IGNORECASE)


@dataclass(frozen=True)
class StatusResult:
    status: str
    status_confidence: float
    status_evidence: list[dict]


def _snippet(text: str, regex: re.Pattern[str], max_chars: int = 220) -> str | None:
    m = regex.search(text or "")
    if not m:
        return None
    start = max(0, m.start() - 40)
    end = min(len(text), m.end() + 120)
    out = (text[start:end] or "").strip().replace("\n", " ")
    return out[:max_chars]


def classify_law_status(
    *,
    preamble_text: str,
    article_count: int,
    source_file: str,
    ingest_status: str | None = None,
) -> StatusResult:
    txt = (preamble_text or "").strip()

    evidence: list[dict] = []

    partial = bool(_PARTIAL_EXCEPTION_RE.search(txt))
    abrog = bool(_ABROGATION_FULL_RE.search(txt))
    is_index = bool(_INDEX_RE.search(txt)) and article_count == 0

    if partial:
        s = _snippet(txt, _PARTIAL_EXCEPTION_RE)
        if s:
            evidence.append({"kind": "partial_abrogation", "snippet": s, "source_file": source_file})
        return StatusResult(status="unknown", status_confidence=0.65, status_evidence=evidence)

    if abrog:
        s = _snippet(txt, _ABROGATION_FULL_RE)
        if s:
            evidence.append({"kind": "abrogation_phrase", "snippet": s, "source_file": source_file})
        return StatusResult(status="past", status_confidence=0.97, status_evidence=evidence)

    if is_index:
        evidence.append(
            {
                "kind": "index_without_articles",
                "snippet": "INDICE con assenza di articoli strutturati",
                "source_file": source_file,
            }
        )
        return StatusResult(status="index_or_empty", status_confidence=0.9, status_evidence=evidence)

    if ingest_status == "abrogated":
        evidence.append({"kind": "ingest_status", "snippet": "abrogated", "source_file": source_file})
        return StatusResult(status="past", status_confidence=0.9, status_evidence=evidence)

    if ingest_status == "in_force" and article_count > 0:
        evidence.append({"kind": "ingest_status", "snippet": "in_force", "source_file": source_file})
        return StatusResult(status="current", status_confidence=0.75, status_evidence=evidence)

    if article_count > 0:
        return StatusResult(status="current", status_confidence=0.72, status_evidence=evidence)

    evidence.append({"kind": "insufficient_evidence", "snippet": "nessuna regola deterministica applicabile", "source_file": source_file})
    return StatusResult(status="unknown", status_confidence=0.5, status_evidence=evidence)


def classify_many_status(laws: Iterable[dict], articles_by_law: dict[str, int]) -> list[dict]:
    out: list[dict] = []
    for law in laws:
        law_id = law.get("law_id")
        result = classify_law_status(
            preamble_text=str(law.get("preamble_text") or ""),
            article_count=int(articles_by_law.get(str(law_id), 0)),
            source_file=str(law.get("source_file") or ""),
            ingest_status=(str(law.get("status")) if law.get("status") is not None else None),
        )
        out.append(
            {
                "law_id": law_id,
                "status": result.status,
                "status_confidence": float(result.status_confidence),
                "status_evidence": result.status_evidence,
            }
        )
    return out

from __future__ import annotations

import hashlib
import re

from laws_ingestion.core.utils import parse_italian_date


_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})(?:\s*°)?\s+(?P<month>[a-zà]+)\s+(?P<year>\d{4})",
    re.IGNORECASE,
)


_EVENT_TYPE_MAP = {
    "ABROGATES": "REPEAL",
    "ABROGATED_BY": "REPEAL",
    "AMENDS": "AMEND",
    "MODIFIED_BY": "AMEND",
    "REPLACES": "REPLACE",
    "REPLACED_BY": "REPLACE",
    "INSERTS": "INSERT",
    "INSERTED_BY": "INSERT",
}


def _effective_date(evidence: str) -> str | None:
    m = _DATE_RE.search(evidence or "")
    if not m:
        return None
    try:
        d = parse_italian_date(int(m.group("day")), m.group("month"), int(m.group("year")))
    except Exception:
        return None
    return d.isoformat()


def _event_id(rec: dict) -> str:
    h = hashlib.sha256()
    key = "|".join(
        [
            str(rec.get("event_type") or ""),
            str(rec.get("source_law_id") or ""),
            str(rec.get("target_law_id") or ""),
            str(rec.get("source_article_id") or ""),
            str(rec.get("source_passage_id") or ""),
            str(rec.get("target_article_label_norm") or ""),
            str(rec.get("evidence") or ""),
        ]
    )
    h.update(key.encode("utf-8"))
    return h.hexdigest()


def extract_events(edges: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()

    for edge in sorted(edges, key=lambda x: str(x.get("norm_edge_id") or "")):
        relation_type = str(edge.get("relation_type") or "")
        event_type = _EVENT_TYPE_MAP.get(relation_type)
        if not event_type:
            continue

        evidence = str(edge.get("evidence") or edge.get("evidence_text") or "").strip()
        rec = {
            "event_type": event_type,
            "source_law_id": edge.get("src_law_id"),
            "target_law_id": edge.get("dst_law_id"),
            "source_article_id": edge.get("src_article_id"),
            "source_passage_id": edge.get("src_passage_id"),
            "target_article_label_norm": edge.get("dst_article_label_norm"),
            "effective_date": _effective_date(evidence),
            "evidence": evidence[:500],
            "confidence": float(edge.get("confidence") or 0.0),
            "source_file": edge.get("source_file"),
        }
        rec["event_id"] = _event_id(rec)
        if rec["event_id"] in seen:
            continue
        seen.add(rec["event_id"])
        out.append(rec)

    out.sort(key=lambda x: str(x.get("event_id") or ""))
    return out

"""Deterministic legal-validity helpers for preprocessing v2."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from .common import normalize_article_label, normalize_ws

ABROGATION_RE = re.compile(
    r"(?:\babrogat(?:o(?:\s*il)?|a|i|e)\b|\babrog[ée]e\b|\bnon\s+pi[uù]\s+in\s+vigore\b)",
    re.IGNORECASE,
)
EXPIRATION_RE = re.compile(
    r"(?:\bcessat\w*.{0,50}\b(?:efficac|vigor)\w*"
    r"|\bcessazion\w*.{0,50}\b(?:efficac|vigor)\w*)",
    re.IGNORECASE,
)
ALL_EXCEPT_RE = re.compile(r"\bad\s+eccezione\b", re.IGNORECASE)
LATER_TOTAL_CESSATION_RE = re.compile(
    r"(?:\bche\s+sono\s+stat[ei]\s+poi\s+abrogat[ei]\b"
    r"|\b(?:e|ed)\s+(?:poi\s+)?(?:interamente\s+)?abrogat[oaie]\b)",
    re.IGNORECASE,
)
ANCHORLESS_NOTE_RE = re.compile(
    r"^\s*\((?P<label>\d[\da-z.]*)\)\s*(?P<rest>\S.*)$",
    re.IGNORECASE,
)
VISIBLE_NOTE_RE = re.compile(r"\((?P<label>\d[\da-z.]*)\)", re.IGNORECASE)
ARTICLE_EXCEPTION_RE = re.compile(
    r"\bart(?:icolo|icoli|t)\.?\s+(?P<label>\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)",
    re.IGNORECASE,
)

DEFAULT_CURRENT_RULE = "status-no-explicit-cessation-v1"
EMPTY_CONTENT_RULE = "status-no-substantive-content-v1"
BRACKETED_UNKNOWN_RULE = "status-bracketed-unresolved-v1"
ROLLUP_MIXED_RULE = "status-rollup-mixed-v1"
ROLLUP_UNKNOWN_RULE = "status-rollup-unknown-v1"


def cessation_event_type(text: str) -> str | None:
    """Return the explicit cessation type found in one source clause."""
    clause = normalize_ws(text)
    if EXPIRATION_RE.search(clause):
        return "expiration"
    if ABROGATION_RE.search(clause):
        return "repeal"
    return None


def target_kind_from_clause(text: str, *, source_kind: str) -> str:
    """Infer the affected structural level from the decisive clause."""
    if source_kind == "preamble":
        return "law"
    clause = normalize_ws(text).lower().lstrip("( ")
    operative = re.split(
        r"\babrogat(?:o(?:\s*il)?|a|i|e)\b|\babrog[ée]e\b|\bcessat",
        clause,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    if re.search(r"\bletter[ae]\b", operative):
        return "letter"
    if re.search(r"\bcomm[ao]\b|\bcommi\b", operative):
        return "comma"
    if re.search(r"\barticol[oi]\b", operative):
        return "article"
    if re.search(r"\ballegat[oi]\b", operative):
        return "annex"
    return "unknown"


def note_kind_from_clause(text: str) -> str:
    """Classify a note using only its first operative clause."""
    clause = normalize_ws(text).lower()
    if cessation_event_type(clause):
        return "abrogated"
    if re.search(r"\bmodificat|\bsostituit|\bsostituisc", clause):
        return "modified"
    if re.search(r"\binserit|\baggiunt", clause):
        return "inserted"
    return "other"


def anchorless_note_definition(text: str) -> tuple[str, str] | None:
    """Recognize visible note markers that lack a named HTML anchor."""
    match = ANCHORLESS_NOTE_RE.match(text or "")
    if not match:
        return None
    return f"nota_{match.group('label')}", match.group("rest").strip()


def visible_note_anchor_names(text: str, known_anchors: set[str]) -> list[str]:
    """Resolve visible markers only when the corresponding note exists."""
    found: list[str] = []
    seen: set[str] = set()
    for match in VISIBLE_NOTE_RE.finditer(text or ""):
        anchor = f"nota_{match.group('label')}"
        if anchor in known_anchors and anchor not in seen:
            seen.add(anchor)
            found.append(anchor)
    return found


def is_fully_bracketed(text: str) -> bool:
    """Return whether a whole passage is editorially enclosed in brackets."""
    value = normalize_ws(text)
    if not value.startswith("["):
        return False
    closing = value.rfind("]")
    if closing <= 0:
        return False
    tail = value[closing + 1 :]
    return bool(re.fullmatch(r"[\s.,;:]*?(?:\(\d[\da-z.]*\)[\s.,;:]*)*", tail, re.IGNORECASE))


def exception_article_labels(text: str) -> list[str]:
    """Extract explicit article exceptions without reading later citations."""
    match = ALL_EXCEPT_RE.search(text or "")
    if not match:
        return []
    tail = (text or "")[match.end() :]
    labels = [normalize_article_label(item.group("label")) for item in ARTICLE_EXCEPTION_RE.finditer(tail)]
    if labels:
        return list(dict.fromkeys(labels))
    list_match = re.search(
        r"\b(?:articoli|artt\.)\s+(?P<labels>[^.;)]*?)(?:\b(?:dalla|dal|ai\s+sensi)\b|[.;)]|$)",
        tail,
        re.IGNORECASE,
    )
    if not list_match:
        return []
    values = re.findall(
        r"\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?",
        list_match.group("labels"),
        re.IGNORECASE,
    )
    return list(dict.fromkeys(normalize_article_label(value) for value in values))


def split_later_total_cessation(text: str) -> tuple[str, str] | None:
    """Split an all-except clause when its exceptions later cease too."""
    value = normalize_ws(text)
    exception = ALL_EXCEPT_RE.search(value)
    if not exception:
        return None
    later = LATER_TOTAL_CESSATION_RE.search(value, exception.end())
    if not later:
        return None
    return value[: later.start()].rstrip(" ,;"), value[later.start() :].strip()


def content_availability(*, text: str, structured: bool, metadata_present: bool = False) -> str:
    """Classify content presence independently from legal validity."""
    if normalize_ws(text):
        return "substantive" if structured else "unstructured"
    return "metadata_only" if metadata_present else "empty"


def rollup_validity(statuses: Iterable[str], *, default: str) -> tuple[str, str]:
    """Roll child validity up without treating ambiguity as repeal."""
    values = list(statuses)
    if not values:
        return default, EMPTY_CONTENT_RULE if default == "unknown" else DEFAULT_CURRENT_RULE
    unique = set(values)
    if unique == {"past"}:
        return "past", "status-rollup-complete-past-v1"
    if "unknown" in unique:
        return "unknown", ROLLUP_UNKNOWN_RULE
    if "partial" in unique or ("past" in unique and len(unique) > 1):
        return "partial", ROLLUP_MIXED_RULE
    return "current", DEFAULT_CURRENT_RULE


def status_event_id(event: dict[str, Any]) -> str:
    """Build a stable event ID from canonical source and resolution fields."""
    canonical = {key: value for key, value in event.items() if key != "status_event_id"}
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

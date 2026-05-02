"""Shared normalization helpers for phase 1 laws preprocessing."""

from __future__ import annotations

import re

ITALIAN_MONTHS = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}

ARTICLE_SUFFIXES = (
    "bis",
    "ter",
    "quater",
    "quinquies",
    "sexies",
    "septies",
    "octies",
    "novies",
    "decies",
)

WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Collapse HTML-derived whitespace into a stable single-line string."""
    return WS_RE.sub(" ", (text or "").replace("\xa0", " ")).strip()


def normalize_month_name(month: str) -> str:
    """Normalize Italian month names before filename/date parsing."""
    return (month or "").strip().lower().replace("à", "a")


def normalize_article_label(raw: str) -> str:
    """Convert raw article labels such as ``Art. 1 bis`` into stable IDs."""
    text = re.sub(r"\s+", " ", (raw or "").strip())
    text = re.sub(r"^(?:articolo|art)\.?\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(".").strip()
    suffixes = "|".join(ARTICLE_SUFFIXES)
    match = re.fullmatch(rf"(?P<num>\d+)\s*(?P<suf>{suffixes})?", text, flags=re.IGNORECASE)
    if match:
        suffix = match.group("suf")
        return str(int(match.group("num"))) + (suffix.lower() if suffix else "")
    if text.lower() == "unico":
        return "unico"
    return re.sub(r"\s+", "", text).lower()

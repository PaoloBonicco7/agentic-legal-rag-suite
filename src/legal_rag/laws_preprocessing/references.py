"""Reference extraction and relation typing for explicit legal citations."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, unquote, urlparse

from .common import normalize_article_label
from .inventory import parse_italian_date
from .models import CorpusRegistry, ResolvedLawRef

LR_SHORT_RE = r"(?:L[\.:]\s*R\.?\.?)"
FULL_DATE_REF_RE = re.compile(
    rf"(?:Legge\s+regionale|{LR_SHORT_RE})\s+"
    rf"(?P<day>\d{{1,2}})(?:\s*°)?\s+(?P<month>[a-zà]+)\s+(?P<year>\d{{4}}),\s*n\.\s*(?P<num>\d+)",
    re.IGNORECASE,
)
YEAR_NUM_REF_RE = re.compile(rf"\b(?:{LR_SHORT_RE})\s*(?P<num>\d+)\s*/\s*(?P<year>\d{{2,4}})\b", re.IGNORECASE)
ART_REF_RE = re.compile(
    r"\bart\.?\s*(?P<label>\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?|unico)\b",
    re.IGNORECASE,
)


def _normalize_two_digit_year(year: int) -> int:
    """Expand two-digit years using the same legal-corpus convention."""
    return 1900 + year if year >= 50 else 2000 + year


def parse_numero_legge_param(href: str) -> tuple[int, int] | None:
    """Read ``numero_legge`` query parameters from corpus hyperlinks."""
    href = unquote(href or "")
    try:
        query = parse_qs(urlparse(href).query)
    except Exception:
        return None
    raw = (query.get("numero_legge") or [""])[0]
    if not raw or "/" not in raw:
        return None
    left, right = (part.strip() for part in raw.split("/", 1))
    if not left.isdigit() or not right.isdigit():
        return None
    year = int(right) if len(right) == 4 else _normalize_two_digit_year(int(right))
    return int(left), year


def resolve_refs_from_text(text: str, registry: CorpusRegistry) -> tuple[list[ResolvedLawRef], int]:
    """Resolve explicit textual law citations against the corpus registry."""
    refs: list[ResolvedLawRef] = []
    unresolved = 0
    for match in FULL_DATE_REF_RE.finditer(text or ""):
        try:
            law_date = parse_italian_date(int(match.group("day")), match.group("month"), int(match.group("year")))
        except ValueError:
            unresolved += 1
            continue
        law_number = int(match.group("num"))
        law_id = registry.resolve_law_id(law_date, law_number)
        if not law_id:
            unresolved += 1
            continue
        refs.append(ResolvedLawRef(law_id, "text_regex", match.group(0), f"{law_date.isoformat()}:{law_number}"))
    for match in YEAR_NUM_REF_RE.finditer(text or ""):
        law_number = int(match.group("num"))
        year_raw = match.group("year")
        year = int(year_raw) if len(year_raw) == 4 else _normalize_two_digit_year(int(year_raw))
        law_id = registry.resolve_law_id_year(year, law_number)
        if not law_id:
            unresolved += 1
            continue
        refs.append(ResolvedLawRef(law_id, "text_regex", match.group(0), f"{year}:{law_number}"))
    deduped: list[ResolvedLawRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.law_id in seen:
            continue
        seen.add(ref.law_id)
        deduped.append(ref)
    return deduped, unresolved


def resolve_ref_from_href_and_text(
    href: str, anchor_text: str, registry: CorpusRegistry
) -> tuple[ResolvedLawRef | None, int]:
    """Resolve a legal hyperlink, preferring its visible citation text."""
    text_refs, unresolved = resolve_refs_from_text(anchor_text, registry)
    if text_refs:
        ref = text_refs[0]
        return ResolvedLawRef(ref.law_id, "href", anchor_text.strip()[:300], ref.law_key_raw), unresolved
    parsed = parse_numero_legge_param(href)
    if not parsed:
        return None, unresolved
    law_number, year = parsed
    law_id = registry.resolve_law_id_year(year, law_number)
    if not law_id:
        return None, unresolved + 1
    return ResolvedLawRef(law_id, "href", href.strip()[:300], f"{year}:{law_number}"), unresolved


def extract_dst_article_label_norm(evidence_text: str) -> str | None:
    """Extract a normalized destination article label when the evidence says it."""
    match = ART_REF_RE.search(evidence_text or "")
    return normalize_article_label(match.group("label")) if match else None


def classify_relation_type(evidence_text: str) -> tuple[str, float]:
    """Map explicit citation evidence to the allowed relation taxonomy."""
    text = (evidence_text or "").lower()
    if re.search(r"\babrogat[oa]\b.*\bdall['a-z]*\b|\blegge\s+abrogata\b|\bcessat\w*\s+efficac", text):
        return "ABROGATED_BY", 0.9
    if re.search(r"\babroga\b|\babrogano\b|\bsono\s+abrogat", text):
        return "ABROGATES", 0.85
    if re.search(r"\bsostituit[oa]\b.*\bdall['a-z]*\b", text):
        return "REPLACED_BY", 0.85
    if re.search(r"\bsostituisc\w*\b|\bsostituit[oa]\b", text):
        return "REPLACES", 0.75
    if re.search(r"\binserit[oa]\b.*\bdall['a-z]*\b", text):
        return "INSERTED_BY", 0.8
    if re.search(r"\binserisc\w*\b|\baggiung\w*\b|\binserit[oa]\b", text):
        return "INSERTS", 0.7
    if re.search(r"\bmodificat[oa]\b.*\bdall['a-z]*\b", text):
        return "MODIFIED_BY", 0.8
    if re.search(r"\bmodific\w*\b", text):
        return "AMENDS", 0.7
    return "REFERENCES", 0.45


def edge_id(
    src_law_id: str,
    src_article_id: str | None,
    src_passage_id: str | None,
    relation_type: str,
    dst_law_id: str,
    dst_article_label_norm: str | None,
    extraction_method: str,
    evidence_text: str,
) -> str:
    """Create a deterministic edge ID from structural endpoints and evidence."""
    parts = [
        src_law_id or "",
        src_article_id or "",
        src_passage_id or "",
        relation_type,
        dst_law_id,
        dst_article_label_norm or "",
        extraction_method,
        evidence_text or "",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

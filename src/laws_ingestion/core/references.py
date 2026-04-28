from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import parse_qs, unquote, urlparse

from .registry import CorpusRegistry
from .utils import normalize_article_label, parse_italian_date

_LR_SHORT_RE = r"(?:L[\.:]\s*R\.?\.?)"

_FULL_DATE_REF_RE = re.compile(
    rf"(?:Legge\s+regionale|{_LR_SHORT_RE})\s+(?P<day>\d{{1,2}})(?:\s*°)?\s+(?P<month>[a-zà]+)\s+(?P<year>\d{{4}}),\s*n\.\s*(?P<num>\d+)",
    re.IGNORECASE,
)

_YEAR_NUM_REF_RE = re.compile(
    rf"\b(?:{_LR_SHORT_RE})\s*(?P<num>\d+)\s*/\s*(?P<year>\d{{2,4}})\b",
    re.IGNORECASE,
)

_ART_REF_RE = re.compile(
    r"\bart\.?\s*(?P<label>\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?|unico)\b",
    re.IGNORECASE,
)


def extract_dst_article_label_norm(evidence_text: str) -> str | None:
    m = _ART_REF_RE.search(evidence_text or "")
    if not m:
        return None
    return normalize_article_label(m.group("label"))


def _normalize_two_digit_year(yy: int) -> int:
    if yy >= 50:
        return 1900 + yy
    return 2000 + yy


def parse_numero_legge_param(href: str) -> tuple[int, int] | None:
    """
    Parse `numero_legge=NUM/YY` or `NUM/YYYY` from a href.
    Returns (law_number, law_year) if found.
    """
    href = unquote(href or "")
    try:
        q = parse_qs(urlparse(href).query)
    except Exception:
        return None
    raw = (q.get("numero_legge") or [""])[0]
    if not raw or "/" not in raw:
        return None
    left, right = (p.strip() for p in raw.split("/", 1))
    if not left.isdigit() or not right.isdigit():
        return None
    num = int(left)
    year_raw = int(right)
    year = year_raw if len(right) == 4 else _normalize_two_digit_year(year_raw)
    return num, year


@dataclass(frozen=True)
class ResolvedLawRef:
    law_id: str
    extraction_method: str  # "href" | "text_regex"
    raw: str
    law_key_raw: str


def resolve_refs_from_text(text: str, registry: CorpusRegistry) -> tuple[list[ResolvedLawRef], int]:
    """
    Returns (resolved_refs, unresolved_count) from text-only patterns.
    """
    resolved: list[ResolvedLawRef] = []
    unresolved = 0

    for m in _FULL_DATE_REF_RE.finditer(text or ""):
        try:
            d = parse_italian_date(int(m.group("day")), m.group("month"), int(m.group("year")))
        except Exception:
            unresolved += 1
            continue
        num = int(m.group("num"))
        law_id = registry.resolve_law_id(d, num)
        if not law_id:
            unresolved += 1
            continue
        resolved.append(
            ResolvedLawRef(
                law_id=law_id,
                extraction_method="text_regex",
                raw=m.group(0),
                law_key_raw=f"{d.isoformat()}:{num}",
            )
        )

    for m in _YEAR_NUM_REF_RE.finditer(text or ""):
        num = int(m.group("num"))
        yraw = m.group("year")
        year = int(yraw) if len(yraw) == 4 else _normalize_two_digit_year(int(yraw))
        law_id = registry.resolve_law_id_year(year, num)
        if not law_id:
            unresolved += 1
            continue
        resolved.append(
            ResolvedLawRef(
                law_id=law_id,
                extraction_method="text_regex",
                raw=m.group(0),
                law_key_raw=f"{year}:{num}",
            )
        )

    seen: set[str] = set()
    out: list[ResolvedLawRef] = []
    for r in resolved:
        if r.law_id in seen:
            continue
        seen.add(r.law_id)
        out.append(r)
    return out, unresolved


def resolve_ref_from_href_and_text(
    href: str, anchor_text: str, registry: CorpusRegistry
) -> tuple[ResolvedLawRef | None, int]:
    """
    Returns (resolved_ref_or_none, unresolved_count).
    This is considered extraction_method="href" (even if the anchor text is used).
    """
    unresolved = 0

    m = _FULL_DATE_REF_RE.search(anchor_text or "")
    if m:
        try:
            d = parse_italian_date(int(m.group("day")), m.group("month"), int(m.group("year")))
            num = int(m.group("num"))
            law_id = registry.resolve_law_id(d, num)
            if law_id:
                return (
                    ResolvedLawRef(
                        law_id=law_id,
                        extraction_method="href",
                        raw=(anchor_text or "").strip()[:300],
                        law_key_raw=f"{d.isoformat()}:{num}",
                    ),
                    0,
                )
        except Exception:
            unresolved += 1

    m2 = _YEAR_NUM_REF_RE.search(anchor_text or "")
    if m2:
        num = int(m2.group("num"))
        yraw = m2.group("year")
        year = int(yraw) if len(yraw) == 4 else _normalize_two_digit_year(int(yraw))
        law_id = registry.resolve_law_id_year(year, num)
        if law_id:
            return (
                ResolvedLawRef(
                    law_id=law_id,
                    extraction_method="href",
                    raw=(anchor_text or "").strip()[:300],
                    law_key_raw=f"{year}:{num}",
                ),
                0,
            )
        unresolved += 1

    parsed = parse_numero_legge_param(href or "")
    if not parsed:
        return None, unresolved
    num, year = parsed
    law_id = registry.resolve_law_id_year(year, num)
    if not law_id:
        return None, unresolved + 1
    return (
        ResolvedLawRef(
            law_id=law_id,
            extraction_method="href",
            raw=(href or "").strip()[:300],
            law_key_raw=f"{year}:{num}",
        ),
        unresolved,
    )


def extract_note_anchor_names_from_hrefs(hrefs: list[str]) -> list[str]:
    out: list[str] = []
    for h in hrefs:
        h = (h or "").strip()
        if h.startswith("#nota_"):
            out.append(h[1:])
    seen: set[str] = set()
    dedup: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup

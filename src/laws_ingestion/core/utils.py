from __future__ import annotations

from datetime import date
import hashlib
import re
from pathlib import Path
from typing import Iterable

_ITALIAN_MONTHS = {
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

_ARTICLE_SUFFIXES = (
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


def normalize_month_name(month: str) -> str:
    month = (month or "").strip().lower()
    return month.replace("à", "a")


def parse_italian_date(day: int, month: str, year: int) -> date:
    month_num = _ITALIAN_MONTHS.get(normalize_month_name(month))
    if not month_num:
        raise ValueError(f"Unknown month name: {month!r}")
    return date(int(year), int(month_num), int(day))


def normalize_article_label(raw: str) -> str:
    """
    Normalize article labels for stable IDs.

    Examples:
    - "Art. 4 bis" -> "4bis"
    - "Art.16" -> "16"
    - "ARTICOLO 1" -> "1"
    - "Articolo 10 bis" -> "10bis"
    """
    text = re.sub(r"\s+", " ", (raw or "").strip())
    text = re.sub(r"^(?:articolo|art)\.?\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(".").strip()

    m = re.fullmatch(
        rf"(?P<num>\d+)\s*(?P<suf>{'|'.join(_ARTICLE_SUFFIXES)})?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        num = str(int(m.group("num")))
        suf = m.group("suf")
        return num + (suf.lower() if suf else "")

    return re.sub(r"\s+", "", text).lower()


def law_id_from_date_number(law_date: date, law_number: int) -> str:
    return f"vda:lr:{law_date.isoformat()}:{int(law_number)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_id(html_files: Iterable[Path]) -> str:
    """
    Deterministic dataset id based on file content.

    The order is stable (sorted by filename) and the hash includes both filename and file sha256.
    """
    h = hashlib.sha256()
    for path in sorted(html_files, key=lambda p: p.name):
        file_sha = sha256_file(path)
        h.update(path.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(file_sha.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


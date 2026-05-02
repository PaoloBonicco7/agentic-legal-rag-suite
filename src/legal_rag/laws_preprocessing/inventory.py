"""Source corpus inventory and stable law identity helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any

from .common import ITALIAN_MONTHS, normalize_month_name
from .models import CorpusRegistry, LawFile

FILENAME_RE = re.compile(
    r"^(?P<prefix>\d+)_LR-(?P<day>\d+)-(?P<month>[a-zà]+)-(?P<year>\d{4})-n(?P<num>\d+)\.html$",
    re.IGNORECASE,
)


def parse_italian_date(day: int, month: str, year: int) -> date:
    """Build a date from the Italian day-month-year tokens used in filenames."""
    month_num = ITALIAN_MONTHS.get(normalize_month_name(month))
    if not month_num:
        raise ValueError(f"Unknown Italian month name: {month!r}")
    return date(int(year), month_num, int(day))


def law_id_from_date_number(law_date: date, law_number: int) -> str:
    """Create the canonical law ID from date and regional law number."""
    return f"vda:lr:{law_date.isoformat()}:{int(law_number)}"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_source_hash(files: list[LawFile]) -> str:
    """Hash the sorted corpus inventory using filename plus file digest."""
    digest = hashlib.sha256()
    for law_file in sorted(files, key=lambda item: item.source_file):
        digest.update(law_file.source_file.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(law_file.path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_law_filename(path_or_name: str | Path) -> LawFile | None:
    """Parse source filenames that encode law date and number."""
    path = Path(path_or_name)
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    law_date = parse_italian_date(int(match.group("day")), match.group("month"), int(match.group("year")))
    law_number = int(match.group("num"))
    return LawFile(
        path=path,
        source_file=path.name,
        law_date=law_date,
        law_year=law_date.year,
        law_number=law_number,
        law_id=law_id_from_date_number(law_date, law_number),
    )


def build_corpus_registry(source_dir: str | Path) -> tuple[CorpusRegistry, dict[str, Any]]:
    """Validate the source directory and build lookup indexes for law refs."""
    root = Path(source_dir)
    if not root.exists():
        raise ValueError(f"HTML corpus directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"HTML corpus path is not a directory: {root}")

    ignored_files: list[str] = []
    invalid_html_files: list[str] = []
    law_files: list[LawFile] = []

    for path in sorted((p for p in root.iterdir() if p.is_file()), key=lambda p: p.name):
        parsed = parse_law_filename(path)
        if parsed is None:
            if path.suffix.lower() == ".html":
                invalid_html_files.append(path.name)
            else:
                ignored_files.append(path.name)
            continue
        law_files.append(
            LawFile(
                path=path,
                source_file=path.name,
                law_date=parsed.law_date,
                law_year=parsed.law_year,
                law_number=parsed.law_number,
                law_id=parsed.law_id,
            )
        )

    if invalid_html_files:
        raise ValueError(f"HTML files outside the expected filename pattern: {invalid_html_files[:10]}")
    if not law_files:
        raise ValueError(f"No valid source HTML law files found in: {root}")

    by_law_id: dict[str, LawFile] = {}
    by_date_number: dict[tuple[date, int], str] = {}
    by_year_number: dict[tuple[int, int], set[str]] = {}
    duplicate_law_ids: list[str] = []
    duplicate_date_numbers: list[str] = []

    for law_file in law_files:
        if law_file.law_id in by_law_id:
            duplicate_law_ids.append(law_file.law_id)
        by_law_id[law_file.law_id] = law_file
        date_key = (law_file.law_date, law_file.law_number)
        if date_key in by_date_number:
            duplicate_date_numbers.append(f"{law_file.law_date.isoformat()}:{law_file.law_number}")
        by_date_number[date_key] = law_file.law_id
        by_year_number.setdefault((law_file.law_year, law_file.law_number), set()).add(law_file.law_id)

    if duplicate_law_ids or duplicate_date_numbers:
        raise ValueError(
            "Duplicate law identities found: "
            f"law_ids={duplicate_law_ids[:10]}, date_numbers={duplicate_date_numbers[:10]}"
        )

    inventory = {
        "source_dir": str(root),
        "valid_html_files": len(law_files),
        "ignored_files": ignored_files,
        "invalid_html_files": invalid_html_files,
    }
    return CorpusRegistry(by_law_id, by_date_number, by_year_number), inventory

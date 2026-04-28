from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from pathlib import Path

from .utils import law_id_from_date_number, parse_italian_date

_FILENAME_RE = re.compile(
    r"^(?P<prefix>\d+)_LR-(?P<day>\d+)-(?P<month>[a-zà]+)-(?P<year>\d{4})-n(?P<num>\d+)\.html$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LawFile:
    path: Path
    source_file: str
    law_date: date
    law_year: int
    law_number: int
    law_id: str


@dataclass
class CorpusRegistry:
    by_law_id: dict[str, LawFile]
    by_date_number: dict[tuple[date, int], str]
    by_year_number: dict[tuple[int, int], set[str]]

    def resolve_law_id(self, law_date: date, law_number: int) -> str | None:
        return self.by_date_number.get((law_date, int(law_number)))

    def resolve_law_id_year(self, law_year: int, law_number: int) -> str | None:
        hits = self.by_year_number.get((int(law_year), int(law_number))) or set()
        if len(hits) == 1:
            return next(iter(hits))
        return None

    def get_path(self, law_id: str) -> Path | None:
        lf = self.by_law_id.get(law_id)
        return lf.path if lf else None


def parse_law_filename(filename: str) -> LawFile | None:
    m = _FILENAME_RE.match(filename or "")
    if not m:
        return None
    day = int(m.group("day"))
    month = m.group("month")
    year = int(m.group("year"))
    num = int(m.group("num"))
    law_date = parse_italian_date(day, month, year)
    law_id = law_id_from_date_number(law_date, num)
    return LawFile(
        path=Path(filename),
        source_file=filename,
        law_date=law_date,
        law_year=law_date.year,
        law_number=num,
        law_id=law_id,
    )


def build_corpus_registry(html_dir: Path) -> CorpusRegistry:
    if not html_dir.exists():
        raise ValueError(f"HTML directory not found: {html_dir}")
    html_files = sorted(html_dir.glob("*.html"), key=lambda p: p.name)
    if not html_files:
        raise ValueError(f"No .html files found in: {html_dir}")

    by_law_id: dict[str, LawFile] = {}
    by_date_number: dict[tuple[date, int], str] = {}
    by_year_number: dict[tuple[int, int], set[str]] = {}

    for path in html_files:
        parsed = parse_law_filename(path.name)
        if not parsed:
            continue
        lf = LawFile(
            path=path,
            source_file=path.name,
            law_date=parsed.law_date,
            law_year=parsed.law_year,
            law_number=parsed.law_number,
            law_id=parsed.law_id,
        )
        if lf.law_id in by_law_id:
            raise ValueError(f"Duplicate law_id {lf.law_id} from {path.name}")
        by_law_id[lf.law_id] = lf

        k_date = (lf.law_date, lf.law_number)
        if k_date in by_date_number:
            raise ValueError(f"Duplicate law key {k_date} from {path.name}")
        by_date_number[k_date] = lf.law_id

        k_year = (lf.law_year, lf.law_number)
        by_year_number.setdefault(k_year, set()).add(lf.law_id)

    return CorpusRegistry(
        by_law_id=by_law_id,
        by_date_number=by_date_number,
        by_year_number=by_year_number,
    )


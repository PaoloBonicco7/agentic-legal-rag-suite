from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


_CANONICAL_LAW_ID_RE = re.compile(r"\b[a-z0-9]+:lr:(?:18|19|20)\d{2}-\d{2}-\d{2}:\d+\b", re.IGNORECASE)
_CANONICAL_ARTICLE_ID_RE = re.compile(
    r"\b[a-z0-9]+:lr:(?:18|19|20)\d{2}-\d{2}-\d{2}:\d+#art:[a-z0-9._:-]+\b",
    re.IGNORECASE,
)
_REF_DATE_NUM_RE = re.compile(
    r"\b(?:legge\s+regionale|l\.?\s*r\.?)\s*"
    r"(?:(\d{1,2})\s+([a-z]+)\s+((?:18|19|20)\d{2}))?\s*,?\s*"
    r"n\.?\s*(\d+)\b",
    re.IGNORECASE,
)
_REF_NUM_YEAR_RE = re.compile(
    r"\b(?:l\.?\s*r\.?)?\s*(?:n\.?\s*)?(\d+)\s*/\s*((?:18|19|20)\d{2})\b",
    re.IGNORECASE,
)
_ARTICLE_LABEL_RE = re.compile(
    r"\bart(?:icolo)?\.?\s*([0-9]+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)\b",
    re.IGNORECASE,
)
_LAW_ID_PARTS_RE = re.compile(
    r"^(?P<jur>[a-z0-9]+):lr:(?P<date>(?:18|19|20)\d{2}-\d{2}-\d{2}):(?P<num>\d+)$",
    re.IGNORECASE,
)

_MONTH_TO_NUM = {
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


@dataclass(frozen=True)
class LawReferenceMention:
    raw: str
    canonical_law_ids: tuple[str, ...]
    law_number: int | None
    law_year: int | None
    law_date: str | None
    article_labels: tuple[str, ...]


@dataclass(frozen=True)
class LawReferenceResolution:
    law_ids: tuple[str, ...]
    article_ids: tuple[str, ...]
    unresolved_mentions: tuple[str, ...]
    mentions: tuple[LawReferenceMention, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "law_ids": list(self.law_ids),
            "article_ids": list(self.article_ids),
            "unresolved_mentions": list(self.unresolved_mentions),
            "mentions": [
                {
                    "raw": m.raw,
                    "canonical_law_ids": list(m.canonical_law_ids),
                    "law_number": m.law_number,
                    "law_year": m.law_year,
                    "law_date": m.law_date,
                    "article_labels": list(m.article_labels),
                }
                for m in self.mentions
            ],
        }


@dataclass(frozen=True)
class LawCatalog:
    law_ids: tuple[str, ...]
    by_year_number: dict[tuple[int, int], tuple[str, ...]]
    by_date_number: dict[tuple[str, int], str]
    article_ids_by_law: dict[str, tuple[str, ...]]
    article_labels_by_law: dict[str, tuple[str, ...]]

    def resolve(self, text: str) -> LawReferenceResolution:
        return resolve_law_references(text, catalog=self)

    def has_law(self, law_id: str) -> bool:
        return str(law_id or "").strip().lower() in {
            x.lower() for x in self.law_ids
        }


@dataclass(frozen=True)
class EvalCoverageReport:
    references_total: int
    references_with_any_law: int
    references_resolved: int
    coverage: float | None
    missing_references_sample: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "references_total": self.references_total,
            "references_with_any_law": self.references_with_any_law,
            "references_resolved": self.references_resolved,
            "coverage": self.coverage,
            "missing_references_sample": list(self.missing_references_sample),
        }


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _normalize_article_label(label: str) -> str:
    text = _norm(label).lower()
    if not text:
        return ""
    text = text.replace(".", "").replace(",", " ")
    text = re.sub(r"\s+", "", text)
    return text


def _parse_law_id_parts(law_id: str) -> tuple[int | None, int | None, str | None]:
    m = _LAW_ID_PARTS_RE.match(_norm(law_id).lower())
    if not m:
        return None, None, None
    date = _norm(m.group("date"))
    year = int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None
    number = int(m.group("num"))
    return year, number, date


def _parse_date(day_raw: str | None, month_raw: str | None, year_raw: str | None) -> str | None:
    if not (day_raw and month_raw and year_raw):
        return None
    day = int(day_raw)
    month = _MONTH_TO_NUM.get(_norm(month_raw).lower())
    year = int(year_raw)
    if month is None:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def build_law_catalog(dataset_dir: Path) -> LawCatalog:
    dataset_dir = dataset_dir.resolve()
    laws_path = dataset_dir / "laws.jsonl"
    articles_path = dataset_dir / "articles.jsonl"

    law_ids: list[str] = []
    by_year_number_acc: dict[tuple[int, int], set[str]] = {}
    by_date_number: dict[tuple[str, int], str] = {}

    if laws_path.exists():
        with laws_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                law_id = _norm(row.get("law_id"))
                if not law_id:
                    continue
                law_ids.append(law_id)
                year, number, date = _parse_law_id_parts(law_id)
                if year is not None and number is not None:
                    by_year_number_acc.setdefault((year, number), set()).add(law_id)
                if date is not None and number is not None:
                    by_date_number[(date, number)] = law_id

    article_ids_by_law_acc: dict[str, set[str]] = {}
    article_labels_by_law_acc: dict[str, set[str]] = {}
    if articles_path.exists():
        with articles_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                law_id = _norm(row.get("law_id"))
                article_id = _norm(row.get("article_id"))
                if not law_id or not article_id:
                    continue
                article_ids_by_law_acc.setdefault(law_id, set()).add(article_id)
                label = _norm(row.get("article_label_norm"))
                if not label and "#art:" in article_id:
                    label = article_id.split("#art:", 1)[1]
                if label:
                    article_labels_by_law_acc.setdefault(law_id, set()).add(
                        _normalize_article_label(label)
                    )

    return LawCatalog(
        law_ids=tuple(sorted(set(law_ids))),
        by_year_number={
            key: tuple(sorted(values)) for key, values in by_year_number_acc.items()
        },
        by_date_number=dict(by_date_number),
        article_ids_by_law={
            key: tuple(sorted(values)) for key, values in article_ids_by_law_acc.items()
        },
        article_labels_by_law={
            key: tuple(sorted(values)) for key, values in article_labels_by_law_acc.items()
        },
    )


def extract_law_reference_mentions(text: str) -> tuple[LawReferenceMention, ...]:
    raw = _norm(text)
    if not raw:
        return tuple()

    mentions: list[LawReferenceMention] = []
    segments = [
        seg.strip()
        for seg in re.split(r"[|\n]+", raw)
        if seg is not None and seg.strip()
    ]
    if not segments:
        segments = [raw]

    for seg in segments:
        canonical_law_ids = {
            _norm(m.group(0)).lower()
            for m in _CANONICAL_LAW_ID_RE.finditer(seg)
            if _norm(m.group(0))
        }
        for article_match in _CANONICAL_ARTICLE_ID_RE.finditer(seg):
            art_id = _norm(article_match.group(0)).lower()
            if "#art:" in art_id:
                canonical_law_ids.add(art_id.split("#art:", 1)[0])

        law_number: int | None = None
        law_year: int | None = None
        law_date: str | None = None

        m_date_num = _REF_DATE_NUM_RE.search(seg)
        if m_date_num:
            law_number = int(m_date_num.group(4))
            law_year = int(m_date_num.group(3)) if m_date_num.group(3) else None
            law_date = _parse_date(
                m_date_num.group(1),
                m_date_num.group(2),
                m_date_num.group(3),
            )

        if law_number is None or law_year is None:
            m_num_year = _REF_NUM_YEAR_RE.search(seg)
            if m_num_year:
                law_number = int(m_num_year.group(1))
                law_year = int(m_num_year.group(2))

        article_labels = {
            _normalize_article_label(m.group(1))
            for m in _ARTICLE_LABEL_RE.finditer(seg)
            if _normalize_article_label(m.group(1))
        }

        if not canonical_law_ids and law_number is None and law_year is None:
            continue
        mentions.append(
            LawReferenceMention(
                raw=seg,
                canonical_law_ids=tuple(sorted(canonical_law_ids)),
                law_number=law_number,
                law_year=law_year,
                law_date=law_date,
                article_labels=tuple(sorted(article_labels)),
            )
        )

    return tuple(mentions)


def resolve_law_references(text: str, *, catalog: LawCatalog) -> LawReferenceResolution:
    mentions = extract_law_reference_mentions(text)
    if not mentions:
        return LawReferenceResolution(
            law_ids=tuple(),
            article_ids=tuple(),
            unresolved_mentions=tuple(),
            mentions=tuple(),
        )

    out_laws: set[str] = set()
    out_articles: set[str] = set()
    unresolved: list[str] = []

    catalog_laws = {x.lower(): x for x in catalog.law_ids}

    for mention in mentions:
        resolved_law_ids: set[str] = set()

        for law_id in mention.canonical_law_ids:
            cur = catalog_laws.get(law_id.lower())
            if cur:
                resolved_law_ids.add(cur)

        if mention.law_number is not None:
            if mention.law_date is not None:
                candidate = catalog.by_date_number.get((mention.law_date, mention.law_number))
                if candidate:
                    resolved_law_ids.add(candidate)
            if mention.law_year is not None:
                candidates = catalog.by_year_number.get((mention.law_year, mention.law_number), tuple())
                resolved_law_ids.update(candidates)

        if not resolved_law_ids:
            unresolved.append(mention.raw)
            continue

        out_laws.update(resolved_law_ids)

        if mention.article_labels:
            for law_id in resolved_law_ids:
                known_labels = set(catalog.article_labels_by_law.get(law_id, tuple()))
                for lbl in mention.article_labels:
                    if lbl in known_labels:
                        out_articles.add(f"{law_id}#art:{lbl}")

    return LawReferenceResolution(
        law_ids=tuple(sorted(out_laws)),
        article_ids=tuple(sorted(out_articles)),
        unresolved_mentions=tuple(unresolved),
        mentions=mentions,
    )


def load_eval_reference_texts(csv_paths: Sequence[Path]) -> list[str]:
    out: list[str] = []
    for path in csv_paths:
        if path is None:
            continue
        cur = Path(path).resolve()
        if not cur.exists():
            continue
        with cur.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref = _norm(row.get("Riferimento legge per la risposta"))
                if ref:
                    out.append(ref)
    return out


def compute_eval_reference_coverage(
    *,
    catalog: LawCatalog,
    references: Sequence[str],
    missing_sample_size: int = 20,
) -> EvalCoverageReport:
    refs = list(references)
    total = len(refs)
    if total <= 0:
        return EvalCoverageReport(
            references_total=0,
            references_with_any_law=0,
            references_resolved=0,
            coverage=None,
            missing_references_sample=tuple(),
        )

    with_any_law = 0
    resolved = 0
    missing: list[str] = []
    for ref in refs:
        res = resolve_law_references(ref, catalog=catalog)
        if res.mentions:
            with_any_law += 1
        if res.law_ids:
            resolved += 1
        elif len(missing) < max(1, int(missing_sample_size)):
            missing.append(_norm(ref))

    denom = with_any_law if with_any_law > 0 else total
    coverage = (resolved / denom) if denom > 0 else None
    return EvalCoverageReport(
        references_total=total,
        references_with_any_law=with_any_law,
        references_resolved=resolved,
        coverage=coverage,
        missing_references_sample=tuple(missing),
    )


__all__ = [
    "LawReferenceMention",
    "LawReferenceResolution",
    "LawCatalog",
    "EvalCoverageReport",
    "build_law_catalog",
    "extract_law_reference_mentions",
    "resolve_law_references",
    "load_eval_reference_texts",
    "compute_eval_reference_coverage",
]

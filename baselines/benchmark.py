from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import csv
import re
from pathlib import Path

from laws_ingestion.core.registry import CorpusRegistry
from laws_ingestion.core.utils import normalize_article_label, parse_italian_date


@dataclass(frozen=True)
class ParsedReference:
    law_date: date
    law_number: int
    article_label_norm: str
    raw: str


@dataclass(frozen=True)
class Option:
    label: str
    text: str


@dataclass(frozen=True)
class GoldTarget:
    law_id: str
    article_label_norm: str
    raw_reference: str


@dataclass(frozen=True)
class Question:
    qid: int
    level: str
    stem: str
    options: tuple[Option, ...]
    correct_answer_label: str
    gold_targets: tuple[GoldTarget, ...]


_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-F])\)\s+(.*)$")


def parse_domanda_field(domanda: str) -> tuple[str, tuple[Option, ...]]:
    domanda = (domanda or "").strip("\n")
    matches = list(_OPTION_LINE_RE.finditer(domanda))
    if not matches:
        raise ValueError("No options found in `Domanda` field")

    first_opt_start = matches[0].start()
    stem = domanda[:first_opt_start].strip()

    options: list[Option] = []
    for m in matches:
        label = m.group(1).strip()
        text = m.group(2).strip()
        options.append(Option(label=label, text=text))

    labels = [o.label for o in options]
    if labels != ["A", "B", "C", "D", "E", "F"]:
        raise ValueError(f"Unexpected option labels/order: {labels!r}")

    return stem, tuple(options)


_REF_LINE_RE = re.compile(
    r"Legge\s+regionale\s+(?P<day>\d{1,2})(?:\s*°)?\s+(?P<month>[a-zà]+)\s+(?P<year>\d{4}),\s*n\.\s*(?P<num>\d+)\s*-\s*Art\.?\s*(?P<art>.+)$",
    flags=re.IGNORECASE,
)


def parse_reference_line(line: str) -> ParsedReference:
    line = (line or "").strip()
    m = _REF_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Unparseable reference line: {line!r}")

    day = int(m.group("day"))
    month = m.group("month")
    year = int(m.group("year"))
    law_number = int(m.group("num"))
    law_date = parse_italian_date(day, month, year)
    article_norm = normalize_article_label(m.group("art"))

    return ParsedReference(
        law_date=law_date,
        law_number=law_number,
        article_label_norm=article_norm,
        raw=line,
    )


def parse_references_field(ref: str) -> tuple[ParsedReference, ...]:
    parts = [p.strip() for p in (ref or "").splitlines() if p.strip()]
    return tuple(parse_reference_line(p) for p in parts)


def iter_complete_questions(csv_path: Path, corpus_registry: CorpusRegistry) -> tuple[Question, ...]:
    questions: list[Question] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            correct = (row.get("Risposta corretta") or "").strip()
            if not correct:
                continue

            qid = int((row.get("#") or "").strip())
            level = (row.get("Livello") or "").strip()

            stem, options = parse_domanda_field(row.get("Domanda") or "")
            refs = parse_references_field(row.get("Riferimento legge per la risposta") or "")

            gold_targets: list[GoldTarget] = []
            for ref in refs:
                law_id = corpus_registry.resolve_law_id(ref.law_date, ref.law_number)
                if not law_id:
                    raise ValueError(f"Reference does not map to any file in corpus: {ref.raw!r}")
                gold_targets.append(
                    GoldTarget(
                        law_id=law_id,
                        article_label_norm=ref.article_label_norm,
                        raw_reference=ref.raw,
                    )
                )

            questions.append(
                Question(
                    qid=qid,
                    level=level,
                    stem=stem,
                    options=options,
                    correct_answer_label=correct,
                    gold_targets=tuple(gold_targets),
                )
            )

    return tuple(questions)


def benchmark_summary(questions: tuple[Question, ...]) -> dict[str, object]:
    levels: dict[str, int] = {}
    laws: set[str] = set()
    law_articles: set[tuple[str, str]] = set()
    ref_parts = 0
    for q in questions:
        levels[q.level] = levels.get(q.level, 0) + 1
        for gt in q.gold_targets:
            ref_parts += 1
            laws.add(gt.law_id)
            law_articles.add((gt.law_id, gt.article_label_norm))

    return {
        "questions": len(questions),
        "levels": dict(sorted(levels.items())),
        "total_gold_refs": ref_parts,
        "unique_laws_referenced": len(laws),
        "unique_law_article_referenced": len(law_articles),
    }


_ARTICLE_ANCHOR_RE = re.compile(r'<a\s+name="articolo_[^"]+">\s*(Art\.[^<]*)<', re.IGNORECASE)


def extract_article_labels_from_html(html_path: Path) -> set[str]:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    labels: set[str] = set()
    for m in _ARTICLE_ANCHOR_RE.finditer(html):
        labels.add(normalize_article_label(m.group(1)))
    return labels


def validate_gold_targets_exist(
    questions: tuple[Question, ...], corpus_registry: CorpusRegistry
) -> tuple[int, list[str]]:
    cache: dict[str, set[str]] = {}
    missing: list[str] = []
    for q in questions:
        for gt in q.gold_targets:
            html_path = corpus_registry.get_path(gt.law_id)
            if not html_path:
                missing.append(f"Q{q.qid}: missing law_id {gt.law_id}")
                continue
            if gt.law_id not in cache:
                cache[gt.law_id] = extract_article_labels_from_html(html_path)
            if gt.article_label_norm not in cache[gt.law_id]:
                missing.append(f"Q{q.qid}: missing article {gt.article_label_norm} in {gt.law_id} ({html_path.name})")
    return len(missing), missing

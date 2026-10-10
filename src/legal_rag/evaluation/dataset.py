"""Validate original questions and keep approved revisions separate."""

import csv
import json
import re
from pathlib import Path

from legal_rag.artifacts import digest, file_hash, read_jsonl, write_jsonl

from .models import Benchmark, Question, Reference, Review

MONTHS = {name: index for index, name in enumerate(
    ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"), 1
)}
LAW_RE = re.compile(r"legge\s+regionale\s+(\d{1,2})\s+(\w+)\s+(\d{4}),?\s+n\.?\s*(\d+)", re.I)
ARTICLE_RE = re.compile(r"\bart\.?\s*(\d+)\s*(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?", re.I)
OPTION_RE = re.compile(r"^\s*([A-F])\)\s*(.*)$")


def normalize(text: str) -> str:
    return " ".join(text.split())


def parse_references(values: list[str]) -> list[Reference]:
    references = []
    for value in values:
        for raw in re.split(r"[|\n]", value):
            text = normalize(raw)
            if not text:
                continue
            law, article = LAW_RE.search(text), ARTICLE_RE.search(text)
            document_id = None
            if law and law[2].lower() in MONTHS:
                document_id = f"vda:lr:{law[3]}-{MONTHS[law[2].lower()]:02d}-{int(law[1]):02d}:{int(law[4])}"
            label = f"{int(article[1])}{(article[2] or '').lower()}" if article else None
            references.append(Reference(text=text, document_id=document_id, article_label=label))
    return references


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{normalize(key or ""): value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _mcq(raw: str) -> tuple[str, dict[str, str]]:
    stem, options, label = [], {}, None
    for line in raw.splitlines():
        match = OPTION_RE.match(line)
        if match:
            label = match[1]
            if label in options:
                raise ValueError(f"Duplicate option {label}")
            options[label] = match[2]
        elif label:
            options[label] += " " + line
        else:
            stem.append(line)
    options = {key: normalize(value) for key, value in options.items()}
    if tuple(options) != tuple("ABCDEF") or not all(options.values()):
        raise ValueError("Every MCQ must contain six nonempty ordered A-F options")
    return normalize(" ".join(stem)), options


def load_benchmark(
    source_dir: Path | str = Path("data/evaluation"), *, expected_count: int = 100,
    view: str = "legacy", review_path: Path | str | None = None,
) -> Benchmark:
    source_dir = Path(source_dir)
    sources = {name: source_dir / name for name in ("questions.csv", "questions_no_hint.csv")}
    mcq = [row for row in _csv_rows(sources["questions.csv"]) if any(normalize(row.get(key, "")) for key in ("Domanda", "Livello", "Risposta corretta"))]
    no_hint = _csv_rows(sources["questions_no_hint.csv"])
    if len(mcq) != expected_count or len(no_hint) != expected_count:
        raise ValueError(f"Expected {expected_count} aligned intents; got {len(mcq)} MCQ and {len(no_hint)} no-hint")
    supplemental = source_dir / "vigency_reference_review.csv"
    annotations = {row["qid"]: row for row in _csv_rows(supplemental)} if supplemental.exists() else {}
    questions = []
    for position, (choice, opened) in enumerate(zip(mcq, no_hint), 1):
        qid = f"eval-{position:04d}"
        stem, options = _mcq(choice["Domanda"])
        label = normalize(choice["Risposta corretta"]).upper()
        if label not in options:
            raise ValueError(f"{qid}: invalid correct label")
        answer = normalize(opened.get("Risposta corretta", ""))
        level = normalize(opened.get("Livello", ""))
        if stem != normalize(opened.get("Domanda", "")) or options[label] != answer or normalize(choice.get("Livello", "")) != level:
            raise ValueError(f"{qid}: MCQ/no-hint intent, answer or level mismatch")
        refs = parse_references([opened.get("Riferimento legge per la risposta", "")])
        if refs != parse_references([choice.get("Riferimento legge per la risposta", "")]):
            raise ValueError(f"{qid}: MCQ/no-hint references mismatch")
        if not stem or not answer or not level or not refs:
            raise ValueError(f"{qid}: empty required question field")
        flags = ["unparsed_reference"] if any(not ref.document_id or not ref.article_label for ref in refs) else []
        annotation = annotations.get(qid, {})
        if annotation.get("answer_support_relation") == "supported_elsewhere":
            flags.append("documented_reference_mismatch")
        temporal = annotation.get("temporal_scope_flag", "")
        if temporal and temporal != "none":
            flags.append(f"documented_temporal_scope:{temporal}")
        questions.append(Question(qid=qid, question=stem, level=level, correct_answer=answer, options=options,
                                  correct_label=label, expected_references=refs, flags=flags))
    hashes = {name: file_hash(path) for name, path in sources.items()}
    if supplemental.exists():
        hashes[supplemental.name] = file_hash(supplemental)
    reviews = read_jsonl(Path(review_path), Review) if review_path else []
    if review_path:
        hashes["review"] = file_hash(Path(review_path))
    known = {question.qid for question in questions}
    if len({review.qid for review in reviews}) != len(reviews) or any(review.qid not in known for review in reviews):
        raise ValueError("Review QIDs must be unique and refer to original questions")
    if view == "current_approved":
        approved = {review.qid: review for review in reviews if review.human_approved and review.status in {"current_supported", "current_revised"}}
        revised = []
        for question in questions:
            if question.qid not in approved:
                continue
            review = approved[question.qid]
            values = question.to_json_record()
            for field in ("question", "correct_answer"):
                if getattr(review, field) is not None:
                    values[field] = normalize(getattr(review, field))
            if review.expected_references is not None:
                values["expected_references"] = [ref.to_json_record() for ref in parse_references(review.expected_references)]
            labels = [label for label, text in question.options.items() if text == values["correct_answer"]]
            values["correct_label"] = labels[0] if len(labels) == 1 else None
            values["flags"] = [*question.flags, "human_approved_revision"]
            if values["correct_label"] is None:
                values["flags"].append("revised_gold_not_in_legacy_options")
            revised.append(Question.model_validate(values))
        questions = revised
        if not questions:
            raise ValueError("The approved-current view contains no approved supported questions")
    elif view != "legacy":
        raise ValueError("view must be legacy or current_approved")
    fingerprint = digest(json.dumps({"view": view, "sources": hashes, "questions": [q.to_json_record() for q in questions]}, sort_keys=True, ensure_ascii=False))
    return Benchmark(view=view, questions=questions, source_hashes=hashes, benchmark_hash=fingerprint,
                     original_count=expected_count, reviews=reviews,
                     warnings=["The legacy questions were previously explored; results do not establish unseen-query performance."])


def write_review_proposals(benchmark: Benchmark, updates: list[dict], output_path: Path | str) -> Path:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    known = {question.qid for question in benchmark.questions}
    reviews = [Review.model_validate(update) for update in updates]
    if len({review.qid for review in reviews}) != len(reviews) or any(review.qid not in known for review in reviews):
        raise ValueError("Review QIDs must be unique and known")
    write_jsonl(path, reviews)
    return path

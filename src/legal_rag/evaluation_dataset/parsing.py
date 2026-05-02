"""CSV parsing and normalization helpers for evaluation questions."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from legal_rag.laws_preprocessing import normalize_ws

from .models import EXPECTED_OPTION_LABELS, mcq_question_record, no_hint_question_record

OPTION_LINE_RE = re.compile(r"^\s*([A-F])\)\s*(.*)$")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries with normalized header names."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation source file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Evaluation source path is not a file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {normalize_ws(str(key or "")): str(value or "") for key, value in row.items()}
            for row in reader
        ]


def is_valid_source_row(row: dict[str, str]) -> bool:
    """Return whether a source row contains benchmark content."""
    required = ("Domanda", "Livello", "Risposta corretta")
    return any(normalize_ws(row.get(field, "")) for field in required)


def normalize_references(raw: str) -> list[str]:
    """Split multiline legal references while preserving human-readable text."""
    return [normalize_ws(line) for line in str(raw or "").splitlines() if normalize_ws(line)]


def parse_mcq_question(raw_question: str) -> tuple[str, dict[str, str]]:
    """Extract the question stem and ordered A-F options from an MCQ field."""
    lines = str(raw_question or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stem_lines: list[str] = []
    option_parts: dict[str, list[str]] = {}
    current_label: str | None = None
    saw_option = False

    for line in lines:
        match = OPTION_LINE_RE.match(line)
        if match:
            saw_option = True
            current_label = match.group(1).upper()
            if current_label in option_parts:
                raise ValueError(f"Duplicate MCQ option label: {current_label}")
            option_parts[current_label] = [match.group(2)]
            continue
        if not saw_option:
            stem_lines.append(line)
            continue
        if current_label and normalize_ws(line):
            option_parts[current_label].append(line)

    labels = tuple(option_parts.keys())
    if labels != EXPECTED_OPTION_LABELS:
        raise ValueError(f"Expected MCQ options {EXPECTED_OPTION_LABELS!r}, found {labels!r}")

    stem = normalize_ws(" ".join(stem_lines))
    options = {label: normalize_ws(" ".join(option_parts[label])) for label in EXPECTED_OPTION_LABELS}
    if not stem:
        raise ValueError("MCQ question stem is empty")
    if any(not text for text in options.values()):
        raise ValueError("One or more MCQ options are empty")
    return stem, options


def build_mcq_record(row: dict[str, str], position: int) -> dict[str, Any]:
    """Build a validated normalized MCQ record from one source CSV row."""
    qid = f"eval-{position:04d}"
    stem, options = parse_mcq_question(row.get("Domanda", ""))
    correct_label = normalize_ws(row.get("Risposta corretta", "")).upper()
    references = normalize_references(row.get("Riferimento legge per la risposta", ""))
    if correct_label not in options:
        raise ValueError(f"{qid}: correct label {correct_label!r} is not present in options")
    return mcq_question_record(
        {
            "qid": qid,
            "source_position": position,
            "level": normalize_ws(row.get("Livello", "")),
            "question_stem": stem,
            "options": options,
            "correct_label": correct_label,
            "correct_answer": options[correct_label],
            "expected_references": references,
        }
    )


def build_no_hint_record(row: dict[str, str], position: int, linked_mcq_qid: str) -> dict[str, Any]:
    """Build a validated normalized no-hint record from one source CSV row."""
    return no_hint_question_record(
        {
            "qid": f"eval-{position:04d}",
            "source_position": position,
            "level": normalize_ws(row.get("Livello", "")),
            "question": normalize_ws(row.get("Domanda", "")),
            "correct_answer": normalize_ws(row.get("Risposta corretta", "")),
            "expected_references": normalize_references(row.get("Riferimento legge per la risposta", "")),
            "linked_mcq_qid": linked_mcq_qid,
        }
    )


def validate_alignment(mcq: dict[str, Any], no_hint: dict[str, Any]) -> None:
    """Validate that paired MCQ and no-hint records describe the same question."""
    qid = str(mcq.get("qid") or "")
    if mcq["qid"] != no_hint["linked_mcq_qid"]:
        raise ValueError(f"{qid}: no-hint linked_mcq_qid does not match MCQ qid")
    if mcq["qid"] != no_hint["qid"]:
        raise ValueError(f"{qid}: no-hint qid does not match MCQ qid")
    if normalize_ws(mcq["level"]) != normalize_ws(no_hint["level"]):
        raise ValueError(f"{qid}: MCQ and no-hint levels do not match")
    if normalize_ws(mcq["question_stem"]) != normalize_ws(no_hint["question"]):
        raise ValueError(f"{qid}: MCQ stem and no-hint question do not match")
    if normalize_ws(mcq["correct_answer"]) != normalize_ws(no_hint["correct_answer"]):
        raise ValueError(f"{qid}: MCQ correct answer and no-hint answer do not match")

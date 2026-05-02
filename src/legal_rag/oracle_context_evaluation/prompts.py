"""Prompt builders for oracle-context evaluation runs."""

from __future__ import annotations

import json
from typing import Any

from .models import JudgeOutput, McqAnswerOutput, NoHintAnswerOutput


def schema_dict(model_cls: type[McqAnswerOutput] | type[NoHintAnswerOutput] | type[JudgeOutput]) -> dict[str, Any]:
    """Return a JSON schema payload accepted by Ollama-compatible structured chat."""
    return model_cls.model_json_schema()


def format_options(options: dict[str, str]) -> str:
    """Render MCQ options in stable A-F order."""
    return "\n".join(f"{label}) {options[label]}" for label in ("A", "B", "C", "D", "E", "F"))


def build_mcq_prompt(record: dict[str, Any], *, context_text: str | None = None) -> str:
    """Build an MCQ answer prompt with optional oracle legal context."""
    context_block = ""
    if context_text:
        context_block = (
            "Use only the following source-of-truth legal context when it is relevant.\n\n"
            f"{context_text}\n\n"
        )
    return (
        "You answer Italian legal multiple-choice questions.\n"
        "Choose exactly one label among A, B, C, D, E, F.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(McqAnswerOutput), ensure_ascii=False)}\n\n"
        f"{context_block}"
        "Question:\n"
        f"{record['question_stem']}\n\n"
        "Options:\n"
        f"{format_options(record['options'])}"
    )


def build_no_hint_prompt(record: dict[str, Any], *, context_text: str | None = None) -> str:
    """Build an open-answer prompt with optional oracle legal context."""
    context_block = ""
    if context_text:
        context_block = (
            "Use only the following source-of-truth legal context when it is relevant.\n\n"
            f"{context_text}\n\n"
        )
    return (
        "You answer Italian legal questions precisely and concisely.\n"
        "Do not mention multiple-choice labels or options.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(NoHintAnswerOutput), ensure_ascii=False)}\n\n"
        f"{context_block}"
        "Question:\n"
        f"{record['question']}"
    )


def build_judge_prompt(record: dict[str, Any], predicted_answer: str) -> str:
    """Build the no-hint judge prompt without exposing MCQ alternatives."""
    candidate = predicted_answer.strip() if predicted_answer.strip() else "[EMPTY]"
    return (
        "You are an impartial semantic judge for Italian legal QA.\n"
        "Score the model answer against the official correct answer.\n\n"
        "Rubric:\n"
        "- score=2: correct or semantically equivalent.\n"
        "- score=1: partially correct, incomplete, and not contradictory.\n"
        "- score=0: wrong, contradictory, empty, ambiguous, or not evaluable.\n\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema_dict(JudgeOutput), ensure_ascii=False)}\n\n"
        "Question:\n"
        f"{record['question']}\n\n"
        "Official correct answer:\n"
        f"{record['correct_answer']}\n\n"
        "Model answer to judge:\n"
        f"{candidate}"
    )

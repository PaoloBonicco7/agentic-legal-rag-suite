from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .schemas import JudgeResult, McqAnswer, NoHintAnswer, schema_to_json_dict


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_line_breaks(text: str) -> str:
    return (
        (text or "")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _validate_http_url(url: str, *, field_name: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"{field_name} must start with http:// or https://. Got: {url!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"{field_name} is missing host. Got: {url!r}")
    return parsed.geturl().rstrip("/")


def resolve_ollama_chat_url(base_url: str, explicit_url: str | None = None) -> str:
    """Resolve benchmark chat endpoint from base URL or explicit override.

    Rules:
    1) if explicit_url is set -> use it as-is (validated)
    2) otherwise normalize base_url by removing trailing `/api` if present
    3) append `/ollama/api/chat`
    """

    if explicit_url is not None and explicit_url.strip():
        return _validate_http_url(explicit_url, field_name="explicit_url")

    normalized_base = _validate_http_url(base_url, field_name="base_url")
    if normalized_base.endswith("/api"):
        normalized_base = normalized_base[:-4]
    return f"{normalized_base}/ollama/api/chat"


def parse_structured_content(response_json: dict[str, Any]) -> dict[str, Any]:
    raw_content = response_json.get("message", {}).get("content")
    if raw_content is None:
        raise KeyError("response_json['message']['content'] mancante")

    if isinstance(raw_content, str):
        obj = json.loads(raw_content)
    elif isinstance(raw_content, dict):
        obj = raw_content
    else:
        raise TypeError(f"Tipo inatteso per message.content: {type(raw_content).__name__}")

    if not isinstance(obj, dict):
        raise TypeError(f"Il payload structured deve essere un oggetto, ricevuto: {type(obj).__name__}")
    return obj


def post_structured_chat(
    *,
    api_url: str,
    headers: dict[str, str],
    payload_schema: dict[str, Any],
    prompt: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": payload_schema,
        "options": {"temperature": 0},
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code} su {api_url}. Body: {response.text}")
    response_json = response.json()
    structured = parse_structured_content(response_json)
    return {"structured": structured, "response_json": response_json}


def load_valid_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if (row.get("Risposta corretta") or "").strip()]


def extract_mcq_options(mcq_question_text: str) -> dict[str, str]:
    normalized_text = normalize_line_breaks(mcq_question_text or "")
    matches = re.findall(r"(?m)^\s*([A-F])\)\s+(.*)$", normalized_text)
    option_map = {label.strip().upper(): text.strip() for label, text in matches}
    expected = {"A", "B", "C", "D", "E", "F"}
    if set(option_map.keys()) != expected:
        raise ValueError(f"Opzioni MCQ non valide. Trovate: {sorted(option_map.keys())}")
    return option_map


def align_record(pos: int, no_hint_rows: list[dict[str, str]], mcq_rows: list[dict[str, str]]) -> dict[str, Any]:
    if pos < 0 or pos >= len(no_hint_rows) or pos >= len(mcq_rows):
        raise IndexError(f"pos fuori range: {pos}")

    row_no_hint = no_hint_rows[pos]
    row_mcq = mcq_rows[pos]

    question_no_hint = (row_no_hint.get("Domanda") or "").strip()
    question_mcq_full = normalize_line_breaks((row_mcq.get("Domanda") or "")).strip()

    stem_match = re.search(r"(?m)^\s*A\)\s+", question_mcq_full)
    question_mcq_stem = question_mcq_full[: stem_match.start()].strip() if stem_match else question_mcq_full

    if normalize_space(question_no_hint) != normalize_space(question_mcq_stem):
        raise RuntimeError(
            "Misallineamento no-hint/MCQ sullo stem. "
            f"pos={pos} | no_hint={question_no_hint!r} | mcq_stem={question_mcq_stem!r}"
        )

    option_map = extract_mcq_options(question_mcq_full)
    correct_label = (row_mcq.get("Risposta corretta") or "").strip().upper()[:1]
    if correct_label not in option_map:
        raise RuntimeError(f"Label corretta non presente nelle opzioni: {correct_label!r}")

    return {
        "pos": pos,
        "qid": (row_mcq.get("#") or str(pos + 1)).strip(),
        "level": (row_no_hint.get("Livello") or "UNKNOWN").strip() or "UNKNOWN",
        "question_no_hint": question_no_hint,
        "question_mcq_full": question_mcq_full,
        "ground_truth_text_no_hint": (row_no_hint.get("Risposta corretta") or "").strip(),
        "ground_truth_label_mcq": correct_label,
        "correct_option_text": option_map[correct_label],
        "ground_truth_reference_law": (
            (row_no_hint.get("Riferimento legge per la risposta") or "").strip()
            or (row_mcq.get("Riferimento legge per la risposta") or "").strip()
            or None
        ),
        "option_map": option_map,
    }


def build_mcq_prompt(record: dict[str, Any]) -> str:
    schema = schema_to_json_dict(McqAnswer)
    return (
        "Sei un assistente che risponde a domande a scelta multipla. "
        "Scegli una sola opzione tra A, B, C, D, E, F.\\n\\n"
        "Rispondi solo in JSON valido conforme a questo schema: "
        f"{json.dumps(schema, ensure_ascii=False)}\\n\\n"
        "Domanda con alternative:\\n"
        f"{record['question_mcq_full']}"
    )


def build_no_hint_prompt(record: dict[str, Any]) -> str:
    schema = schema_to_json_dict(NoHintAnswer)
    return (
        "Sei un assistente che risponde a domande normative. "
        "Fornisci la risposta testuale piu precisa e concisa, senza lettere A/B/C.\\n\\n"
        "Rispondi solo in JSON valido conforme a questo schema: "
        f"{json.dumps(schema, ensure_ascii=False)}\\n\\n"
        "Domanda:\\n"
        f"{record['question_no_hint']}"
    )


def build_judge_prompt(record: dict[str, Any], step1_answer_text: str) -> str:
    schema = schema_to_json_dict(JudgeResult)
    candidate_answer = step1_answer_text if step1_answer_text else "[VUOTA]"
    return (
        "Sei un valutatore imparziale della correttezza semantica di risposte a domande a scelta multipla.\\n"
        "Valuta se la risposta del modello e' equivalente alla soluzione ufficiale.\\n\\n"
        "Regole di scoring:\\n"
        "- score=1 solo se la risposta e' semanticamente equivalente all'opzione corretta (anche parafrasi).\\n"
        "- score=0 se e' errata, ambigua, troppo incompleta, o coerente con opzione diversa.\\n"
        "- Se la risposta e' vuota/non interpretabile, usa score=0 e matched_option_label=NONE.\\n\\n"
        "Domanda con alternative:\\n"
        f"{record['question_mcq_full']}\\n\\n"
        "Soluzione ufficiale:\\n"
        f"- label: {record['ground_truth_label_mcq']}\\n"
        f"- testo: {record['correct_option_text']}\\n\\n"
        "Risposta del modello da valutare:\\n"
        f"{candidate_answer}\\n\\n"
        "Rispondi solo in JSON valido conforme a questo schema: "
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def validate_mcq_output(obj: dict[str, Any]) -> McqAnswer:
    return McqAnswer.model_validate(obj)


def validate_no_hint_output(obj: dict[str, Any]) -> NoHintAnswer:
    return NoHintAnswer.model_validate(obj)


def validate_judge_output(obj: dict[str, Any]) -> JudgeResult:
    return JudgeResult.model_validate(obj)


def aggregate_by_level(results: list[dict[str, Any]], score_key: str) -> dict[str, dict[str, Any]]:
    by_level: dict[str, dict[str, Any]] = {}

    for row in results:
        level = row.get("level") or "UNKNOWN"
        stats = by_level.setdefault(level, {"n": 0, "judged": 0, "correct": 0, "wrong": 0, "errors": 0})
        stats["n"] += 1

        if row.get("error"):
            stats["errors"] += 1

        score = row.get(score_key)
        if score in (0, 1):
            stats["judged"] += 1
            stats["correct"] += int(score)

    for stats in by_level.values():
        stats["wrong"] = max(stats["judged"] - stats["correct"], 0)
        stats["accuracy"] = (stats["correct"] / stats["judged"]) if stats["judged"] else None

    return dict(sorted(by_level.items(), key=lambda kv: level_sort_key(kv[0])))


def build_dataset_summary(name: str, results: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    valid_scores = [int(r[score_key]) for r in results if r.get(score_key) in (0, 1)]
    score_sum = int(sum(valid_scores))
    judged = len(valid_scores)

    return {
        "dataset": name,
        "processed": len(results),
        "judged": judged,
        "score_sum": score_sum,
        "accuracy": (score_sum / judged) if judged else None,
        "errors": sum(1 for r in results if r.get("error")),
        "by_level": aggregate_by_level(results, score_key=score_key),
    }


def is_effective_answer(text: str, *, min_chars: int = 8) -> bool:
    normalized = normalize_space(text)
    if not normalized:
        return False
    if len(normalized) < max(1, int(min_chars)):
        return False
    invalid_markers = {"[vuota]", "n/a", "nessuna risposta"}
    if normalized.lower() in invalid_markers:
        return False
    return True


def categorize_row_error(
    row: dict[str, Any],
    *,
    score_key: str,
    error_key: str = "error",
) -> str | None:
    raw_error = normalize_space(str(row.get(error_key) or ""))
    if raw_error:
        low = raw_error.lower()
        if "rag_error" in low:
            return "rag_error"
        if "judge" in low:
            return "judge_structured_error"
        if "mcq" in low:
            return "mcq_structured_error"
        if "http " in low:
            return "transport_error"
        if "timeout" in low:
            return "timeout_error"
        return "technical_error"

    status = normalize_space(str(row.get("status") or "")).lower()
    if status == "answer_empty_or_invalid":
        return "technical_generation_empty_answer"

    if "predicted_answer" in row:
        answer_text = str(row.get("predicted_answer") or "")
        if not is_effective_answer(answer_text):
            if row.get(score_key) not in (0, 1):
                return "technical_generation_empty_answer"

    if row.get(score_key) not in (0, 1):
        return "not_judged_unclassified"
    return None


def _compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = (max(0.0, min(100.0, percentile)) / 100.0) * (len(values) - 1)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return (values[low] * (1.0 - weight)) + (values[high] * weight)


def _timing_summary(
    results: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for field in fields:
        numeric_values: list[float] = []
        for row in results:
            raw = row.get(field)
            if raw is None:
                continue
            try:
                numeric_values.append(float(raw))
            except (TypeError, ValueError):
                continue
        numeric_values.sort()
        if not numeric_values:
            out[field] = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p90": None,
                "min": None,
                "max": None,
            }
            continue
        out[field] = {
            "count": len(numeric_values),
            "mean": sum(numeric_values) / len(numeric_values),
            "p50": _compute_percentile(numeric_values, 50.0),
            "p90": _compute_percentile(numeric_values, 90.0),
            "min": numeric_values[0],
            "max": numeric_values[-1],
        }
    return out


def build_extended_summary(
    name: str,
    results: list[dict[str, Any]],
    *,
    score_key: str,
    error_key: str = "error",
    timing_fields: tuple[str, ...] = (
        "t_retrieval_context_s",
        "t_task_llm_s",
        "t_judge_s",
        "t_total_s",
    ),
) -> dict[str, Any]:
    base = build_dataset_summary(name, results, score_key=score_key)
    processed = int(base["processed"])
    judged = int(base["judged"])
    score_sum = int(base["score_sum"])

    empty_answer_count = sum(
        1
        for row in results
        if "predicted_answer" in row and not is_effective_answer(str(row.get("predicted_answer") or ""))
    )

    error_categories: dict[str, int] = {}
    for row in results:
        cat = categorize_row_error(row, score_key=score_key, error_key=error_key)
        if not cat:
            continue
        error_categories[cat] = error_categories.get(cat, 0) + 1

    out = dict(base)
    out.update(
        {
            "coverage": (judged / processed) if processed else None,
            "strict_accuracy": (score_sum / processed) if processed else None,
            "empty_answer_count": empty_answer_count,
            "error_categories": error_categories,
            "timing_summary": _timing_summary(results, fields=timing_fields),
        }
    )
    return out


def summarize_error_heads(
    results: list[dict[str, Any]], *, error_key: str = "error"
) -> list[dict[str, Any]]:
    """Aggregate row errors by first line for quick diagnostics."""

    counts: dict[str, int] = {}
    for row in results:
        raw_error = row.get(error_key)
        if raw_error is None:
            continue
        text = normalize_line_breaks(str(raw_error)).strip()
        if not text:
            continue
        head = text.splitlines()[0].strip()
        counts[head] = counts.get(head, 0) + 1

    items = [
        {"error_head": error_head, "count": count}
        for error_head, count in counts.items()
    ]
    items.sort(key=lambda x: (-int(x["count"]), str(x["error_head"])))
    return items


def build_comparison_table(mcq_summary: dict[str, Any], no_hint_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    global_rows = [
        {
            "dataset": "MCQ",
            "processed": mcq_summary["processed"],
            "judged": mcq_summary["judged"],
            "correct": mcq_summary["score_sum"],
            "accuracy": mcq_summary["accuracy"],
            "errors": mcq_summary["errors"],
        },
        {
            "dataset": "No-Hint + Judge",
            "processed": no_hint_summary["processed"],
            "judged": no_hint_summary["judged"],
            "correct": no_hint_summary["score_sum"],
            "accuracy": no_hint_summary["accuracy"],
            "errors": no_hint_summary["errors"],
        },
    ]

    levels = sorted(set(mcq_summary["by_level"]) | set(no_hint_summary["by_level"]), key=level_sort_key)
    level_rows: list[dict[str, Any]] = []
    for level in levels:
        m_stats = mcq_summary["by_level"].get(level, {})
        n_stats = no_hint_summary["by_level"].get(level, {})

        m_acc = m_stats.get("accuracy")
        n_acc = n_stats.get("accuracy")
        delta = (n_acc - m_acc) if (m_acc is not None and n_acc is not None) else None

        level_rows.append(
            {
                "level": level,
                "mcq_correct": m_stats.get("correct", 0),
                "mcq_judged": m_stats.get("judged", 0),
                "mcq_accuracy": m_acc,
                "no_hint_correct": n_stats.get("correct", 0),
                "no_hint_judged": n_stats.get("judged", 0),
                "no_hint_accuracy": n_acc,
                "delta_no_hint_minus_mcq": delta,
            }
        )

    return {"global_rows": global_rows, "level_rows": level_rows}


def level_sort_key(level_name: str) -> tuple[int, Any]:
    match = re.match(r"^L(\d+)$", str(level_name).strip().upper())
    return (0, int(match.group(1))) if match else (1, str(level_name))

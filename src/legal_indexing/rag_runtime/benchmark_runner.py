from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from legal_indexing.law_references import LawCatalog

from .benchmarking import (
    align_record,
    build_comparison_table,
    build_dataset_summary,
    build_judge_prompt,
    build_mcq_prompt,
    post_structured_chat,
    validate_judge_output,
    validate_mcq_output,
)
from .langgraph_app import RuntimeResources, run_rag_question
from .schemas import JudgeResult, McqAnswer, schema_to_json_dict


RagRunner = Callable[[str], dict[str, Any]]
StructuredChatFn = Callable[..., dict[str, Any]]


def _compute_reference_law_hit(
    *,
    expected_reference: str | None,
    rag_state: dict[str, Any],
) -> bool | None:
    expected = str(expected_reference or "").strip().lower()
    if not expected:
        return None
    retrieved_laws = _extract_retrieved_law_ids(rag_state)
    if not retrieved_laws:
        return False
    return any(expected in law.lower() for law in retrieved_laws)


def _extract_retrieved_law_ids(rag_state: dict[str, Any]) -> list[str]:
    retrieved = rag_state.get("retrieved") or []
    if not isinstance(retrieved, list):
        return []

    out: list[str] = []
    for doc in retrieved:
        law_id = None
        if isinstance(doc, dict):
            law_id = doc.get("law_id") or (doc.get("payload") or {}).get("law_id")
        else:
            law_id = getattr(doc, "law_id", None)
            if law_id is None:
                payload = getattr(doc, "payload", None) or {}
                law_id = payload.get("law_id")
        cur = str(law_id or "").strip()
        if cur:
            out.append(cur)
    return out


def _compute_reference_metrics(
    *,
    expected_reference: str | None,
    rag_state: dict[str, Any],
    law_catalog: LawCatalog | None,
) -> dict[str, Any]:
    if law_catalog is None:
        return {
            "reference_law_hit": _compute_reference_law_hit(
                expected_reference=expected_reference,
                rag_state=rag_state,
            ),
            "top1_law_match": None,
            "context_precision_proxy": None,
            "expected_law_ids": [],
        }

    expected_text = str(expected_reference or "").strip()
    if not expected_text:
        return {
            "reference_law_hit": None,
            "top1_law_match": None,
            "context_precision_proxy": None,
            "expected_law_ids": [],
        }

    resolved = law_catalog.resolve(expected_text)
    expected_law_ids = set(resolved.law_ids)
    if not expected_law_ids:
        return {
            "reference_law_hit": None,
            "top1_law_match": None,
            "context_precision_proxy": None,
            "expected_law_ids": [],
        }

    retrieved_laws = _extract_retrieved_law_ids(rag_state)
    if not retrieved_laws:
        return {
            "reference_law_hit": False,
            "top1_law_match": False,
            "context_precision_proxy": 0.0,
            "expected_law_ids": sorted(expected_law_ids),
        }

    retrieved_set = set(retrieved_laws)
    hit = bool(retrieved_set & expected_law_ids)
    top1 = bool(retrieved_laws[0] in expected_law_ids)
    in_scope = sum(1 for law in retrieved_laws if law in expected_law_ids)
    precision_proxy = in_scope / float(len(retrieved_laws))
    return {
        "reference_law_hit": hit,
        "top1_law_match": top1,
        "context_precision_proxy": precision_proxy,
        "expected_law_ids": sorted(expected_law_ids),
    }


def _categorize_failure_category(row: dict[str, Any]) -> str | None:
    if row.get("error"):
        return "retrieval_miss"
    score = row.get("final_binary_score")
    if score == 1:
        return None
    if bool(row.get("rag_needs_more_context")):
        return "abstention"
    if int(row.get("retrieved_count") or 0) <= 0:
        return "retrieval_miss"
    ref_hit = row.get("reference_law_hit")
    if ref_hit is False:
        return "retrieval_miss"
    if str((row.get("rag_answer_source") or "")).strip().lower() == "fallback":
        return "abstention"
    judge = row.get("judge_result") or {}
    if isinstance(judge, dict):
        matched = str(judge.get("matched_option_label") or "").strip().upper()
        if matched and matched not in {"", "NONE"}:
            return "contradiction"
    return "context_noise"


def resolve_eval_positions(
    *,
    mcq_rows: list[dict[str, str]],
    no_hint_rows: list[dict[str, str]],
    start_pos: int = 0,
    limit: int = 20,
    run_full: bool = False,
) -> list[int]:
    max_available = min(len(mcq_rows), len(no_hint_rows))
    if max_available <= 0:
        raise RuntimeError("No valid rows available in benchmark datasets.")
    if start_pos < 0 or start_pos >= max_available:
        raise IndexError(f"start_pos out of range: {start_pos}")

    if run_full:
        positions = list(range(start_pos, max_available))
    else:
        end_pos = min(start_pos + max(1, int(limit)), max_available)
        positions = list(range(start_pos, end_pos))
    if not positions:
        raise RuntimeError("No benchmark positions selected.")
    return positions


def _default_rag_runner(runtime: RuntimeResources) -> RagRunner:
    def _runner(question: str) -> dict[str, Any]:
        return run_rag_question(runtime.config, question, resources=runtime)

    return _runner


def run_mcq_benchmark(
    *,
    positions: Sequence[int],
    mcq_rows: list[dict[str, str]],
    no_hint_rows: list[dict[str, str]],
    rag_runner: RagRunner,
    post_chat_fn: StructuredChatFn,
    api_url: str,
    headers: dict[str, str],
    chat_model: str,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for pos in positions:
        record = align_record(int(pos), no_hint_rows, mcq_rows)

        predicted_label = ""
        score: int | None = None
        raw_structured = None
        error = None
        rag_result: dict[str, Any] | None = None
        rag_errors: list[Any] = []

        try:
            rag_result = rag_runner(record["question_mcq_full"])
            context_text = str((rag_result.get("state") or {}).get("context") or "")
        except Exception as exc:
            context_text = ""
            error = f"rag_error: {type(exc).__name__}: {exc}"

        if rag_result is not None:
            rag_errors = list(((rag_result.get("state") or {}).get("pipeline_errors") or []))
        state_payload = (rag_result or {}).get("state") or {}
        retrieved_state = state_payload.get("retrieved") or []
        retrieval_mode = str(state_payload.get("retrieval_mode") or "dense_only")
        dense_retrieved_count = int(state_payload.get("dense_retrieved_count") or 0)
        sparse_retrieved_count = int(state_payload.get("sparse_retrieved_count") or 0)
        fusion_overlap_count = int(state_payload.get("fusion_overlap_count") or 0)
        retrieved_count = (
            len(retrieved_state)
            if isinstance(retrieved_state, list)
            else len((rag_result or {}).get("retrieved_preview") or [])
        )
        context_summary = (rag_result or {}).get("context_summary") or {}
        context_included_count = (
            int(context_summary.get("included_count"))
            if isinstance(context_summary, dict)
            and context_summary.get("included_count") is not None
            else None
        )

        if error is None:
            try:
                prompt = (
                    build_mcq_prompt(record)
                    + "\n\nContesto RAG (usa questo contesto come base informativa):\n"
                    + context_text
                )
                call_out = post_chat_fn(
                    api_url=api_url,
                    headers=headers,
                    payload_schema=schema_to_json_dict(McqAnswer),
                    prompt=prompt,
                    model=chat_model,
                    timeout=timeout_sec,
                )
                raw_structured = call_out["structured"]
                mcq_obj = validate_mcq_output(raw_structured)
                predicted_label = mcq_obj.answer_label
                score = 1 if predicted_label == record["ground_truth_label_mcq"] else 0
            except Exception as exc:
                error = f"mcq_structured_error: {type(exc).__name__}: {exc}"

        item = {
            "qid": record["qid"],
            "level": record["level"],
            "pos": int(pos),
            "ground_truth_label": record["ground_truth_label_mcq"],
            "predicted_label": predicted_label,
            "score": score,
            "is_correct": (score == 1) if score in (0, 1) else None,
            "error": error,
            "raw_response": raw_structured,
            "retrieved_count": retrieved_count,
            "retrieval_mode": retrieval_mode,
            "dense_retrieved_count": dense_retrieved_count,
            "sparse_retrieved_count": sparse_retrieved_count,
            "fusion_overlap_count": fusion_overlap_count,
            "context_included_count": context_included_count,
            "rag_pipeline_errors": rag_errors,
        }
        results.append(item)

    return results


def run_no_hint_benchmark(
    *,
    positions: Sequence[int],
    mcq_rows: list[dict[str, str]],
    no_hint_rows: list[dict[str, str]],
    rag_runner: RagRunner,
    post_chat_fn: StructuredChatFn,
    api_url: str,
    headers: dict[str, str],
    judge_model: str,
    timeout_sec: int,
    law_catalog: LawCatalog | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for pos in positions:
        record = align_record(int(pos), no_hint_rows, mcq_rows)

        rag_answer_text = ""
        rag_answer_source = ""
        rag_was_empty_before_guard = False
        rag_needs_more_context = False
        rag_result: dict[str, Any] | None = None
        rag_errors: list[Any] = []
        judge_result = None
        raw_judge = None
        final_binary_score = None
        error = None

        try:
            rag_result = rag_runner(record["question_no_hint"])
            answer_summary = rag_result.get("answer_summary") or {}
            rag_answer_text = str(answer_summary.get("answer") or "").strip()
            rag_answer_source = str(answer_summary.get("answer_source") or "")
            rag_was_empty_before_guard = bool(answer_summary.get("was_empty_before_guard"))
            rag_needs_more_context = bool(answer_summary.get("needs_more_context"))
            rag_errors = list(((rag_result.get("state") or {}).get("pipeline_errors") or []))
        except Exception as exc:
            error = f"rag_error: {type(exc).__name__}: {exc}"
        state_payload = (rag_result or {}).get("state") or {}
        retrieved_state = state_payload.get("retrieved") or []
        retrieval_mode = str(state_payload.get("retrieval_mode") or "dense_only")
        dense_retrieved_count = int(state_payload.get("dense_retrieved_count") or 0)
        sparse_retrieved_count = int(state_payload.get("sparse_retrieved_count") or 0)
        fusion_overlap_count = int(state_payload.get("fusion_overlap_count") or 0)
        retrieved_count = (
            len(retrieved_state)
            if isinstance(retrieved_state, list)
            else len((rag_result or {}).get("retrieved_preview") or [])
        )
        context_summary = (rag_result or {}).get("context_summary") or {}
        context_included_count = (
            int(context_summary.get("included_count"))
            if isinstance(context_summary, dict)
            and context_summary.get("included_count") is not None
            else None
        )
        ref_metrics = _compute_reference_metrics(
            expected_reference=record.get("ground_truth_reference_law"),
            rag_state=state_payload if isinstance(state_payload, dict) else {},
            law_catalog=law_catalog,
        )

        if error is None:
            try:
                judge_out = post_chat_fn(
                    api_url=api_url,
                    headers=headers,
                    payload_schema=schema_to_json_dict(JudgeResult),
                    prompt=build_judge_prompt(record, rag_answer_text),
                    model=judge_model,
                    timeout=timeout_sec,
                )
                raw_judge = judge_out["structured"]
                judge_obj = validate_judge_output(raw_judge)
                judge_result = judge_obj.model_dump()
                final_binary_score = int(judge_obj.score)
            except Exception as exc:
                error = f"judge_structured_error: {type(exc).__name__}: {exc}"

        item = {
            "qid": record["qid"],
            "level": record["level"],
            "pos": int(pos),
            "predicted_answer": rag_answer_text,
            "judge_context": {
                "question_mcq_full": record["question_mcq_full"],
                "correct_option_label": record["ground_truth_label_mcq"],
                "correct_option_text": record["correct_option_text"],
            },
            "judge_result": judge_result,
            "final_binary_score": final_binary_score,
            "error": error,
            "raw_judge": raw_judge,
            "retrieved_count": retrieved_count,
            "retrieval_mode": retrieval_mode,
            "dense_retrieved_count": dense_retrieved_count,
            "sparse_retrieved_count": sparse_retrieved_count,
            "fusion_overlap_count": fusion_overlap_count,
            "context_included_count": context_included_count,
            "rag_pipeline_errors": rag_errors,
            "rag_answer_source": rag_answer_source,
            "rag_was_empty_before_guard": rag_was_empty_before_guard,
            "rag_needs_more_context": rag_needs_more_context,
            "reference_law_hit": ref_metrics["reference_law_hit"],
            "top1_law_match": ref_metrics["top1_law_match"],
            "context_precision_proxy": ref_metrics["context_precision_proxy"],
            "expected_law_ids": ref_metrics["expected_law_ids"],
        }
        item["failure_category"] = _categorize_failure_category(item)
        results.append(item)

    return results


def run_advanced_benchmark(
    *,
    runtime: RuntimeResources,
    mcq_rows: list[dict[str, str]],
    no_hint_rows: list[dict[str, str]],
    positions: Sequence[int],
    api_url: str,
    headers: dict[str, str],
    chat_model: str,
    judge_model: str,
    timeout_sec: int = 120,
    rag_runner: RagRunner | None = None,
    post_chat_fn: StructuredChatFn = post_structured_chat,
) -> dict[str, Any]:
    rag_runner = rag_runner or _default_rag_runner(runtime)

    mcq_results = run_mcq_benchmark(
        positions=positions,
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        rag_runner=rag_runner,
        post_chat_fn=post_chat_fn,
        api_url=api_url,
        headers=headers,
        chat_model=chat_model,
        timeout_sec=timeout_sec,
    )
    no_hint_results = run_no_hint_benchmark(
        positions=positions,
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        rag_runner=rag_runner,
        post_chat_fn=post_chat_fn,
        api_url=api_url,
        headers=headers,
        judge_model=judge_model,
        timeout_sec=timeout_sec,
        law_catalog=getattr(runtime, "law_catalog", None),
    )

    mcq_summary = build_dataset_summary("RAG-ADV-MCQ", mcq_results, score_key="score")
    no_hint_summary = build_dataset_summary(
        "RAG-ADV No-Hint + Judge",
        no_hint_results,
        score_key="final_binary_score",
    )
    comparison_table = build_comparison_table(mcq_summary, no_hint_summary)
    mcq_acc = mcq_summary.get("accuracy")
    no_hint_acc = no_hint_summary.get("accuracy")
    comparison_summary = {
        "global": {
            "mcq_accuracy": mcq_acc,
            "no_hint_accuracy": no_hint_acc,
            "delta_no_hint_minus_mcq": (
                (no_hint_acc - mcq_acc) if (mcq_acc is not None and no_hint_acc is not None) else None
            ),
        },
        "by_level": comparison_table["level_rows"],
    }

    no_hint_by_level: dict[str, dict[str, Any]] = {}
    for row in no_hint_results:
        level = str(row.get("level") or "")
        stats = no_hint_by_level.setdefault(
            level,
            {
                "processed": 0,
                "empty_detected": 0,
                "fallback_used": 0,
            },
        )
        stats["processed"] += 1
        if bool(row.get("rag_was_empty_before_guard")):
            stats["empty_detected"] += 1
        if str(row.get("rag_answer_source") or "").strip().lower() == "fallback":
            stats["fallback_used"] += 1

    for stats in no_hint_by_level.values():
        processed = int(stats["processed"])
        stats["empty_rate"] = (stats["empty_detected"] / processed) if processed > 0 else 0.0

    failure_breakdown: dict[str, int] = {}
    failure_by_level: dict[str, dict[str, int]] = {}
    retrieval_mode_counts: dict[str, int] = {}
    ref_hit_known = 0
    ref_hit_true = 0
    top1_known = 0
    top1_true = 0
    precision_values: list[float] = []
    ref_hit_by_level: dict[str, dict[str, int]] = {}
    for row in no_hint_results:
        category = str(row.get("failure_category") or "")
        if category:
            failure_breakdown[category] = failure_breakdown.get(category, 0) + 1
            lvl = str(row.get("level") or "")
            lb = failure_by_level.setdefault(lvl, {})
            lb[category] = lb.get(category, 0) + 1

        mode = str(row.get("retrieval_mode") or "dense_only")
        retrieval_mode_counts[mode] = retrieval_mode_counts.get(mode, 0) + 1

        if row.get("reference_law_hit") is not None:
            ref_hit_known += 1
            if bool(row.get("reference_law_hit")):
                ref_hit_true += 1
            lvl = str(row.get("level") or "")
            by_lvl = ref_hit_by_level.setdefault(lvl, {"known": 0, "hit": 0})
            by_lvl["known"] += 1
            if bool(row.get("reference_law_hit")):
                by_lvl["hit"] += 1

        if row.get("top1_law_match") is not None:
            top1_known += 1
            if bool(row.get("top1_law_match")):
                top1_true += 1

        cpp = row.get("context_precision_proxy")
        if cpp is not None:
            try:
                precision_values.append(float(cpp))
            except Exception:
                pass

    reference_hit_by_level = {
        level: (
            (vals["hit"] / vals["known"]) if vals["known"] > 0 else None
        )
        for level, vals in sorted(ref_hit_by_level.items())
    }

    diagnostics = {
        "mcq_pipeline_error_rows": sum(1 for row in mcq_results if row.get("rag_pipeline_errors")),
        "no_hint_pipeline_error_rows": sum(
            1 for row in no_hint_results if row.get("rag_pipeline_errors")
        ),
        "no_hint_empty_detected_count": sum(
            1 for row in no_hint_results if row.get("rag_was_empty_before_guard")
        ),
        "no_hint_fallback_used_count": sum(
            1
            for row in no_hint_results
            if str(row.get("rag_answer_source") or "").strip().lower() == "fallback"
        ),
        "no_hint_empty_by_level": no_hint_by_level,
        "no_hint_failure_breakdown": failure_breakdown,
        "no_hint_failure_by_level": failure_by_level,
        "no_hint_retrieval_mode_counts": retrieval_mode_counts,
        "no_hint_reference_hit_rate": (
            (ref_hit_true / ref_hit_known) if ref_hit_known > 0 else None
        ),
        "no_hint_top1_law_match_rate": (
            (top1_true / top1_known) if top1_known > 0 else None
        ),
        "no_hint_context_precision_proxy_avg": (
            (sum(precision_values) / len(precision_values))
            if precision_values
            else None
        ),
        "no_hint_reference_hit_rate_by_level": reference_hit_by_level,
    }

    return {
        "mcq_results": mcq_results,
        "no_hint_results": no_hint_results,
        "mcq_summary": mcq_summary,
        "no_hint_summary": no_hint_summary,
        "comparison_table": comparison_table,
        "comparison_summary": comparison_summary,
        "diagnostics": diagnostics,
    }


def persist_benchmark_artifacts(
    *,
    artifacts_dir: Path,
    label: str = "advanced",
    mode: str,
    config_payload: dict[str, Any],
    index_contract: dict[str, Any],
    benchmark_payload: dict[str, Any],
) -> dict[str, Path]:
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"rag_{str(label or 'advanced').strip().lower()}_{mode}_benchmark_{run_stamp}"
    json_path = artifacts_dir / f"{prefix}.json"
    mcq_csv_path = artifacts_dir / f"{prefix}_mcq.csv"
    no_hint_csv_path = artifacts_dir / f"{prefix}_no_hint.csv"

    payload = {
        "run_stamp_utc": run_stamp,
        "mode": mode,
        "config": config_payload,
        "index_contract": index_contract,
        "mcq_summary": benchmark_payload["mcq_summary"],
        "no_hint_summary": benchmark_payload["no_hint_summary"],
        "comparison_summary": benchmark_payload["comparison_summary"],
        "diagnostics": benchmark_payload.get("diagnostics") or {},
        "mcq_results": benchmark_payload["mcq_results"],
        "no_hint_results": benchmark_payload["no_hint_results"],
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(benchmark_payload["mcq_results"]).to_csv(
        mcq_csv_path, index=False, encoding="utf-8"
    )
    pd.DataFrame(benchmark_payload["no_hint_results"]).to_csv(
        no_hint_csv_path, index=False, encoding="utf-8"
    )

    return {
        "json": json_path,
        "mcq_csv": mcq_csv_path,
        "no_hint_csv": no_hint_csv_path,
    }


def build_naive_vs_advanced_comparison(
    *,
    naive_payload: dict[str, Any],
    advanced_payload: dict[str, Any],
) -> dict[str, Any]:
    naive_mcq = naive_payload.get("mcq_summary") or {}
    naive_no_hint = naive_payload.get("no_hint_summary") or {}
    adv_mcq = advanced_payload.get("mcq_summary") or {}
    adv_no_hint = advanced_payload.get("no_hint_summary") or {}

    def _delta(a: Any, b: Any) -> float | None:
        try:
            if a is None or b is None:
                return None
            return float(a) - float(b)
        except Exception:
            return None

    by_level_rows: list[dict[str, Any]] = []
    levels = set((naive_no_hint.get("by_level") or {}).keys()) | set(
        (adv_no_hint.get("by_level") or {}).keys()
    )
    for level in sorted(levels):
        n = (naive_no_hint.get("by_level") or {}).get(level) or {}
        a = (adv_no_hint.get("by_level") or {}).get(level) or {}
        by_level_rows.append(
            {
                "level": level,
                "naive_no_hint_accuracy": n.get("accuracy"),
                "advanced_no_hint_accuracy": a.get("accuracy"),
                "delta_advanced_minus_naive": _delta(a.get("accuracy"), n.get("accuracy")),
            }
        )

    return {
        "global": {
            "naive_mcq_accuracy": naive_mcq.get("accuracy"),
            "advanced_mcq_accuracy": adv_mcq.get("accuracy"),
            "delta_mcq_advanced_minus_naive": _delta(
                adv_mcq.get("accuracy"), naive_mcq.get("accuracy")
            ),
            "naive_no_hint_accuracy": naive_no_hint.get("accuracy"),
            "advanced_no_hint_accuracy": adv_no_hint.get("accuracy"),
            "delta_no_hint_advanced_minus_naive": _delta(
                adv_no_hint.get("accuracy"), naive_no_hint.get("accuracy")
            ),
        },
        "by_level": by_level_rows,
    }


def persist_naive_vs_advanced_comparison(
    *,
    artifacts_dir: Path,
    comparison_payload: dict[str, Any],
) -> Path:
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = artifacts_dir / f"rag_comparison_naive_vs_advanced_{run_stamp}.json"
    path.write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path

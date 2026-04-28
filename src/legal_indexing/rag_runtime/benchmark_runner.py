from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
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
from .langgraph_app import (
    RuntimeResources,
    build_rag_graph,
    build_rag_retrieval_context_graph,
    run_rag_question,
    run_rag_retrieval_context,
)
from .schemas import JudgeResult, McqAnswer, schema_to_json_dict


RagRunner = Callable[[str], dict[str, Any]]
StructuredChatFn = Callable[..., dict[str, Any]]


def _sorted_unique_positions(positions: Sequence[int]) -> list[int]:
    return sorted({int(pos) for pos in positions})


def _normalize_for_fingerprint(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_fingerprint(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_for_fingerprint(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_fingerprint(v) for v in value]
    if isinstance(value, set):
        return sorted([_normalize_for_fingerprint(v) for v in value], key=lambda x: str(x))
    return value


def _stable_sha256(payload: Any) -> str:
    normalized = _normalize_for_fingerprint(payload)
    body = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _is_transient_chat_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "timeout" in text:
        return True
    if "connection" in text:
        return True
    if "http 429" in text:
        return True
    if "http 500" in text or "http 502" in text or "http 503" in text or "http 504" in text:
        return True
    return False


def _is_transient_error_text(text: str) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    if "timeout" in low:
        return True
    if "connection" in low:
        return True
    if "http 429" in low:
        return True
    if "http 500" in low or "http 502" in low or "http 503" in low or "http 504" in low:
        return True
    return False


def _call_structured_with_retry(
    *,
    post_chat_fn: StructuredChatFn,
    api_url: str,
    headers: dict[str, str],
    payload_schema: dict[str, Any],
    prompt: str,
    model: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> dict[str, Any]:
    attempts = max(0, int(max_retries)) + 1
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return post_chat_fn(
                api_url=api_url,
                headers=headers,
                payload_schema=payload_schema,
                prompt=prompt,
                model=model,
                timeout=timeout_sec,
            )
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
            if attempt >= attempts or not _is_transient_chat_error(exc):
                raise
            sleep_seconds = max(0.0, float(retry_backoff_sec)) * float(attempt)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def _build_checkpoint_meta(
    *,
    runtime: RuntimeResources,
    chat_model: str,
    judge_model: str,
    timeout_sec: int,
    max_workers: int,
    post_chat_max_retries: int,
    post_chat_retry_backoff_sec: float,
    positions: Sequence[int],
) -> dict[str, Any]:
    config_fingerprint = _stable_sha256(
        {
            "runtime_config": getattr(runtime, "config", None),
            "chat_model": str(chat_model),
            "judge_model": str(judge_model),
            "timeout_sec": int(timeout_sec),
            "max_workers": int(max_workers),
            "post_chat_max_retries": int(post_chat_max_retries),
            "post_chat_retry_backoff_sec": float(post_chat_retry_backoff_sec),
        }
    )
    positions_sorted = _sorted_unique_positions(positions)
    positions_fingerprint = _stable_sha256(positions_sorted)
    return {
        "schema_version": "2",
        "config_fingerprint": config_fingerprint,
        "positions_fingerprint": positions_fingerprint,
    }


def _load_checkpoint_rows(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any] | None]:
    mcq_rows: dict[int, dict[str, Any]] = {}
    no_hint_rows: dict[int, dict[str, Any]] = {}
    meta_row: dict[str, Any] | None = None
    if not path.exists():
        return mcq_rows, no_hint_rows, meta_row
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            section = str(payload.get("section") or "").strip().lower()
            row = payload.get("row")
            if section == "meta":
                if isinstance(row, dict):
                    meta_row = row
                continue
            if not isinstance(row, dict):
                continue
            try:
                pos = int(row.get("pos"))
            except Exception:
                continue
            if section == "mcq":
                mcq_rows[pos] = row
            elif section == "no_hint":
                no_hint_rows[pos] = row
    return mcq_rows, no_hint_rows, meta_row


def _append_checkpoint_meta(path: Path, *, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"section": "meta", "row": meta}, ensure_ascii=False) + "\n")


def _validate_checkpoint_meta(
    *,
    checkpoint_file: Path,
    loaded_meta: dict[str, Any] | None,
    expected_meta: dict[str, Any],
    has_loaded_rows: bool,
) -> None:
    if loaded_meta is None:
        if has_loaded_rows:
            raise RuntimeError(
                "Legacy checkpoint without meta/fingerprint detected; resume is not supported. "
                f"Delete checkpoint file and rerun: {checkpoint_file}"
            )
        return

    if str(loaded_meta.get("schema_version") or "") != "2":
        raise RuntimeError(
            "Checkpoint schema mismatch for resume. "
            f"Expected schema_version=2 in {checkpoint_file}"
        )

    if str(loaded_meta.get("config_fingerprint") or "") != str(
        expected_meta.get("config_fingerprint") or ""
    ):
        raise RuntimeError(
            "Checkpoint config fingerprint mismatch; resume blocked to avoid mixed runs. "
            f"Delete checkpoint file and rerun: {checkpoint_file}"
        )

    if str(loaded_meta.get("positions_fingerprint") or "") != str(
        expected_meta.get("positions_fingerprint") or ""
    ):
        raise RuntimeError(
            "Checkpoint positions fingerprint mismatch; resume blocked to avoid mixed runs. "
            f"Delete checkpoint file and rerun: {checkpoint_file}"
        )


def _append_checkpoint_rows(path: Path, *, section: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps({"section": section, "pos": int(row.get("pos", -1)), "row": row}, ensure_ascii=False)
                + "\n"
            )


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


def _extract_chunk_law_pairs(rag_state: dict[str, Any]) -> list[tuple[str, str]]:
    retrieved = rag_state.get("retrieved") or []
    if not isinstance(retrieved, list):
        return []

    out: list[tuple[str, str]] = []
    for doc in retrieved:
        chunk_id = ""
        law_id = ""
        if isinstance(doc, dict):
            chunk_id = str(doc.get("chunk_id") or "").strip()
            law_id = str(doc.get("law_id") or (doc.get("payload") or {}).get("law_id") or "").strip()
        else:
            chunk_id = str(getattr(doc, "chunk_id", "") or "").strip()
            law_id = str(getattr(doc, "law_id", "") or "").strip()
            if not law_id:
                payload = getattr(doc, "payload", None) or {}
                law_id = str(payload.get("law_id") or "").strip()
        if chunk_id or law_id:
            out.append((chunk_id, law_id))
    return out


def _compute_unique_law_counts(
    *,
    rag_state: dict[str, Any],
    context_summary: dict[str, Any] | None,
) -> tuple[int, int]:
    pairs = _extract_chunk_law_pairs(rag_state)
    unique_retrieved = {law_id for _, law_id in pairs if law_id}
    included_chunk_ids = set((context_summary or {}).get("included_chunk_ids") or [])
    if not included_chunk_ids:
        return len(unique_retrieved), len(unique_retrieved)
    unique_context = {
        law_id for chunk_id, law_id in pairs if law_id and chunk_id and chunk_id in included_chunk_ids
    }
    return len(unique_retrieved), len(unique_context)


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
    compiled_app = build_rag_graph(runtime.config, runtime)

    def _runner(question: str) -> dict[str, Any]:
        return run_rag_question(
            runtime.config,
            question,
            resources=runtime,
            compiled_app=compiled_app,
        )

    return _runner


def _default_retrieval_context_runner(runtime: RuntimeResources) -> RagRunner:
    pipeline_mode = str(runtime.config.pipeline_mode or "naive")
    compiled_app = build_rag_retrieval_context_graph(
        runtime.config,
        runtime,
        pipeline_mode=pipeline_mode,
    )

    def _runner(question: str) -> dict[str, Any]:
        return run_rag_retrieval_context(
            runtime.config,
            question,
            resources=runtime,
            pipeline_mode=pipeline_mode,
            compiled_app=compiled_app,
        )

    return _runner


def run_mcq_benchmark(
    *,
    positions: Sequence[int],
    mcq_rows: list[dict[str, str]],
    no_hint_rows: list[dict[str, str]],
    rag_runner: RagRunner,
    rag_retrieval_runner: RagRunner | None = None,
    post_chat_fn: StructuredChatFn,
    api_url: str,
    headers: dict[str, str],
    chat_model: str,
    timeout_sec: int,
    max_workers: int = 3,
    post_chat_max_retries: int = 2,
    post_chat_retry_backoff_sec: float = 0.5,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    retrieval_runner = rag_retrieval_runner or rag_runner
    positions_sorted = _sorted_unique_positions(positions)

    def _run_position(pos: int) -> dict[str, Any]:
        record = align_record(int(pos), no_hint_rows, mcq_rows)

        predicted_label = ""
        score: int | None = None
        raw_structured = None
        error = None
        rag_result: dict[str, Any] | None = None
        rag_errors: list[Any] = []

        try:
            rag_result = retrieval_runner(record["question_mcq_full"])
            context_text = str((rag_result.get("state") or {}).get("context") or "")
        except Exception as exc:
            context_text = ""
            error = f"rag_error: {type(exc).__name__}: {exc}"

        if rag_result is not None:
            rag_errors = list(((rag_result.get("state") or {}).get("pipeline_errors") or []))
        state_payload = (rag_result or {}).get("state") or {}
        retrieved_state = state_payload.get("retrieved") or []
        filters_summary = (rag_result or {}).get("filters_summary") or {}
        graph_payload = (rag_result or {}).get("graph_expansion") or {}
        rewritten_queries = (rag_result or {}).get("rewritten_queries") or []
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
        unique_laws_retrieved, unique_laws_in_context = _compute_unique_law_counts(
            rag_state=state_payload if isinstance(state_payload, dict) else {},
            context_summary=context_summary if isinstance(context_summary, dict) else None,
        )

        if error is None:
            try:
                prompt = (
                    build_mcq_prompt(record)
                    + "\n\nContesto RAG (usa questo contesto come base informativa):\n"
                    + context_text
                )
                call_out = _call_structured_with_retry(
                    post_chat_fn=post_chat_fn,
                    api_url=api_url,
                    headers=headers,
                    payload_schema=schema_to_json_dict(McqAnswer),
                    prompt=prompt,
                    model=chat_model,
                    timeout_sec=timeout_sec,
                    max_retries=post_chat_max_retries,
                    retry_backoff_sec=post_chat_retry_backoff_sec,
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
            "rewrite_count": len(rewritten_queries) if isinstance(rewritten_queries, list) else 0,
            "metadata_heuristics": list(filters_summary.get("metadata_heuristics") or []),
            "metadata_hard_law_filter_applied": bool(
                filters_summary.get("metadata_hard_law_filter_applied")
            ),
            "metadata_hard_article_filter_applied": bool(
                filters_summary.get("metadata_hard_article_filter_applied")
            ),
            "graph_enabled": bool(graph_payload.get("enabled")),
            "graph_reason": str(graph_payload.get("reason") or ""),
            "graph_retrieved_count": int(graph_payload.get("graph_retrieved_count") or 0),
            "unique_laws_retrieved": unique_laws_retrieved,
            "unique_laws_in_context": unique_laws_in_context,
            "rag_pipeline_errors": rag_errors,
        }
        return item

    results: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers <= 1:
        for pos in positions_sorted:
            item = _run_position(pos)
            results.append(item)
            if on_result is not None:
                on_result(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_position, pos): pos for pos in positions_sorted}
            for future in as_completed(futures):
                item = future.result()
                results.append(item)
                if on_result is not None:
                    on_result(item)

    results.sort(key=lambda row: int(row.get("pos") or -1))
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
    max_workers: int = 3,
    post_chat_max_retries: int = 2,
    post_chat_retry_backoff_sec: float = 0.5,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    positions_sorted = _sorted_unique_positions(positions)

    def _run_position(pos: int) -> dict[str, Any]:
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
        filters_summary = (rag_result or {}).get("filters_summary") or {}
        graph_payload = (rag_result or {}).get("graph_expansion") or {}
        rewritten_queries = (rag_result or {}).get("rewritten_queries") or []
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
        unique_laws_retrieved, unique_laws_in_context = _compute_unique_law_counts(
            rag_state=state_payload if isinstance(state_payload, dict) else {},
            context_summary=context_summary if isinstance(context_summary, dict) else None,
        )
        ref_metrics = _compute_reference_metrics(
            expected_reference=record.get("ground_truth_reference_law"),
            rag_state=state_payload if isinstance(state_payload, dict) else {},
            law_catalog=law_catalog,
        )

        if error is None:
            try:
                judge_out = _call_structured_with_retry(
                    post_chat_fn=post_chat_fn,
                    api_url=api_url,
                    headers=headers,
                    payload_schema=schema_to_json_dict(JudgeResult),
                    prompt=build_judge_prompt(record, rag_answer_text),
                    model=judge_model,
                    timeout_sec=timeout_sec,
                    max_retries=post_chat_max_retries,
                    retry_backoff_sec=post_chat_retry_backoff_sec,
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
            "rewrite_count": len(rewritten_queries) if isinstance(rewritten_queries, list) else 0,
            "metadata_heuristics": list(filters_summary.get("metadata_heuristics") or []),
            "metadata_hard_law_filter_applied": bool(
                filters_summary.get("metadata_hard_law_filter_applied")
            ),
            "metadata_hard_article_filter_applied": bool(
                filters_summary.get("metadata_hard_article_filter_applied")
            ),
            "graph_enabled": bool(graph_payload.get("enabled")),
            "graph_reason": str(graph_payload.get("reason") or ""),
            "graph_retrieved_count": int(graph_payload.get("graph_retrieved_count") or 0),
            "unique_laws_retrieved": unique_laws_retrieved,
            "unique_laws_in_context": unique_laws_in_context,
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
        return item

    results: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers <= 1:
        for pos in positions_sorted:
            item = _run_position(pos)
            results.append(item)
            if on_result is not None:
                on_result(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_position, pos): pos for pos in positions_sorted}
            for future in as_completed(futures):
                item = future.result()
                results.append(item)
                if on_result is not None:
                    on_result(item)

    results.sort(key=lambda row: int(row.get("pos") or -1))
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
    rag_retrieval_runner: RagRunner | None = None,
    post_chat_fn: StructuredChatFn = post_structured_chat,
    max_workers: int = 3,
    post_chat_max_retries: int = 2,
    post_chat_retry_backoff_sec: float = 0.5,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
    resume: bool = False,
) -> dict[str, Any]:
    if rag_runner is None:
        rag_runner = _default_rag_runner(runtime)
    if rag_retrieval_runner is None:
        try:
            rag_retrieval_runner = _default_retrieval_context_runner(runtime)
        except Exception:
            rag_retrieval_runner = rag_runner

    positions_sorted = _sorted_unique_positions(positions)
    expected_checkpoint_meta = _build_checkpoint_meta(
        runtime=runtime,
        chat_model=chat_model,
        judge_model=judge_model,
        timeout_sec=timeout_sec,
        max_workers=max_workers,
        post_chat_max_retries=post_chat_max_retries,
        post_chat_retry_backoff_sec=post_chat_retry_backoff_sec,
        positions=positions_sorted,
    )
    checkpoint_file: Path | None = None
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path).resolve()
        if checkpoint_file.exists() and not resume:
            checkpoint_file.unlink()

    loaded_mcq_rows: dict[int, dict[str, Any]] = {}
    loaded_no_hint_rows: dict[int, dict[str, Any]] = {}
    checkpoint_meta_written = False
    if checkpoint_file is not None and resume:
        loaded_mcq_rows, loaded_no_hint_rows, loaded_meta = _load_checkpoint_rows(checkpoint_file)
        _validate_checkpoint_meta(
            checkpoint_file=checkpoint_file,
            loaded_meta=loaded_meta,
            expected_meta=expected_checkpoint_meta,
            has_loaded_rows=bool(loaded_mcq_rows or loaded_no_hint_rows),
        )
        if loaded_meta is not None:
            checkpoint_meta_written = True
        selected_positions = set(positions_sorted)
        loaded_mcq_rows = {
            pos: row for pos, row in loaded_mcq_rows.items() if pos in selected_positions
        }
        loaded_no_hint_rows = {
            pos: row for pos, row in loaded_no_hint_rows.items() if pos in selected_positions
        }
    elif checkpoint_file is not None:
        _append_checkpoint_meta(checkpoint_file, meta=expected_checkpoint_meta)
        checkpoint_meta_written = True

    pending_mcq_positions = [pos for pos in positions_sorted if pos not in loaded_mcq_rows]
    pending_no_hint_positions = [pos for pos in positions_sorted if pos not in loaded_no_hint_rows]
    checkpoint_batch_size = max(1, int(checkpoint_every))
    mcq_checkpoint_buffer: list[dict[str, Any]] = []
    no_hint_checkpoint_buffer: list[dict[str, Any]] = []

    def _flush_checkpoint(*, force: bool = False) -> None:
        if checkpoint_file is None:
            return
        nonlocal checkpoint_meta_written
        if not checkpoint_meta_written:
            _append_checkpoint_meta(checkpoint_file, meta=expected_checkpoint_meta)
            checkpoint_meta_written = True
        if force or len(mcq_checkpoint_buffer) >= checkpoint_batch_size:
            _append_checkpoint_rows(checkpoint_file, section="mcq", rows=mcq_checkpoint_buffer)
            mcq_checkpoint_buffer.clear()
        if force or len(no_hint_checkpoint_buffer) >= checkpoint_batch_size:
            _append_checkpoint_rows(checkpoint_file, section="no_hint", rows=no_hint_checkpoint_buffer)
            no_hint_checkpoint_buffer.clear()

    def _on_mcq_row(row: dict[str, Any]) -> None:
        if checkpoint_file is None:
            return
        mcq_checkpoint_buffer.append(row)
        _flush_checkpoint(force=False)

    def _on_no_hint_row(row: dict[str, Any]) -> None:
        if checkpoint_file is None:
            return
        no_hint_checkpoint_buffer.append(row)
        _flush_checkpoint(force=False)

    mcq_new_results = run_mcq_benchmark(
        positions=pending_mcq_positions,
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        rag_runner=rag_runner,
        rag_retrieval_runner=rag_retrieval_runner,
        post_chat_fn=post_chat_fn,
        api_url=api_url,
        headers=headers,
        chat_model=chat_model,
        timeout_sec=timeout_sec,
        max_workers=max_workers,
        post_chat_max_retries=post_chat_max_retries,
        post_chat_retry_backoff_sec=post_chat_retry_backoff_sec,
        on_result=_on_mcq_row,
    )
    no_hint_new_results = run_no_hint_benchmark(
        positions=pending_no_hint_positions,
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        rag_runner=rag_runner,
        post_chat_fn=post_chat_fn,
        api_url=api_url,
        headers=headers,
        judge_model=judge_model,
        timeout_sec=timeout_sec,
        law_catalog=getattr(runtime, "law_catalog", None),
        max_workers=max_workers,
        post_chat_max_retries=post_chat_max_retries,
        post_chat_retry_backoff_sec=post_chat_retry_backoff_sec,
        on_result=_on_no_hint_row,
    )
    _flush_checkpoint(force=True)

    merged_mcq = dict(loaded_mcq_rows)
    for row in mcq_new_results:
        merged_mcq[int(row.get("pos") or -1)] = row
    mcq_results = [merged_mcq[pos] for pos in sorted(merged_mcq.keys())]

    merged_no_hint = dict(loaded_no_hint_rows)
    for row in no_hint_new_results:
        merged_no_hint[int(row.get("pos") or -1)] = row
    no_hint_results = [merged_no_hint[pos] for pos in sorted(merged_no_hint.keys())]

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
    graph_reason_counts: dict[str, int] = {}
    graph_enabled_count = 0
    graph_positive_count = 0
    rewrite_count_values: list[int] = []
    metadata_hard_law_filter_applied_count = 0
    unique_laws_retrieved_values: list[int] = []
    unique_laws_in_context_values: list[int] = []
    for row in no_hint_results:
        category = str(row.get("failure_category") or "")
        if category:
            failure_breakdown[category] = failure_breakdown.get(category, 0) + 1
            lvl = str(row.get("level") or "")
            lb = failure_by_level.setdefault(lvl, {})
            lb[category] = lb.get(category, 0) + 1

        mode = str(row.get("retrieval_mode") or "dense_only")
        retrieval_mode_counts[mode] = retrieval_mode_counts.get(mode, 0) + 1
        graph_reason = str(row.get("graph_reason") or "")
        graph_reason_counts[graph_reason] = graph_reason_counts.get(graph_reason, 0) + 1
        if bool(row.get("graph_enabled")):
            graph_enabled_count += 1
        if int(row.get("graph_retrieved_count") or 0) > 0:
            graph_positive_count += 1
        rewrite_count_values.append(int(row.get("rewrite_count") or 0))
        if bool(row.get("metadata_hard_law_filter_applied")):
            metadata_hard_law_filter_applied_count += 1
        unique_laws_retrieved_values.append(int(row.get("unique_laws_retrieved") or 0))
        unique_laws_in_context_values.append(int(row.get("unique_laws_in_context") or 0))

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

    mcq_pipeline_error_rows_legacy = sum(1 for row in mcq_results if row.get("rag_pipeline_errors"))
    no_hint_pipeline_error_rows_legacy = sum(
        1 for row in no_hint_results if row.get("rag_pipeline_errors")
    )
    no_hint_hard_error_rows = sum(
        1 for row in no_hint_results if str(row.get("error") or "").strip()
    )
    no_hint_soft_guard_event_rows = sum(
        1
        for row in no_hint_results
        if bool(row.get("rag_pipeline_errors")) and not str(row.get("error") or "").strip()
    )

    no_hint_guard_empty_first_pass_count = 0
    no_hint_guard_retry_needs_more_context_count = 0
    no_hint_guard_missing_valid_citation_count = 0
    for row in no_hint_results:
        errors = row.get("rag_pipeline_errors") or []
        if not isinstance(errors, list):
            continue
        for event in errors:
            if not isinstance(event, dict):
                continue
            stage = str(event.get("stage") or "").strip()
            err = str(event.get("error") or "").strip()
            if err == "empty_answer_detected:first_pass":
                no_hint_guard_empty_first_pass_count += 1
            if stage == "generate_answer_structured_retry_needs_more_context":
                no_hint_guard_retry_needs_more_context_count += 1
            if err.startswith("missing_valid_citation"):
                no_hint_guard_missing_valid_citation_count += 1

    transient_error_rows = sum(
        1
        for row in [*mcq_results, *no_hint_results]
        if _is_transient_error_text(str(row.get("error") or ""))
    )
    total_rows = len(mcq_results) + len(no_hint_results)
    transient_error_rate = (transient_error_rows / total_rows) if total_rows > 0 else 0.0
    recommended_max_workers = (
        3 if (transient_error_rate > 0.03 and int(max_workers) > 3) else int(max_workers)
    )

    diagnostics = {
        "benchmark_max_workers": int(max_workers),
        "benchmark_applied_max_workers": int(max_workers),
        "benchmark_recommended_max_workers_next_run": recommended_max_workers,
        "benchmark_post_chat_max_retries": int(post_chat_max_retries),
        "benchmark_checkpoint_path": str(checkpoint_file) if checkpoint_file is not None else None,
        "benchmark_checkpoint_resume": bool(resume),
        "benchmark_checkpoint_schema_version": str(expected_checkpoint_meta["schema_version"]),
        "benchmark_checkpoint_config_fingerprint": str(
            expected_checkpoint_meta["config_fingerprint"]
        ),
        "benchmark_checkpoint_positions_fingerprint": str(
            expected_checkpoint_meta["positions_fingerprint"]
        ),
        "mcq_checkpoint_loaded_rows": len(loaded_mcq_rows),
        "no_hint_checkpoint_loaded_rows": len(loaded_no_hint_rows),
        "mcq_pipeline_error_rows": mcq_pipeline_error_rows_legacy,
        "no_hint_pipeline_error_rows": no_hint_pipeline_error_rows_legacy,
        "no_hint_hard_error_rows": no_hint_hard_error_rows,
        "no_hint_soft_guard_event_rows": no_hint_soft_guard_event_rows,
        "no_hint_guard_empty_first_pass_count": no_hint_guard_empty_first_pass_count,
        "no_hint_guard_retry_needs_more_context_count": no_hint_guard_retry_needs_more_context_count,
        "no_hint_guard_missing_valid_citation_count": no_hint_guard_missing_valid_citation_count,
        "transient_error_rows": transient_error_rows,
        "transient_error_rate": transient_error_rate,
        "legacy_aggregate_notes": {
            "mcq_pipeline_error_rows": (
                "Legacy aggregate: counts rows with rag_pipeline_errors; includes non-fatal guard events."
            ),
            "no_hint_pipeline_error_rows": (
                "Legacy aggregate: counts rows with rag_pipeline_errors; includes non-fatal guard events."
            ),
        },
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
        "no_hint_graph_enabled_count": graph_enabled_count,
        "no_hint_graph_retrieved_positive_count": graph_positive_count,
        "no_hint_graph_reason_counts": graph_reason_counts,
        "no_hint_avg_rewrite_count": (
            sum(rewrite_count_values) / len(rewrite_count_values)
            if rewrite_count_values
            else 0.0
        ),
        "no_hint_metadata_hard_law_filter_applied_count": metadata_hard_law_filter_applied_count,
        "no_hint_unique_laws_retrieved_avg": (
            sum(unique_laws_retrieved_values) / len(unique_laws_retrieved_values)
            if unique_laws_retrieved_values
            else 0.0
        ),
        "no_hint_unique_laws_in_context_avg": (
            sum(unique_laws_in_context_values) / len(unique_laws_in_context_values)
            if unique_laws_in_context_values
            else 0.0
        ),
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
        "run_id": index_contract.get("run_id"),
        "collection_points_count": index_contract.get("collection_points_count"),
        "eval_reference_coverage": index_contract.get("eval_reference_coverage"),
        "qdrant_url_used": config_payload.get("qdrant_url"),
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

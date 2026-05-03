"""Oracle-context evaluation pipeline orchestration."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from legal_rag.laws_preprocessing.common import normalize_ws

from .env import load_env_file, resolve_env_file
from .io import (
    now_utc,
    prepare_tmp_output_dir,
    read_json,
    read_jsonl,
    replace_output_dir,
    sha256_file,
    write_json,
    write_jsonl,
)
from .llm import StructuredChatClient, UtopiaStructuredChatClient, resolve_ollama_chat_url
from .models import (
    DEFAULT_CHAT_MODEL,
    ORACLE_CONTEXT_SCHEMA_VERSION,
    JudgeOutput,
    McqAnswerOutput,
    McqResultRow,
    NoHintAnswerOutput,
    NoHintResultRow,
    OracleContextRecord,
    OracleEvaluationConfig,
)
from .prompts import build_judge_prompt, build_mcq_prompt, build_no_hint_prompt, schema_dict
from .references import OracleReferenceResolver
from .scoring import add_delta, aggregate_results, score_mcq_label

T = TypeVar("T")
U = TypeVar("U")


def load_inputs(config: OracleEvaluationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load clean evaluation records and manifests."""
    evaluation_dir = Path(config.evaluation_dir)
    laws_dir = Path(config.laws_dir)
    mcq_records = read_jsonl(evaluation_dir / "questions_mcq.jsonl")
    no_hint_records = read_jsonl(evaluation_dir / "questions_no_hint.jsonl")
    evaluation_manifest = read_json(evaluation_dir / "evaluation_manifest.json")
    laws_manifest = read_json(laws_dir / "manifest.json")
    return mcq_records, no_hint_records, evaluation_manifest, laws_manifest


def select_records(records: list[dict[str, Any]], config: OracleEvaluationConfig) -> list[dict[str, Any]]:
    """Apply start/limit selection to a stable record list."""
    start = int(config.start)
    limit = config.effective_limit
    if limit is None:
        return records[start:]
    return records[start : start + int(limit)]


def validate_mcq_no_hint_alignment(mcq_records: list[dict[str, Any]], no_hint_records: list[dict[str, Any]]) -> None:
    """Ensure the four runs can operate over the same question IDs."""
    if len(mcq_records) != len(no_hint_records):
        raise ValueError(f"MCQ/no-hint record counts differ: {len(mcq_records)} != {len(no_hint_records)}")
    for mcq, no_hint in zip(mcq_records, no_hint_records):
        qid = str(mcq.get("qid") or "")
        if qid != str(no_hint.get("qid") or ""):
            raise ValueError(f"{qid}: MCQ/no-hint qid mismatch")
        if qid != str(no_hint.get("linked_mcq_qid") or ""):
            raise ValueError(f"{qid}: no-hint linked_mcq_qid mismatch")
        if normalize_ws(str(mcq.get("question_stem") or "")) != normalize_ws(str(no_hint.get("question") or "")):
            raise ValueError(f"{qid}: MCQ stem and no-hint question do not match")


def build_oracle_contexts(records: list[dict[str, Any]], laws_dir: str | Path) -> list[OracleContextRecord]:
    """Resolve oracle contexts for the selected no-hint records."""
    resolver = OracleReferenceResolver.from_dir(laws_dir)
    return [resolver.build_context_record(record) for record in records]


def resolve_utopia_runtime(config: OracleEvaluationConfig) -> dict[str, Any]:
    """Resolve Utopia connection settings without exposing secret values."""
    loaded_env = load_env_file(config.env_file)
    resolved_env_file = resolve_env_file(config.env_file)
    api_key = config.api_key or os.getenv("UTOPIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("UTOPIA_API_KEY is missing for oracle-context evaluation")
    explicit_url = config.api_url or os.getenv("UTOPIA_OLLAMA_CHAT_URL", "")
    base_url = os.getenv("UTOPIA_BASE_URL", config.base_url)
    api_url = resolve_ollama_chat_url(base_url, explicit_url=explicit_url)
    return {
        "api_url": api_url,
        "base_url": base_url,
        "api_key": api_key,
        "api_key_present": bool(api_key),
        "env_file": str(resolved_env_file) if resolved_env_file else None,
        "env_file_loaded": bool(loaded_env),
        "env_keys_loaded": sorted(key for key in loaded_env if not key.endswith("API_KEY")),
    }


def create_default_client(config: OracleEvaluationConfig) -> UtopiaStructuredChatClient:
    """Create the default remote client from config and environment."""
    runtime = resolve_utopia_runtime(config)
    return UtopiaStructuredChatClient(
        api_url=runtime["api_url"],
        api_key=runtime["api_key"],
        retry_attempts=config.retry_attempts,
    )


def resolve_answer_model(config: OracleEvaluationConfig) -> str:
    """Resolve the answer model after .env loading."""
    if config.chat_model and config.chat_model != DEFAULT_CHAT_MODEL:
        return config.chat_model
    return os.getenv("UTOPIA_CHAT_MODEL", config.chat_model)


def resolve_judge_model(config: OracleEvaluationConfig, answer_model: str) -> str:
    """Resolve the judge model after .env loading."""
    if config.judge_model:
        return config.judge_model
    return os.getenv("UTOPIA_JUDGE_MODEL", answer_model)


def _parallel_map_ordered(items: list[T], func: Callable[[T], U], *, max_workers: int) -> list[U]:
    """Run independent tasks in parallel while preserving input order."""
    if max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    workers = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))


def run_mcq(
    *,
    records: list[dict[str, Any]],
    contexts_by_qid: dict[str, OracleContextRecord],
    client: StructuredChatClient,
    config: OracleEvaluationConfig,
    use_context: bool,
) -> list[dict[str, Any]]:
    """Run one MCQ condition and return row-level results."""
    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        qid = str(record["qid"])
        context = contexts_by_qid[qid]
        context_text = context.context_text if use_context else None
        context_article_ids = context.context_article_ids if use_context else []
        predicted_label: str | None = None
        score: int | None = None
        error: str | None = context.error if use_context and context.error else None
        if error is None:
            try:
                call = client.structured_chat(
                    prompt=build_mcq_prompt(record, context_text=context_text),
                    model=config.chat_model,
                    payload_schema=schema_dict(McqAnswerOutput),
                    timeout_seconds=config.timeout_seconds,
                )
                answer = McqAnswerOutput.model_validate(call["structured"])
                predicted_label = answer.answer_label
                score, error = score_mcq_label(predicted_label, str(record["correct_label"]))
            except Exception as exc:
                error = f"mcq_structured_error: {type(exc).__name__}: {exc}"
        return McqResultRow(
            qid=qid,
            level=str(record["level"]),
            question=str(record["question_stem"]),
            options=dict(record["options"]),
            correct_label=str(record["correct_label"]),
            predicted_label=predicted_label,
            score=score,
            context_article_ids=context_article_ids,
            error=error,
        ).to_json_record()

    return _parallel_map_ordered(records, run_one, max_workers=config.max_concurrency)


def run_no_hint(
    *,
    records: list[dict[str, Any]],
    contexts_by_qid: dict[str, OracleContextRecord],
    client: StructuredChatClient,
    config: OracleEvaluationConfig,
    use_context: bool,
) -> list[dict[str, Any]]:
    """Run one no-hint condition and judge each answer on a 0-2 scale."""
    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        qid = str(record["qid"])
        context = contexts_by_qid[qid]
        context_text = context.context_text if use_context else None
        context_article_ids = context.context_article_ids if use_context else []
        predicted_answer: str | None = None
        judge_score: int | None = None
        judge_explanation: str | None = None
        error: str | None = context.error if use_context and context.error else None
        if error is None:
            try:
                answer_call = client.structured_chat(
                    prompt=build_no_hint_prompt(record, context_text=context_text),
                    model=config.chat_model,
                    payload_schema=schema_dict(NoHintAnswerOutput),
                    timeout_seconds=config.timeout_seconds,
                )
                answer = NoHintAnswerOutput.model_validate(answer_call["structured"])
                predicted_answer = answer.answer_text
            except Exception as exc:
                error = f"no_hint_structured_error: {type(exc).__name__}: {exc}"
        if error is None:
            try:
                judge_call = client.structured_chat(
                    prompt=build_judge_prompt(record, predicted_answer),
                    model=config.resolved_judge_model,
                    payload_schema=schema_dict(JudgeOutput),
                    timeout_seconds=config.timeout_seconds,
                )
                judge = JudgeOutput.model_validate(judge_call["structured"])
                judge_score = int(judge.score)
                judge_explanation = judge.explanation
            except ValidationError as exc:
                error = f"judge_structured_error: {type(exc).__name__}: {exc}"
            except Exception as exc:
                error = f"judge_structured_error: {type(exc).__name__}: {exc}"
        return NoHintResultRow(
            qid=qid,
            level=str(record["level"]),
            question=str(record["question"]),
            predicted_answer=predicted_answer,
            correct_answer=str(record["correct_answer"]),
            judge_score=judge_score,
            judge_explanation=judge_explanation,
            context_article_ids=context_article_ids,
            error=error,
        ).to_json_record()

    return _parallel_map_ordered(records, run_one, max_workers=config.max_concurrency)


def build_summary(
    *,
    mcq_no_context: list[dict[str, Any]],
    mcq_oracle_context: list[dict[str, Any]],
    no_hint_no_context: list[dict[str, Any]],
    no_hint_oracle_context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build all aggregate metrics and context deltas."""
    mcq_no = aggregate_results("mcq_no_context", mcq_no_context, score_key="score", max_score_per_row=1)
    mcq_ctx = aggregate_results("mcq_oracle_context", mcq_oracle_context, score_key="score", max_score_per_row=1)
    no_hint_no = aggregate_results(
        "no_hint_no_context",
        no_hint_no_context,
        score_key="judge_score",
        max_score_per_row=2,
    )
    no_hint_ctx = aggregate_results(
        "no_hint_oracle_context",
        no_hint_oracle_context,
        score_key="judge_score",
        max_score_per_row=2,
    )
    return {
        "mcq_no_context": mcq_no,
        "mcq_oracle_context": mcq_ctx,
        "no_hint_no_context": no_hint_no,
        "no_hint_oracle_context": no_hint_ctx,
        "delta_oracle_minus_no_context": {
            "mcq": add_delta(mcq_no, mcq_ctx),
            "no_hint": add_delta(no_hint_no, no_hint_ctx),
        },
    }


def build_quality_report(
    *,
    contexts: list[OracleContextRecord],
    summary: dict[str, Any],
) -> str:
    """Render a concise human-readable quality report."""
    context_errors = [ctx for ctx in contexts if ctx.error]
    total_refs = sum(len(ctx.expected_references) for ctx in contexts)
    resolved_refs = sum(len(ctx.resolved_references) for ctx in contexts)
    lines = [
        "# 02b - Oracle Context Evaluation Quality Report",
        "",
        f"- Context rows: {len(contexts)}",
        f"- Expected article references: {total_refs}",
        f"- Resolved article references: {resolved_refs}",
        f"- Context errors: {len(context_errors)}",
        f"- Context coverage: {(resolved_refs / total_refs) if total_refs else None}",
        "",
        "## Run Metrics",
    ]
    for name in ("mcq_no_context", "mcq_oracle_context", "no_hint_no_context", "no_hint_oracle_context"):
        metrics = summary[name]
        lines.append(
            f"- `{name}`: processed={metrics['processed']}, judged={metrics['judged']}, "
            f"accuracy={metrics['accuracy']}, strict_accuracy={metrics['strict_accuracy']}, errors={metrics['errors']}"
        )
    if context_errors:
        lines.extend(["", "## Context Errors"])
        for ctx in context_errors[:20]:
            lines.append(f"- `{ctx.qid}`: {ctx.error}")
    return "\n".join(lines) + "\n"


def run_oracle_context_evaluation(
    config: OracleEvaluationConfig | None = None,
    *,
    client: StructuredChatClient | None = None,
) -> dict[str, Any]:
    """Run the complete controlled oracle-context evaluation pipeline."""
    cfg = config or OracleEvaluationConfig()
    if not isinstance(cfg, OracleEvaluationConfig):
        cfg = OracleEvaluationConfig.model_validate(cfg)
    runtime_connection: dict[str, Any] | None = None
    if client is None:
        runtime_connection = resolve_utopia_runtime(cfg)
        remote_client = UtopiaStructuredChatClient(
            api_url=runtime_connection["api_url"],
            api_key=runtime_connection["api_key"],
            retry_attempts=cfg.retry_attempts,
        )
    else:
        remote_client = client
    answer_model = resolve_answer_model(cfg)
    judge_model = resolve_judge_model(cfg, answer_model)
    effective_cfg = cfg.model_copy(update={"chat_model": answer_model, "judge_model": judge_model})

    mcq_all, no_hint_all, evaluation_manifest, laws_manifest = load_inputs(effective_cfg)
    mcq_records = select_records(mcq_all, effective_cfg)
    no_hint_records = select_records(no_hint_all, effective_cfg)
    validate_mcq_no_hint_alignment(mcq_records, no_hint_records)
    contexts = build_oracle_contexts(no_hint_records, effective_cfg.laws_dir)
    contexts_by_qid = {ctx.qid: ctx for ctx in contexts}

    output_dir = Path(cfg.output_dir)
    tmp_dir = prepare_tmp_output_dir(output_dir)
    try:
        mcq_no_context = run_mcq(
            records=mcq_records,
            contexts_by_qid=contexts_by_qid,
            client=remote_client,
            config=effective_cfg,
            use_context=False,
        )
        mcq_oracle_context = run_mcq(
            records=mcq_records,
            contexts_by_qid=contexts_by_qid,
            client=remote_client,
            config=effective_cfg,
            use_context=True,
        )
        no_hint_no_context = run_no_hint(
            records=no_hint_records,
            contexts_by_qid=contexts_by_qid,
            client=remote_client,
            config=effective_cfg,
            use_context=False,
        )
        no_hint_oracle_context = run_no_hint(
            records=no_hint_records,
            contexts_by_qid=contexts_by_qid,
            client=remote_client,
            config=effective_cfg,
            use_context=True,
        )
        summary = build_summary(
            mcq_no_context=mcq_no_context,
            mcq_oracle_context=mcq_oracle_context,
            no_hint_no_context=no_hint_no_context,
            no_hint_oracle_context=no_hint_oracle_context,
        )

        context_rows = [ctx.to_json_record() for ctx in contexts]
        files = {
            "source_truth_contexts": "source_truth_contexts.jsonl",
            "mcq_no_context_results": "mcq_no_context_results.jsonl",
            "mcq_oracle_context_results": "mcq_oracle_context_results.jsonl",
            "no_hint_no_context_results": "no_hint_no_context_results.jsonl",
            "no_hint_oracle_context_results": "no_hint_oracle_context_results.jsonl",
            "oracle_context_summary": "oracle_context_summary.json",
            "quality_report": "quality_report.md",
            "oracle_context_manifest": "oracle_context_manifest.json",
        }
        write_jsonl(tmp_dir / files["source_truth_contexts"], context_rows)
        write_jsonl(tmp_dir / files["mcq_no_context_results"], mcq_no_context)
        write_jsonl(tmp_dir / files["mcq_oracle_context_results"], mcq_oracle_context)
        write_jsonl(tmp_dir / files["no_hint_no_context_results"], no_hint_no_context)
        write_jsonl(tmp_dir / files["no_hint_oracle_context_results"], no_hint_oracle_context)
        write_json(tmp_dir / files["oracle_context_summary"], summary)
        (tmp_dir / files["quality_report"]).write_text(
            build_quality_report(contexts=contexts, summary=summary),
            encoding="utf-8",
        )

        output_hashes = {
            key: sha256_file(tmp_dir / filename)
            for key, filename in files.items()
            if key != "oracle_context_manifest"
        }
        source_hashes = {
            "questions_mcq": sha256_file(Path(effective_cfg.evaluation_dir) / "questions_mcq.jsonl"),
            "questions_no_hint": sha256_file(Path(effective_cfg.evaluation_dir) / "questions_no_hint.jsonl"),
            "evaluation_manifest": sha256_file(Path(effective_cfg.evaluation_dir) / "evaluation_manifest.json"),
            "laws": sha256_file(Path(effective_cfg.laws_dir) / "laws.jsonl"),
            "articles": sha256_file(Path(effective_cfg.laws_dir) / "articles.jsonl"),
            "laws_manifest": sha256_file(Path(effective_cfg.laws_dir) / "manifest.json"),
        }
        manifest = {
            "schema_version": ORACLE_CONTEXT_SCHEMA_VERSION,
            "created_at": now_utc(),
            "config": effective_cfg.model_dump(mode="json"),
            "models": {
                "answer_model": effective_cfg.chat_model,
                "judge_model": effective_cfg.resolved_judge_model,
            },
            "connection": (
                {
                    key: value
                    for key, value in runtime_connection.items()
                    if key != "api_key"
                }
                if runtime_connection
                else {"client": "injected"}
            ),
            "prompt_version": effective_cfg.prompt_version,
            "source_hashes": source_hashes,
            "upstream_manifests": {
                "evaluation_schema_version": evaluation_manifest.get("schema_version"),
                "laws_schema_version": laws_manifest.get("schema_version"),
            },
            "counts": {
                "mcq": len(mcq_records),
                "no_hint": len(no_hint_records),
                "context_rows": len(contexts),
                "context_errors": sum(1 for ctx in contexts if ctx.error),
                "article_references": sum(len(ctx.expected_references) for ctx in contexts),
                "resolved_article_references": sum(len(ctx.resolved_references) for ctx in contexts),
            },
            "outputs": files,
            "output_hashes": output_hashes,
            "summary": summary,
            "manifest_hash_note": "oracle_context_manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
        }
        write_json(tmp_dir / files["oracle_context_manifest"], manifest)
        replace_output_dir(tmp_dir, output_dir)
        return manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

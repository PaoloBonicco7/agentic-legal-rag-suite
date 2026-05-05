"""No-RAG baseline pipeline orchestration."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from legal_rag.laws_preprocessing.common import normalize_ws
from legal_rag.oracle_context_evaluation.env import load_env_file, resolve_env_file
from legal_rag.oracle_context_evaluation.io import (
    now_utc,
    prepare_tmp_output_dir,
    read_json,
    read_jsonl,
    replace_output_dir,
    sha256_file,
    write_json,
    write_jsonl,
)
from legal_rag.oracle_context_evaluation.llm import StructuredChatClient, UtopiaStructuredChatClient, resolve_ollama_chat_url
from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL, JudgeOutput, McqAnswerOutput, NoHintAnswerOutput
from legal_rag.oracle_context_evaluation.scoring import aggregate_results, score_mcq_label

from .models import (
    NO_RAG_SCHEMA_VERSION,
    McqResultRow,
    NoHintResultRow,
    NoRagConfig,
    NoRagConnectionRecord,
    NoRagManifest,
    NoRagModelIdentities,
    NoRagOutputFiles,
    NoRagRunCounts,
    NoRagSummary,
    SafeNoRagConfigRecord,
)
from .prompts import build_judge_prompt, build_mcq_prompt, build_no_hint_prompt, schema_dict

T = TypeVar("T")
U = TypeVar("U")


def load_inputs(config: NoRagConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load clean evaluation records and manifest."""
    evaluation_dir = Path(config.evaluation_dir)
    mcq_records = read_jsonl(evaluation_dir / "questions_mcq.jsonl")
    no_hint_records = read_jsonl(evaluation_dir / "questions_no_hint.jsonl")
    evaluation_manifest = read_json(evaluation_dir / "evaluation_manifest.json")
    return mcq_records, no_hint_records, evaluation_manifest


def select_records(records: list[dict[str, Any]], config: NoRagConfig) -> list[dict[str, Any]]:
    """Apply start/benchmark-size selection to a stable record list."""
    start = int(config.start)
    limit = config.effective_benchmark_size
    if limit is None:
        return records[start:]
    return records[start : start + int(limit)]


def validate_mcq_no_hint_alignment(mcq_records: list[dict[str, Any]], no_hint_records: list[dict[str, Any]]) -> None:
    """Ensure MCQ and no-hint rows refer to the same questions."""
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


def resolve_utopia_runtime(config: NoRagConfig) -> dict[str, Any]:
    """Resolve Utopia connection settings without exposing secret values."""
    loaded_env = load_env_file(config.env_file)
    resolved_env_file = resolve_env_file(config.env_file)
    api_key = config.api_key or os.getenv("UTOPIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("UTOPIA_API_KEY is missing for no-RAG baseline evaluation")
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


def resolve_answer_model(config: NoRagConfig) -> str:
    """Resolve the answer model after .env loading."""
    if config.chat_model and config.chat_model != DEFAULT_CHAT_MODEL:
        return config.chat_model
    return os.getenv("UTOPIA_CHAT_MODEL", config.chat_model)


def resolve_judge_model(config: NoRagConfig, answer_model: str) -> str:
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
    client: StructuredChatClient,
    config: NoRagConfig,
) -> list[dict[str, Any]]:
    """Run no-RAG MCQ answering and return row-level results."""

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        predicted_label: str | None = None
        score: int | None = None
        error: str | None = None
        try:
            call = client.structured_chat(
                prompt=build_mcq_prompt(record),
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
            qid=str(record["qid"]),
            level=str(record["level"]),
            question=str(record["question_stem"]),
            options=dict(record["options"]),
            predicted_label=predicted_label,
            correct_label=str(record["correct_label"]),
            score=score,
            error=error,
        ).to_json_record()

    return _parallel_map_ordered(records, run_one, max_workers=config.max_concurrency)


def run_no_hint(
    *,
    records: list[dict[str, Any]],
    client: StructuredChatClient,
    config: NoRagConfig,
) -> list[dict[str, Any]]:
    """Run no-RAG open answering and judge each successful answer."""

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        predicted_answer: str | None = None
        judge_score: int | None = None
        judge_explanation: str | None = None
        error: str | None = None
        try:
            answer_call = client.structured_chat(
                prompt=build_no_hint_prompt(record),
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
                    prompt=build_judge_prompt(record, predicted_answer or ""),
                    model=config.resolved_judge_model,
                    payload_schema=schema_dict(JudgeOutput),
                    timeout_seconds=config.timeout_seconds,
                )
                judge = JudgeOutput.model_validate(judge_call["structured"])
                judge_score = int(judge.score)
                judge_explanation = judge.explanation
            except ValidationError as exc:
                error = f"judge_error: {type(exc).__name__}: {exc}"
            except Exception as exc:
                error = f"judge_error: {type(exc).__name__}: {exc}"
        return NoHintResultRow(
            qid=str(record["qid"]),
            level=str(record["level"]),
            question=str(record["question"]),
            predicted_answer=predicted_answer,
            correct_answer=str(record["correct_answer"]),
            judge_score=judge_score,
            judge_explanation=judge_explanation,
            error=error,
        ).to_json_record()

    return _parallel_map_ordered(records, run_one, max_workers=config.max_concurrency)


def build_summary(*, mcq_results: list[dict[str, Any]], no_hint_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics for the no-RAG baseline."""
    return {
        "mcq": aggregate_results("mcq", mcq_results, score_key="score", max_score_per_row=1),
        "no_hint": aggregate_results("no_hint", no_hint_results, score_key="judge_score", max_score_per_row=2),
    }


def build_quality_report(*, summary: dict[str, Any], mcq_results: list[dict[str, Any]], no_hint_results: list[dict[str, Any]]) -> str:
    """Render a concise human-readable quality report."""
    lines = [
        "# 04 - No-RAG Baseline Quality Report",
        "",
        "## Run Metrics",
    ]
    for name in ("mcq", "no_hint"):
        metrics = summary[name]
        lines.append(
            f"- `{name}`: processed={metrics['processed']}, judged={metrics['judged']}, "
            f"accuracy={metrics['accuracy']}, strict_accuracy={metrics['strict_accuracy']}, errors={metrics['errors']}"
        )
    errors = [
        (row["qid"], row["error"])
        for row in [*mcq_results, *no_hint_results]
        if row.get("error")
    ]
    if errors:
        lines.extend(["", "## Errors"])
        for qid, error in errors[:20]:
            lines.append(f"- `{qid}`: {error}")
    return "\n".join(lines) + "\n"


def _safe_config_dump(config: NoRagConfig, *, api_key_present: bool | None = None) -> dict[str, Any]:
    """Serialize runtime config without secret values."""
    data = config.model_dump(mode="json")
    data.pop("api_key", None)
    data["api_key_present"] = bool(config.api_key) if api_key_present is None else api_key_present
    return SafeNoRagConfigRecord.model_validate(data).to_json_record()


def _safe_connection_dump(runtime_connection: dict[str, Any] | None) -> dict[str, Any]:
    """Serialize connection metadata without secret values."""
    if runtime_connection is None:
        return NoRagConnectionRecord(client="injected").to_json_record()
    connection = {key: value for key, value in runtime_connection.items() if key != "api_key"}
    connection["client"] = "utopia"
    return NoRagConnectionRecord.model_validate(connection).to_json_record()


def _source_hashes(config: NoRagConfig) -> dict[str, str]:
    """Hash the clean evaluation inputs used by the run."""
    evaluation_dir = Path(config.evaluation_dir)
    return {
        "questions_mcq": sha256_file(evaluation_dir / "questions_mcq.jsonl"),
        "questions_no_hint": sha256_file(evaluation_dir / "questions_no_hint.jsonl"),
        "evaluation_manifest": sha256_file(evaluation_dir / "evaluation_manifest.json"),
    }


def _output_hashes(tmp_dir: Path, files: NoRagOutputFiles) -> dict[str, str]:
    """Hash generated artifacts, excluding the self-referential manifest."""
    return {
        key: sha256_file(tmp_dir / filename)
        for key, filename in files.to_json_record().items()
        if key != "no_rag_manifest"
    }


def _build_manifest(
    *,
    config: NoRagConfig,
    runtime_connection: dict[str, Any] | None,
    evaluation_manifest: dict[str, Any],
    mcq_count: int,
    no_hint_count: int,
    files: NoRagOutputFiles,
    output_hashes: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate the run manifest."""
    api_key_present = bool(config.api_key)
    if runtime_connection is not None:
        api_key_present = bool(runtime_connection.get("api_key_present"))
    manifest = NoRagManifest(
        schema_version=NO_RAG_SCHEMA_VERSION,
        created_at=now_utc(),
        config=SafeNoRagConfigRecord.model_validate(_safe_config_dump(config, api_key_present=api_key_present)),
        models=NoRagModelIdentities(
            answer_model=config.chat_model,
            judge_model=config.resolved_judge_model,
        ),
        connection=NoRagConnectionRecord.model_validate(_safe_connection_dump(runtime_connection)),
        prompt_version=config.prompt_version,
        random_seed=config.random_seed,
        source_hashes=_source_hashes(config),
        upstream_manifests={
            "evaluation_schema_version": evaluation_manifest.get("schema_version"),
        },
        counts=NoRagRunCounts(
            mcq=mcq_count,
            no_hint=no_hint_count,
            mcq_errors=int(summary["mcq"]["errors"]),
            no_hint_errors=int(summary["no_hint"]["errors"]),
        ),
        outputs=files,
        output_hashes=output_hashes,
        summary=NoRagSummary.model_validate(summary),
        manifest_hash_note="no_rag_manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
    )
    return manifest.to_json_record()


def run_no_rag_baseline(
    config: NoRagConfig | None = None,
    *,
    client: StructuredChatClient | None = None,
) -> dict[str, Any]:
    """Run the complete no-RAG baseline pipeline."""
    cfg = config or NoRagConfig()
    if not isinstance(cfg, NoRagConfig):
        cfg = NoRagConfig.model_validate(cfg)
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

    mcq_all, no_hint_all, evaluation_manifest = load_inputs(effective_cfg)
    mcq_records = select_records(mcq_all, effective_cfg)
    no_hint_records = select_records(no_hint_all, effective_cfg)
    validate_mcq_no_hint_alignment(mcq_records, no_hint_records)

    output_dir = Path(cfg.output_dir)
    tmp_dir = prepare_tmp_output_dir(output_dir)
    try:
        mcq_results = run_mcq(records=mcq_records, client=remote_client, config=effective_cfg)
        no_hint_results = run_no_hint(records=no_hint_records, client=remote_client, config=effective_cfg)
        summary = build_summary(mcq_results=mcq_results, no_hint_results=no_hint_results)

        files = NoRagOutputFiles()
        write_jsonl(tmp_dir / files.mcq_results, mcq_results)
        write_jsonl(tmp_dir / files.no_hint_results, no_hint_results)
        write_json(tmp_dir / files.no_rag_summary, summary)
        (tmp_dir / files.quality_report).write_text(
            build_quality_report(summary=summary, mcq_results=mcq_results, no_hint_results=no_hint_results),
            encoding="utf-8",
        )

        manifest = _build_manifest(
            config=effective_cfg,
            runtime_connection=runtime_connection,
            evaluation_manifest=evaluation_manifest,
            mcq_count=len(mcq_records),
            no_hint_count=len(no_hint_records),
            files=files,
            output_hashes=_output_hashes(tmp_dir, files),
            summary=summary,
        )
        write_json(tmp_dir / files.no_rag_manifest, manifest)
        replace_output_dir(tmp_dir, output_dir)
        return manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

"""Simple RAG baseline pipeline orchestration."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

from pydantic import ValidationError
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding, build_embedder
from legal_rag.indexing.models import IndexingConfig
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
from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL, JudgeOutput
from legal_rag.oracle_context_evaluation.scoring import aggregate_results, score_mcq_label

from .models import (
    SIMPLE_RAG_SCHEMA_VERSION,
    Citation,
    McqResultRow,
    NoHintResultRow,
    RetrievedChunkRecord,
    SimpleMcqAnswerOutput,
    SimpleNoHintAnswerOutput,
    SimpleRagConfig,
)
from .prompts import build_mcq_prompt, build_no_hint_prompt, format_context_chunks
from .retrieval import connect_qdrant, load_index_manifest, resolve_collection_name, search_dense

T = TypeVar("T")
U = TypeVar("U")


def load_inputs(config: SimpleRagConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load clean evaluation records and manifest."""
    evaluation_dir = Path(config.evaluation_dir)
    mcq_records = read_jsonl(evaluation_dir / "questions_mcq.jsonl")
    no_hint_records = read_jsonl(evaluation_dir / "questions_no_hint.jsonl")
    evaluation_manifest = read_json(evaluation_dir / "evaluation_manifest.json")
    return mcq_records, no_hint_records, evaluation_manifest


def select_records(records: list[dict[str, Any]], config: SimpleRagConfig) -> list[dict[str, Any]]:
    """Apply start/benchmark-size selection to a stable record list."""
    limit = config.effective_benchmark_size
    if limit is None:
        return records[config.start :]
    return records[config.start : config.start + int(limit)]


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


def resolve_utopia_runtime(config: SimpleRagConfig) -> dict[str, Any]:
    """Resolve Utopia connection settings without exposing secret values."""
    loaded_env = load_env_file(config.env_file)
    resolved_env_file = resolve_env_file(config.env_file)
    api_key = config.api_key or os.getenv("UTOPIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("UTOPIA_API_KEY is missing for simple RAG evaluation")
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


def resolve_answer_model(config: SimpleRagConfig) -> str:
    """Resolve the answer model after .env loading."""
    if config.chat_model and config.chat_model != DEFAULT_CHAT_MODEL:
        return config.chat_model
    return os.getenv("UTOPIA_CHAT_MODEL", config.chat_model)


def resolve_judge_model(config: SimpleRagConfig, answer_model: str) -> str:
    """Resolve the judge model after .env loading."""
    if config.judge_model:
        return config.judge_model
    return os.getenv("UTOPIA_JUDGE_MODEL", answer_model)


def build_query_embedder(config: SimpleRagConfig, index_manifest: dict[str, Any]) -> SupportsEmbedding:
    """Build the query embedder from the index manifest and local environment."""
    load_env_file(config.env_file)
    embedding = dict(index_manifest.get("embedding") or {})
    manifest_config = dict(index_manifest.get("config") or {})
    provider = str(embedding.get("backend") or embedding.get("provider") or manifest_config.get("embedding_backend") or manifest_config.get("embedding_provider") or "utopia")
    if provider != "utopia":
        raise RuntimeError(f"Unsupported simple RAG embedding provider from index manifest: {provider!r}")
    model = str(
        embedding.get("resolved_model")
        or embedding.get("configured_model")
        or embedding.get("model")
        or manifest_config.get("resolved_embedding_model")
        or manifest_config.get("embedding_model")
        or ""
    )
    if not model:
        raise RuntimeError("Index manifest does not record an embedding model for query embedding")
    data = {
        "embedding_backend": provider,
        "embedding_model": model,
        "embedding_api_key": config.api_key or os.getenv("UTOPIA_API_KEY", ""),
        "utopia_base_url": os.getenv("UTOPIA_BASE_URL", manifest_config.get("utopia_base_url", "https://utopia.hpc4ai.unito.it/api")),
        "utopia_embed_url": os.getenv(
            "UTOPIA_EMBED_URL",
            manifest_config.get("utopia_embed_url", "https://utopia.hpc4ai.unito.it/ollama/api/embeddings"),
        ),
        "utopia_embed_api_mode": str(embedding.get("mode") or manifest_config.get("utopia_embed_api_mode") or "ollama"),
        "batch_size": int(manifest_config.get("batch_size") or 64),
        "embedding_timeout_seconds": float(manifest_config.get("embedding_timeout_seconds") or 60.0),
        "hybrid_enabled": False,
        "env_file": config.env_file,
    }
    return build_embedder(IndexingConfig.model_validate(data))


def _format_seconds(value: float) -> str:
    if value < 60:
        return f"{value:.0f}s"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m{seconds:02d}s"


def _progress_message(label: str, *, done: int, total: int, errors: int, started: float) -> str:
    elapsed = perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    return (
        f"[{label}] {done}/{total} ({done / total:.0%}) "
        f"errors={errors} elapsed={_format_seconds(elapsed)} eta={_format_seconds(remaining)}"
    )


def _result_has_error(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("error"))


def _parallel_map_ordered(
    items: list[T],
    func: Callable[[T], U],
    *,
    max_workers: int,
    progress_label: str | None = None,
    progress_interval: int = 5,
) -> list[U]:
    """Run independent tasks in parallel while preserving input order."""
    total = len(items)
    if total == 0:
        return []
    started = perf_counter()
    errors = 0

    def report(done: int) -> None:
        if progress_label and (done == 1 or done == total or done % progress_interval == 0):
            print(_progress_message(progress_label, done=done, total=total, errors=errors, started=started), flush=True)

    if max_workers <= 1 or len(items) <= 1:
        results: list[U] = []
        for item in items:
            result = func(item)
            results.append(result)
            errors += int(_result_has_error(result))
            report(len(results))
        return results
    workers = min(max_workers, len(items))
    results: list[U | None] = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            done += 1
            errors += int(_result_has_error(result))
            report(done)
    return [result for result in results]  # type: ignore[misc]


def build_context(
    retrieved: list[RetrievedChunkRecord],
    *,
    max_context_chunks: int,
    max_context_chars: int,
) -> tuple[list[RetrievedChunkRecord], str]:
    """Build a bounded context from retrieved chunks."""
    selected: list[RetrievedChunkRecord] = []
    text_overrides: dict[str, str] = {}
    used_chars = 0
    for chunk in retrieved:
        if len(selected) >= max_context_chunks:
            break
        text = chunk.text
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            if selected:
                break
            text_overrides[chunk.chunk_id] = text[:remaining]
            selected.append(chunk)
            used_chars += remaining
            break
        selected.append(chunk)
        used_chars += len(text)
    return selected, format_context_chunks(selected, text_overrides=text_overrides)


def _retrieved_law_ids(chunks: list[RetrievedChunkRecord]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        law_id = str(chunk.payload.get("law_id") or "")
        if law_id and law_id not in seen:
            out.append(law_id)
            seen.add(law_id)
    return out


def _build_citations(citation_ids: list[str], context_chunks: list[RetrievedChunkRecord]) -> tuple[list[Citation], list[str]]:
    by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    citations: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for chunk_id in citation_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = by_id.get(chunk_id)
        if chunk is None:
            invalid.append(chunk_id)
            continue
        citations.append(
            Citation(
                law_id=str(chunk.payload.get("law_id") or ""),
                article_id=str(chunk.payload.get("article_id") or ""),
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.text,
            )
        )
    return citations, invalid


def _join_error(current: str | None, new_error: str | None) -> str | None:
    if not new_error:
        return current
    return f"{current}; {new_error}" if current else new_error


def _base_trace(
    retrieved: list[RetrievedChunkRecord],
    context_chunks: list[RetrievedChunkRecord],
) -> dict[str, Any]:
    return {
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved],
        "retrieved_law_ids": _retrieved_law_ids(retrieved),
        "context_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
        "retrieved_count": len(retrieved),
        "context_count": len(context_chunks),
    }


def run_mcq(
    *,
    records: list[dict[str, Any]],
    llm_client: StructuredChatClient,
    qdrant_client: QdrantClient,
    embedder: SupportsEmbedding,
    collection_name: str,
    config: SimpleRagConfig,
) -> list[dict[str, Any]]:
    """Run simple-RAG MCQ answering and return row-level results."""

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        retrieved: list[RetrievedChunkRecord] = []
        context_chunks: list[RetrievedChunkRecord] = []
        predicted_label: str | None = None
        answer_text: str | None = None
        citations: list[Citation] = []
        score: int | None = None
        error: str | None = None
        try:
            retrieved = search_dense(
                qdrant_client,
                collection_name=collection_name,
                embedder=embedder,
                query_text=str(record["question_stem"]),
                limit=config.top_k,
                static_filters=config.static_filters,
            )
            context_chunks, context_text = build_context(
                retrieved,
                max_context_chunks=config.max_context_chunks,
                max_context_chars=config.max_context_chars,
            )
            if not retrieved:
                error = "empty_retrieval"
            elif not context_chunks:
                error = "empty_context"
            else:
                call = llm_client.structured_chat(
                    prompt=build_mcq_prompt(record, context_text),
                    model=config.chat_model,
                    payload_schema=SimpleMcqAnswerOutput.model_json_schema(),
                    timeout_seconds=config.timeout_seconds,
                )
                answer = SimpleMcqAnswerOutput.model_validate(call["structured"])
                predicted_label = answer.answer_label
                answer_text = str(record["options"].get(predicted_label) or "")
                citations, invalid_citations = _build_citations(answer.citation_chunk_ids, context_chunks)
                if invalid_citations:
                    error = _join_error(error, f"citation_error: invalid_chunk_ids={invalid_citations}")
                score, score_error = score_mcq_label(predicted_label, str(record["correct_label"]))
                error = _join_error(error, score_error)
        except Exception as exc:
            error = _join_error(error, f"mcq_structured_error: {type(exc).__name__}: {exc}")
        trace = _base_trace(retrieved, context_chunks)
        return McqResultRow(
            qid=str(record["qid"]),
            level=str(record["level"]),
            question=str(record["question_stem"]),
            answer=answer_text,
            citations=citations,
            options=dict(record["options"]),
            correct_label=str(record["correct_label"]),
            predicted_label=predicted_label,
            score=score,
            error=error,
            **trace,
        ).to_json_record()

    return _parallel_map_ordered(
        records,
        run_one,
        max_workers=config.max_concurrency,
        progress_label="simple-rag mcq" if config.show_progress else None,
        progress_interval=config.progress_interval,
    )


def run_no_hint(
    *,
    records: list[dict[str, Any]],
    llm_client: StructuredChatClient,
    qdrant_client: QdrantClient,
    embedder: SupportsEmbedding,
    collection_name: str,
    config: SimpleRagConfig,
) -> list[dict[str, Any]]:
    """Run simple-RAG open answering and judge each generated answer."""

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        retrieved: list[RetrievedChunkRecord] = []
        context_chunks: list[RetrievedChunkRecord] = []
        predicted_answer: str | None = None
        citations: list[Citation] = []
        judge_score: int | None = None
        judge_explanation: str | None = None
        error: str | None = None
        try:
            retrieved = search_dense(
                qdrant_client,
                collection_name=collection_name,
                embedder=embedder,
                query_text=str(record["question"]),
                limit=config.top_k,
                static_filters=config.static_filters,
            )
            context_chunks, context_text = build_context(
                retrieved,
                max_context_chunks=config.max_context_chunks,
                max_context_chars=config.max_context_chars,
            )
            if not retrieved:
                error = "empty_retrieval"
            elif not context_chunks:
                error = "empty_context"
            else:
                answer_call = llm_client.structured_chat(
                    prompt=build_no_hint_prompt(record, context_text),
                    model=config.chat_model,
                    payload_schema=SimpleNoHintAnswerOutput.model_json_schema(),
                    timeout_seconds=config.timeout_seconds,
                )
                answer = SimpleNoHintAnswerOutput.model_validate(answer_call["structured"])
                predicted_answer = answer.answer_text
                citations, invalid_citations = _build_citations(answer.citation_chunk_ids, context_chunks)
                if invalid_citations:
                    error = _join_error(error, f"citation_error: invalid_chunk_ids={invalid_citations}")
        except Exception as exc:
            error = _join_error(error, f"no_hint_structured_error: {type(exc).__name__}: {exc}")
        if predicted_answer is not None:
            try:
                judge_call = llm_client.structured_chat(
                    prompt=_build_judge_prompt(record, predicted_answer),
                    model=config.resolved_judge_model,
                    payload_schema=JudgeOutput.model_json_schema(),
                    timeout_seconds=config.timeout_seconds,
                )
                judge = JudgeOutput.model_validate(judge_call["structured"])
                judge_score = int(judge.score)
                judge_explanation = judge.explanation
            except ValidationError as exc:
                error = _join_error(error, f"judge_error: {type(exc).__name__}: {exc}")
            except Exception as exc:
                error = _join_error(error, f"judge_error: {type(exc).__name__}: {exc}")
        trace = _base_trace(retrieved, context_chunks)
        return NoHintResultRow(
            qid=str(record["qid"]),
            level=str(record["level"]),
            question=str(record["question"]),
            answer=predicted_answer,
            citations=citations,
            predicted_answer=predicted_answer,
            correct_answer=str(record["correct_answer"]),
            judge_score=judge_score,
            judge_explanation=judge_explanation,
            error=error,
            **trace,
        ).to_json_record()

    return _parallel_map_ordered(
        records,
        run_one,
        max_workers=config.max_concurrency,
        progress_label="simple-rag no-hint" if config.show_progress else None,
        progress_interval=config.progress_interval,
    )


def _build_judge_prompt(record: dict[str, Any], predicted_answer: str) -> str:
    candidate = predicted_answer.strip() if predicted_answer.strip() else "[EMPTY]"
    return (
        "You are an impartial semantic judge for Italian legal QA.\n"
        "Score the model answer against the official correct answer.\n\n"
        "Rubric:\n"
        "- score=2: correct or semantically equivalent.\n"
        "- score=1: partially correct, incomplete, and not contradictory.\n"
        "- score=0: wrong, contradictory, empty, ambiguous, or not evaluable.\n\n"
        "Return only valid JSON matching this schema:\n"
        f"{JudgeOutput.model_json_schema()}\n\n"
        "Question:\n"
        f"{record['question']}\n\n"
        "Official correct answer:\n"
        f"{record['correct_answer']}\n\n"
        "Model answer to judge:\n"
        f"{candidate}"
    )


def build_summary(*, mcq_results: list[dict[str, Any]], no_hint_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics for the simple RAG baseline."""
    return {
        "mcq": aggregate_results("mcq", mcq_results, score_key="score", max_score_per_row=1),
        "no_hint": aggregate_results("no_hint", no_hint_results, score_key="judge_score", max_score_per_row=2),
    }


def build_quality_report(*, summary: dict[str, Any], mcq_results: list[dict[str, Any]], no_hint_results: list[dict[str, Any]]) -> str:
    """Render a concise human-readable quality report."""
    all_rows = [*mcq_results, *no_hint_results]
    error_counts = {
        "empty_retrieval": sum(1 for row in all_rows if row.get("error") and "empty_retrieval" in row["error"]),
        "generation_errors": sum(1 for row in all_rows if row.get("error") and "structured_error" in row["error"]),
        "judge_errors": sum(1 for row in all_rows if row.get("error") and "judge_error" in row["error"]),
        "citation_errors": sum(1 for row in all_rows if row.get("error") and "citation_error" in row["error"]),
    }
    avg_retrieved = (sum(int(row["retrieved_count"]) for row in all_rows) / len(all_rows)) if all_rows else 0.0
    avg_context = (sum(int(row["context_count"]) for row in all_rows) / len(all_rows)) if all_rows else 0.0
    lines = [
        "# 05 - Simple RAG Quality Report",
        "",
        "## Run Metrics",
    ]
    for name in ("mcq", "no_hint"):
        metrics = summary[name]
        lines.append(
            f"- `{name}`: processed={metrics['processed']}, judged={metrics['judged']}, "
            f"accuracy={metrics['accuracy']}, strict_accuracy={metrics['strict_accuracy']}, errors={metrics['errors']}"
        )
    lines.extend(
        [
            "",
            "## Retrieval and Context",
            f"- average_retrieved_count={avg_retrieved}",
            f"- average_context_count={avg_context}",
            f"- empty_retrieval={error_counts['empty_retrieval']}",
            "",
            "## Error Counts",
        ]
    )
    for key, count in error_counts.items():
        lines.append(f"- `{key}`: {count}")
    errors = [(row["qid"], row["error"]) for row in all_rows if row.get("error")]
    if errors:
        lines.extend(["", "## Errors"])
        for qid, error in errors[:20]:
            lines.append(f"- `{qid}`: {error}")
    return "\n".join(lines) + "\n"


def _safe_config_dump(config: SimpleRagConfig) -> dict[str, Any]:
    """Serialize runtime config without secret values."""
    data = config.model_dump(mode="json")
    data.pop("api_key", None)
    data["api_key_present"] = bool(config.api_key)
    return data


def run_simple_rag(
    config: SimpleRagConfig | dict[str, Any] | None = None,
    *,
    client: StructuredChatClient | None = None,
    qdrant_client: QdrantClient | None = None,
    embedder: SupportsEmbedding | None = None,
) -> dict[str, Any]:
    """Run the complete simple RAG baseline pipeline."""
    cfg = config if isinstance(config, SimpleRagConfig) else SimpleRagConfig.model_validate(config or {})
    runtime_connection: dict[str, Any] | None = None
    if client is None:
        runtime_connection = resolve_utopia_runtime(cfg)
        llm_client = UtopiaStructuredChatClient(
            api_url=runtime_connection["api_url"],
            api_key=runtime_connection["api_key"],
            retry_attempts=cfg.retry_attempts,
        )
    else:
        llm_client = client
        load_env_file(cfg.env_file)

    answer_model = resolve_answer_model(cfg)
    judge_model = resolve_judge_model(cfg, answer_model)
    effective_cfg = cfg.model_copy(update={"chat_model": answer_model, "judge_model": judge_model})

    index_manifest_path, index_manifest = load_index_manifest(effective_cfg)
    collection_name = resolve_collection_name(effective_cfg, index_manifest)
    if effective_cfg.show_progress:
        print(f"[simple-rag] index_manifest={index_manifest_path}", flush=True)
        print(f"[simple-rag] collection={collection_name}", flush=True)
    owned_qdrant_client = qdrant_client is None
    if qdrant_client is None:
        qdrant_client = connect_qdrant(effective_cfg, index_manifest)
    if embedder is None:
        embedder = build_query_embedder(effective_cfg, index_manifest)

    mcq_all, no_hint_all, evaluation_manifest = load_inputs(effective_cfg)
    mcq_records = select_records(mcq_all, effective_cfg)
    no_hint_records = select_records(no_hint_all, effective_cfg)
    validate_mcq_no_hint_alignment(mcq_records, no_hint_records)
    if effective_cfg.show_progress:
        print(
            f"[simple-rag] selected mcq={len(mcq_records)} no_hint={len(no_hint_records)} "
            f"concurrency={effective_cfg.max_concurrency}",
            flush=True,
        )

    output_dir = Path(effective_cfg.output_dir)
    tmp_dir = prepare_tmp_output_dir(output_dir)
    started = perf_counter()
    try:
        if effective_cfg.show_progress:
            print("[simple-rag] starting MCQ run", flush=True)
        mcq_results = run_mcq(
            records=mcq_records,
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name=collection_name,
            config=effective_cfg,
        )
        if effective_cfg.show_progress:
            print("[simple-rag] starting no-hint run", flush=True)
        no_hint_results = run_no_hint(
            records=no_hint_records,
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name=collection_name,
            config=effective_cfg,
        )
        summary = build_summary(mcq_results=mcq_results, no_hint_results=no_hint_results)

        files = {
            "mcq_results": "mcq_results.jsonl",
            "no_hint_results": "no_hint_results.jsonl",
            "simple_rag_summary": "simple_rag_summary.json",
            "quality_report": "quality_report.md",
            "simple_rag_manifest": "simple_rag_manifest.json",
        }
        write_jsonl(tmp_dir / files["mcq_results"], mcq_results)
        write_jsonl(tmp_dir / files["no_hint_results"], no_hint_results)
        write_json(tmp_dir / files["simple_rag_summary"], summary)
        (tmp_dir / files["quality_report"]).write_text(
            build_quality_report(summary=summary, mcq_results=mcq_results, no_hint_results=no_hint_results),
            encoding="utf-8",
        )

        output_hashes = {
            key: sha256_file(tmp_dir / filename)
            for key, filename in files.items()
            if key != "simple_rag_manifest"
        }
        source_hashes = {
            "questions_mcq": sha256_file(Path(effective_cfg.evaluation_dir) / "questions_mcq.jsonl"),
            "questions_no_hint": sha256_file(Path(effective_cfg.evaluation_dir) / "questions_no_hint.jsonl"),
            "evaluation_manifest": sha256_file(Path(effective_cfg.evaluation_dir) / "evaluation_manifest.json"),
            "index_manifest": sha256_file(index_manifest_path),
        }
        all_rows = [*mcq_results, *no_hint_results]
        manifest = {
            "schema_version": SIMPLE_RAG_SCHEMA_VERSION,
            "created_at": now_utc(),
            "duration_seconds": perf_counter() - started,
            "config": _safe_config_dump(effective_cfg),
            "models": {
                "answer_model": effective_cfg.chat_model,
                "judge_model": effective_cfg.resolved_judge_model,
                "embedding_model": getattr(embedder, "model_name", None),
            },
            "connection": (
                {key: value for key, value in runtime_connection.items() if key != "api_key"}
                if runtime_connection
                else {"client": "injected"}
            ),
            "prompt_version": effective_cfg.prompt_version,
            "collection_name": collection_name,
            "index_manifest_path": str(index_manifest_path),
            "source_hashes": source_hashes,
            "upstream_manifests": {
                "evaluation_schema_version": evaluation_manifest.get("schema_version"),
                "index_schema_version": index_manifest.get("schema_version"),
                "index_ready_for_retrieval": index_manifest.get("ready_for_retrieval"),
            },
            "counts": {
                "mcq": len(mcq_records),
                "no_hint": len(no_hint_records),
                "mcq_errors": summary["mcq"]["errors"],
                "no_hint_errors": summary["no_hint"]["errors"],
                "empty_retrieval": sum(1 for row in all_rows if row.get("error") and "empty_retrieval" in row["error"]),
                "citation_errors": sum(1 for row in all_rows if row.get("error") and "citation_error" in row["error"]),
            },
            "outputs": files,
            "output_hashes": output_hashes,
            "summary": summary,
            "manifest_hash_note": "simple_rag_manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
        }
        write_json(tmp_dir / files["simple_rag_manifest"], manifest)
        replace_output_dir(tmp_dir, output_dir)
        if effective_cfg.show_progress:
            print(f"[simple-rag] completed in {_format_seconds(float(manifest['duration_seconds']))}", flush=True)
            print(f"[simple-rag] output_dir={output_dir}", flush=True)
        return manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    finally:
        if owned_qdrant_client and qdrant_client is not None:
            close = getattr(qdrant_client, "close", None)
            if callable(close):
                close()

"""Advanced graph-aware RAG pipeline orchestration."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

from pydantic import ValidationError
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding
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
from legal_rag.oracle_context_evaluation.references import OracleReferenceResolver, split_reference_values
from legal_rag.oracle_context_evaluation.scoring import aggregate_results, score_mcq_label
from legal_rag.simple_rag.models import Citation, RetrievedChunkRecord, SimpleRagConfig
from legal_rag.simple_rag.prompts import format_context_chunks
from legal_rag.simple_rag.runner import build_query_embedder

from .models import (
    ADVANCED_RAG_SCHEMA_VERSION,
    AdvancedMcqAnswerOutput,
    AdvancedMcqResultRow,
    AdvancedNoHintAnswerOutput,
    AdvancedNoHintResultRow,
    AdvancedRagConfig,
    FailureCategory,
    GraphRelationUsed,
    RerankOutput,
    RetrievalTrace,
)
from .prompts import build_mcq_prompt, build_no_hint_prompt, build_rerank_prompt
from .retrieval import (
    GraphIndex,
    connect_qdrant,
    embed_sparse_query,
    expand_with_graph,
    load_index_manifest,
    load_simple_rag_manifest,
    manifest_disables_sparse,
    resolve_collection_name,
    resolve_qdrant_target,
    search_dense,
    search_hybrid,
    sparse_vector_name,
)

T = TypeVar("T")
U = TypeVar("U")


def load_inputs(config: AdvancedRagConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load clean evaluation records and manifest."""
    evaluation_dir = Path(config.evaluation_dir)
    return (
        read_jsonl(evaluation_dir / "questions_mcq.jsonl"),
        read_jsonl(evaluation_dir / "questions_no_hint.jsonl"),
        read_json(evaluation_dir / "evaluation_manifest.json"),
    )


def select_records(records: list[dict[str, Any]], config: AdvancedRagConfig) -> list[dict[str, Any]]:
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


def resolve_utopia_runtime(config: AdvancedRagConfig) -> dict[str, Any]:
    """Resolve Utopia connection settings without exposing secret values."""
    loaded_env = load_env_file(config.env_file)
    resolved_env_file = resolve_env_file(config.env_file)
    api_key = config.api_key or os.getenv("UTOPIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("UTOPIA_API_KEY is missing for advanced RAG evaluation")
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


def resolve_answer_model(config: AdvancedRagConfig) -> str:
    """Resolve the answer model after .env loading."""
    if config.chat_model and config.chat_model != DEFAULT_CHAT_MODEL:
        return config.chat_model
    return os.getenv("UTOPIA_CHAT_MODEL", config.chat_model)


def resolve_judge_model(config: AdvancedRagConfig, answer_model: str) -> str:
    """Resolve the judge model after .env loading."""
    if config.judge_model:
        return config.judge_model
    return os.getenv("UTOPIA_JUDGE_MODEL", answer_model)


def build_advanced_query_embedder(config: AdvancedRagConfig, index_manifest: dict[str, Any]) -> SupportsEmbedding:
    """Build the query embedder from the index manifest and local environment."""
    return build_query_embedder(_simple_config(config), index_manifest)


def validate_simple_baseline(
    *,
    config: AdvancedRagConfig,
    simple_manifest: dict[str, Any],
    index_manifest_path: Path,
) -> None:
    """Ensure the advanced run is comparable with its simple RAG baseline."""
    simple_hashes = dict(simple_manifest.get("source_hashes") or {})
    expected = {
        "questions_mcq": sha256_file(Path(config.evaluation_dir) / "questions_mcq.jsonl"),
        "questions_no_hint": sha256_file(Path(config.evaluation_dir) / "questions_no_hint.jsonl"),
        "evaluation_manifest": sha256_file(Path(config.evaluation_dir) / "evaluation_manifest.json"),
        "index_manifest": sha256_file(index_manifest_path),
    }
    missing = [key for key in expected if key not in simple_hashes]
    if missing:
        raise RuntimeError(f"Simple RAG manifest is missing source hashes: {missing}")
    mismatches = [key for key, value in expected.items() if simple_hashes.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Simple RAG baseline is not comparable; hash mismatches: {mismatches}")


def validate_hybrid_ready(
    *,
    config: AdvancedRagConfig,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    index_manifest: dict[str, Any],
) -> None:
    """Fail before row processing when hybrid retrieval is enabled without sparse support."""
    if not config.hybrid_enabled:
        return
    if sparse_vector_name(qdrant_client, collection_name=collection_name) is None or manifest_disables_sparse(index_manifest):
        raise RuntimeError("hybrid_enabled=True requires a Qdrant collection with sparse vectors")
    embed_sparse_query(embedder, "hybrid sparse support probe")


def validate_qdrant_collection_ready(
    *,
    qdrant_client: QdrantClient,
    collection_name: str,
    qdrant_target: dict[str, str | None],
) -> None:
    """Fail before row processing when the configured Qdrant collection is unavailable."""
    target = _qdrant_target_label(qdrant_target)
    try:
        collections_response = qdrant_client.get_collections()
    except Exception as exc:
        hint = _qdrant_start_hint(qdrant_target)
        raise RuntimeError(
            f"Qdrant preflight failed for target {target}: {type(exc).__name__}: {exc}.{hint}"
        ) from exc

    collection_names = sorted(str(item.name) for item in getattr(collections_response, "collections", []))
    if collection_name not in collection_names:
        hint = _qdrant_start_hint(qdrant_target)
        raise RuntimeError(
            "Qdrant collection not found before advanced RAG run: "
            f"target={target}, collection={collection_name!r}, available_collections={collection_names}.{hint}"
        )

    try:
        points_count = int(qdrant_client.count(collection_name=collection_name, exact=True).count)
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant preflight count failed for target {target}, collection={collection_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if points_count <= 0:
        raise RuntimeError(
            f"Qdrant collection is empty before advanced RAG run: target={target}, collection={collection_name!r}, points_count={points_count}"
        )


def _qdrant_target_label(qdrant_target: dict[str, str | None]) -> str:
    if qdrant_target.get("url"):
        return str(qdrant_target["url"])
    if qdrant_target.get("path"):
        return str(qdrant_target["path"])
    return "injected-client"


def _qdrant_start_hint(qdrant_target: dict[str, str | None]) -> str:
    url = str(qdrant_target.get("url") or "")
    if "127.0.0.1:6333" in url or "localhost:6333" in url:
        return " Start Qdrant with: docker compose -f docker-compose.qdrant.yml up -d qdrant"
    return ""


def _parallel_map_ordered(items: list[T], func: Callable[[T], U], *, max_workers: int) -> list[U]:
    """Run independent tasks in parallel while preserving input order."""
    if max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    workers = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))


def retrieve_candidates(
    *,
    record: dict[str, Any],
    question_key: str,
    llm_client: StructuredChatClient,
    qdrant_client: QdrantClient,
    embedder: SupportsEmbedding,
    collection_name: str,
    index_manifest: dict[str, Any],
    graph: GraphIndex,
    config: AdvancedRagConfig,
) -> RetrievalTrace:
    """Retrieve, expand and optionally rerank candidates for one question."""
    question = str(record[question_key])
    metadata_filters = config.active_static_filters
    retrieval_mode = "hybrid" if config.hybrid_enabled else "dense"
    if config.hybrid_enabled:
        retrieved = search_hybrid(
            qdrant_client,
            collection_name=collection_name,
            embedder=embedder,
            query_text=question,
            limit=config.top_k,
            rrf_k=config.rrf_k,
            static_filters=metadata_filters,
            index_manifest=index_manifest,
        )
    else:
        retrieved = search_dense(
            qdrant_client,
            collection_name=collection_name,
            embedder=embedder,
            query_text=question,
            limit=config.top_k,
            static_filters=metadata_filters,
        )

    expanded: list[RetrievedChunkRecord] = []
    relations: list[GraphRelationUsed] = []
    if config.graph_expansion_enabled:
        expanded, relations = expand_with_graph(
            qdrant_client,
            collection_name=collection_name,
            graph=graph,
            seeds=retrieved[: config.graph_expansion_seed_k],
            relation_types=config.graph_expansion_relation_types,
            static_filters=metadata_filters,
            max_chunks_per_law=config.max_chunks_per_expanded_law,
        )

    candidates = _dedupe_chunks([*retrieved, *expanded])
    rerank_scores: list[int] = []
    if config.rerank_enabled and candidates:
        try:
            reranked, rerank_scores = rerank_candidates(
                llm_client=llm_client,
                question=question,
                candidates=candidates[: config.rerank_input_k],
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(f"rerank_error: {type(exc).__name__}: {exc}") from exc
    else:
        reranked = candidates
    return RetrievalTrace(
        retrieved=retrieved,
        expanded=expanded,
        graph_relations_used=relations,
        reranked=reranked,
        rerank_scores=rerank_scores,
        retrieval_mode=retrieval_mode,
        metadata_filters=metadata_filters,
    )


def rerank_candidates(
    *,
    llm_client: StructuredChatClient,
    question: str,
    candidates: list[RetrievedChunkRecord],
    config: AdvancedRagConfig,
) -> tuple[list[RetrievedChunkRecord], list[int]]:
    """Rerank candidates with the LLM relevance scorer."""
    call = llm_client.structured_chat(
        prompt=build_rerank_prompt(question, candidates),
        model=config.resolved_judge_model,
        payload_schema=RerankOutput.model_json_schema(),
        timeout_seconds=config.timeout_seconds,
    )
    output = RerankOutput.model_validate(call["structured"])
    candidate_by_id = {chunk.chunk_id: chunk for chunk in candidates}
    original_rank = {chunk.chunk_id: idx for idx, chunk in enumerate(candidates)}
    scores_by_id = {item.chunk_id: int(item.score) for item in output.scores if item.chunk_id in candidate_by_id}
    scored = [(chunk, scores_by_id.get(chunk.chunk_id, 0)) for chunk in candidates]
    scored.sort(key=lambda item: (-item[1], original_rank[item[0].chunk_id]))
    kept = scored[: config.rerank_output_k]
    return [chunk for chunk, _ in kept], [score for _, score in kept]


def build_context(
    reranked: list[RetrievedChunkRecord],
    *,
    max_context_chunks: int,
    max_context_chars: int,
) -> tuple[list[RetrievedChunkRecord], str]:
    """Build a bounded context from reranked chunks."""
    selected: list[RetrievedChunkRecord] = []
    text_overrides: dict[str, str] = {}
    used_chars = 0
    for chunk in reranked:
        if len(selected) >= max_context_chunks:
            break
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        if len(chunk.text) > remaining:
            if selected:
                break
            text_overrides[chunk.chunk_id] = chunk.text[:remaining]
            selected.append(chunk)
            used_chars += remaining
            break
        selected.append(chunk)
        used_chars += len(chunk.text)
    return selected, format_context_chunks(selected, text_overrides=text_overrides)


def run_mcq(
    *,
    records: list[dict[str, Any]],
    llm_client: StructuredChatClient,
    qdrant_client: QdrantClient,
    embedder: SupportsEmbedding,
    collection_name: str,
    index_manifest: dict[str, Any],
    graph: GraphIndex,
    reference_resolver: OracleReferenceResolver,
    config: AdvancedRagConfig,
) -> list[dict[str, Any]]:
    """Run advanced-RAG MCQ answering and return row-level results."""

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        trace = _empty_trace(config)
        context_chunks: list[RetrievedChunkRecord] = []
        predicted_label: str | None = None
        answer_text: str | None = None
        citations: list[Citation] = []
        score: int | None = None
        error: str | None = None
        try:
            retrieval = retrieve_candidates(
                record=record,
                question_key="question_stem",
                llm_client=llm_client,
                qdrant_client=qdrant_client,
                embedder=embedder,
                collection_name=collection_name,
                index_manifest=index_manifest,
                graph=graph,
                config=config,
            )
            context_chunks, context_text = build_context(
                retrieval.reranked,
                max_context_chunks=config.rerank_output_k,
                max_context_chars=config.max_context_chars,
            )
            trace = _trace(record=record, retrieval=retrieval, context_chunks=context_chunks, config=config, resolver=reference_resolver)
            if not retrieval.retrieved:
                error = "empty_retrieval"
            elif not context_chunks:
                error = "empty_context"
            else:
                call = llm_client.structured_chat(
                    prompt=build_mcq_prompt(record, context_text),
                    model=config.chat_model,
                    payload_schema=AdvancedMcqAnswerOutput.model_json_schema(),
                    timeout_seconds=config.timeout_seconds,
                )
                answer = AdvancedMcqAnswerOutput.model_validate(call["structured"])
                predicted_label = answer.answer_label
                answer_text = str(record["options"].get(predicted_label) or "")
                citations, invalid_citations = _build_citations(answer.citation_chunk_ids, context_chunks)
                if invalid_citations:
                    error = _join_error(error, f"citation_error: invalid_chunk_ids={invalid_citations}")
                score, score_error = score_mcq_label(predicted_label, str(record["correct_label"]))
                error = _join_error(error, score_error)
        except ValidationError as exc:
            error = _join_error(error, f"generation_error: {type(exc).__name__}: {exc}")
        except Exception as exc:
            error = _join_error(error, f"generation_error: {type(exc).__name__}: {exc}")
        failure_category = _failure_category(error=error, score=score, reference_law_hit=trace["reference_law_hit"], answer=answer_text)
        return AdvancedMcqResultRow(
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
            failure_category=failure_category,
            **trace,
        ).to_json_record()

    return _parallel_map_ordered(records, run_one, max_workers=config.max_concurrency)


def run_no_hint(
    *,
    records: list[dict[str, Any]],
    llm_client: StructuredChatClient,
    qdrant_client: QdrantClient,
    embedder: SupportsEmbedding,
    collection_name: str,
    index_manifest: dict[str, Any],
    graph: GraphIndex,
    reference_resolver: OracleReferenceResolver,
    config: AdvancedRagConfig,
) -> list[dict[str, Any]]:
    """Run advanced-RAG open answering and judge each generated answer."""

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        trace = _empty_trace(config)
        context_chunks: list[RetrievedChunkRecord] = []
        predicted_answer: str | None = None
        citations: list[Citation] = []
        judge_score: int | None = None
        judge_explanation: str | None = None
        error: str | None = None
        try:
            retrieval = retrieve_candidates(
                record=record,
                question_key="question",
                llm_client=llm_client,
                qdrant_client=qdrant_client,
                embedder=embedder,
                collection_name=collection_name,
                index_manifest=index_manifest,
                graph=graph,
                config=config,
            )
            context_chunks, context_text = build_context(
                retrieval.reranked,
                max_context_chunks=config.rerank_output_k,
                max_context_chars=config.max_context_chars,
            )
            trace = _trace(record=record, retrieval=retrieval, context_chunks=context_chunks, config=config, resolver=reference_resolver)
            if not retrieval.retrieved:
                error = "empty_retrieval"
            elif not context_chunks:
                error = "empty_context"
            else:
                answer_call = llm_client.structured_chat(
                    prompt=build_no_hint_prompt(record, context_text),
                    model=config.chat_model,
                    payload_schema=AdvancedNoHintAnswerOutput.model_json_schema(),
                    timeout_seconds=config.timeout_seconds,
                )
                answer = AdvancedNoHintAnswerOutput.model_validate(answer_call["structured"])
                predicted_answer = answer.answer_text
                citations, invalid_citations = _build_citations(answer.citation_chunk_ids, context_chunks)
                if invalid_citations:
                    error = _join_error(error, f"citation_error: invalid_chunk_ids={invalid_citations}")
        except ValidationError as exc:
            error = _join_error(error, f"generation_error: {type(exc).__name__}: {exc}")
        except Exception as exc:
            error = _join_error(error, f"generation_error: {type(exc).__name__}: {exc}")
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
        failure_category = _failure_category(error=error, score=judge_score, reference_law_hit=trace["reference_law_hit"], answer=predicted_answer)
        return AdvancedNoHintResultRow(
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
            failure_category=failure_category,
            **trace,
        ).to_json_record()

    return _parallel_map_ordered(records, run_one, max_workers=config.max_concurrency)


def build_summary(*, mcq_results: list[dict[str, Any]], no_hint_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics for the advanced RAG run."""
    return {
        "mcq": aggregate_results("mcq", mcq_results, score_key="score", max_score_per_row=1),
        "no_hint": aggregate_results("no_hint", no_hint_results, score_key="judge_score", max_score_per_row=2),
    }


def build_diagnostics(*, mcq_results: list[dict[str, Any]], no_hint_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-feature counters and failure diagnostics."""
    rows = [*mcq_results, *no_hint_results]
    rerank_scores = [int(score) for row in rows for score in row.get("rerank_scores", [])]
    failure_counts = Counter(str(row.get("failure_category") or "none") for row in rows)
    relation_counts = Counter(
        str(relation.get("relation_type") or "")
        for row in rows
        for relation in row.get("graph_relations_used", [])
    )
    return {
        "processed_rows": len(rows),
        "metadata_filtered_rows": sum(1 for row in rows if row.get("metadata_filters")),
        "hybrid_rows": sum(1 for row in rows if row.get("retrieval_mode") == "hybrid"),
        "dense_rows": sum(1 for row in rows if row.get("retrieval_mode") == "dense"),
        "graph_expanded_rows": sum(1 for row in rows if row.get("graph_expanded_chunk_ids")),
        "graph_expanded_chunks": sum(len(row.get("graph_expanded_chunk_ids", [])) for row in rows),
        "graph_relations_used": sum(len(row.get("graph_relations_used", [])) for row in rows),
        "graph_relation_type_counts": dict(sorted(relation_counts.items())),
        "reranked_rows": sum(1 for row in rows if row.get("rerank_scores")),
        "rerank_score_distribution": dict(sorted(Counter(rerank_scores).items())),
        "reference_law_hits": sum(1 for row in rows if row.get("reference_law_hit")),
        "failure_category_counts": dict(sorted(failure_counts.items())),
    }


def build_quality_report(
    *,
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
    mcq_results: list[dict[str, Any]],
    no_hint_results: list[dict[str, Any]],
) -> str:
    """Render a concise human-readable quality report."""
    lines = [
        "# 06 - Advanced Graph RAG Quality Report",
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
            "## Diagnostics",
            f"- metadata_filtered_rows={diagnostics['metadata_filtered_rows']}",
            f"- hybrid_rows={diagnostics['hybrid_rows']}",
            f"- graph_expanded_rows={diagnostics['graph_expanded_rows']}",
            f"- reranked_rows={diagnostics['reranked_rows']}",
            f"- reference_law_hits={diagnostics['reference_law_hits']}",
            "",
            "## Failure Categories",
        ]
    )
    for category, count in diagnostics["failure_category_counts"].items():
        lines.append(f"- `{category}`: {count}")
    errors = [(row["qid"], row["error"]) for row in [*mcq_results, *no_hint_results] if row.get("error")]
    if errors:
        lines.extend(["", "## Errors"])
        for qid, error in errors[:20]:
            lines.append(f"- `{qid}`: {error}")
    return "\n".join(lines) + "\n"


def run_advanced_graph_rag(
    config: AdvancedRagConfig | dict[str, Any] | None = None,
    *,
    client: StructuredChatClient | None = None,
    qdrant_client: QdrantClient | None = None,
    embedder: SupportsEmbedding | None = None,
) -> dict[str, Any]:
    """Run the complete advanced graph-aware RAG pipeline."""
    cfg = config if isinstance(config, AdvancedRagConfig) else AdvancedRagConfig.model_validate(config or {})
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
    simple_manifest_path, simple_manifest = load_simple_rag_manifest(effective_cfg)
    validate_simple_baseline(config=effective_cfg, simple_manifest=simple_manifest, index_manifest_path=index_manifest_path)
    collection_name = resolve_collection_name(effective_cfg, index_manifest)
    qdrant_target = resolve_qdrant_target(effective_cfg, index_manifest)
    owned_qdrant_client = qdrant_client is None
    if qdrant_client is None:
        qdrant_client = connect_qdrant(effective_cfg, index_manifest)
    validate_qdrant_collection_ready(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        qdrant_target=qdrant_target if owned_qdrant_client else {"mode": "injected", "url": None, "path": "injected-client"},
    )
    if embedder is None:
        embedder = build_advanced_query_embedder(effective_cfg, index_manifest)
    validate_hybrid_ready(
        config=effective_cfg,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embedder=embedder,
        index_manifest=index_manifest,
    )

    mcq_all, no_hint_all, evaluation_manifest = load_inputs(effective_cfg)
    mcq_records = select_records(mcq_all, effective_cfg)
    no_hint_records = select_records(no_hint_all, effective_cfg)
    validate_mcq_no_hint_alignment(mcq_records, no_hint_records)
    graph = GraphIndex.from_dir(effective_cfg.laws_dir)
    reference_resolver = OracleReferenceResolver.from_dir(effective_cfg.laws_dir)

    output_dir = Path(effective_cfg.output_dir)
    tmp_dir = prepare_tmp_output_dir(output_dir)
    started = perf_counter()
    try:
        mcq_results = run_mcq(
            records=mcq_records,
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name=collection_name,
            index_manifest=index_manifest,
            graph=graph,
            reference_resolver=reference_resolver,
            config=effective_cfg,
        )
        no_hint_results = run_no_hint(
            records=no_hint_records,
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            embedder=embedder,
            collection_name=collection_name,
            index_manifest=index_manifest,
            graph=graph,
            reference_resolver=reference_resolver,
            config=effective_cfg,
        )
        summary = build_summary(mcq_results=mcq_results, no_hint_results=no_hint_results)
        diagnostics = build_diagnostics(mcq_results=mcq_results, no_hint_results=no_hint_results)

        files = {
            "mcq_results": "mcq_results.jsonl",
            "no_hint_results": "no_hint_results.jsonl",
            "advanced_rag_summary": "advanced_rag_summary.json",
            "advanced_diagnostics": "advanced_diagnostics.json",
            "quality_report": "quality_report.md",
            "advanced_rag_manifest": "advanced_rag_manifest.json",
        }
        write_jsonl(tmp_dir / files["mcq_results"], mcq_results)
        write_jsonl(tmp_dir / files["no_hint_results"], no_hint_results)
        write_json(tmp_dir / files["advanced_rag_summary"], summary)
        write_json(tmp_dir / files["advanced_diagnostics"], diagnostics)
        (tmp_dir / files["quality_report"]).write_text(
            build_quality_report(
                summary=summary,
                diagnostics=diagnostics,
                mcq_results=mcq_results,
                no_hint_results=no_hint_results,
            ),
            encoding="utf-8",
        )

        output_hashes = {
            key: sha256_file(tmp_dir / filename)
            for key, filename in files.items()
            if key != "advanced_rag_manifest"
        }
        source_hashes = {
            "questions_mcq": sha256_file(Path(effective_cfg.evaluation_dir) / "questions_mcq.jsonl"),
            "questions_no_hint": sha256_file(Path(effective_cfg.evaluation_dir) / "questions_no_hint.jsonl"),
            "evaluation_manifest": sha256_file(Path(effective_cfg.evaluation_dir) / "evaluation_manifest.json"),
            "index_manifest": sha256_file(index_manifest_path),
            "simple_rag_manifest": sha256_file(simple_manifest_path),
            "edges": sha256_file(Path(effective_cfg.laws_dir) / "edges.jsonl"),
            "chunks": sha256_file(Path(effective_cfg.laws_dir) / "chunks.jsonl"),
        }
        manifest = {
            "schema_version": ADVANCED_RAG_SCHEMA_VERSION,
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
            "simple_rag_manifest_path": str(simple_manifest_path),
            "source_hashes": source_hashes,
            "upstream_manifests": {
                "evaluation_schema_version": evaluation_manifest.get("schema_version"),
                "index_schema_version": index_manifest.get("schema_version"),
                "simple_rag_schema_version": simple_manifest.get("schema_version"),
                "index_ready_for_retrieval": index_manifest.get("ready_for_retrieval"),
            },
            "counts": {
                "mcq": len(mcq_records),
                "no_hint": len(no_hint_records),
                "mcq_errors": summary["mcq"]["errors"],
                "no_hint_errors": summary["no_hint"]["errors"],
                **diagnostics,
            },
            "outputs": files,
            "output_hashes": output_hashes,
            "summary": summary,
            "diagnostics": diagnostics,
            "manifest_hash_note": "advanced_rag_manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
        }
        write_json(tmp_dir / files["advanced_rag_manifest"], manifest)
        replace_output_dir(tmp_dir, output_dir)
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


def _trace(
    *,
    record: dict[str, Any],
    retrieval: RetrievalTrace,
    context_chunks: list[RetrievedChunkRecord],
    config: AdvancedRagConfig,
    resolver: OracleReferenceResolver,
) -> dict[str, Any]:
    retrieved_plus_expanded = _dedupe_chunks([*retrieval.retrieved, *retrieval.expanded])
    expected_law_ids = _expected_law_ids(record, resolver)
    context_law_ids = {str(chunk.payload.get("law_id") or "") for chunk in context_chunks}
    return {
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_plus_expanded],
        "retrieved_law_ids": _unique(str(chunk.payload.get("law_id") or "") for chunk in retrieved_plus_expanded),
        "context_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
        "retrieved_count": len(retrieved_plus_expanded),
        "context_count": len(context_chunks),
        "metadata_filters": retrieval.metadata_filters,
        "retrieval_mode": retrieval.retrieval_mode,
        "graph_expanded_law_ids": _unique(str(chunk.payload.get("law_id") or "") for chunk in retrieval.expanded),
        "graph_expanded_chunk_ids": [chunk.chunk_id for chunk in retrieval.expanded],
        "graph_relations_used": retrieval.graph_relations_used,
        "reranked_chunk_ids": [chunk.chunk_id for chunk in retrieval.reranked],
        "rerank_scores": retrieval.rerank_scores if config.rerank_enabled else [],
        "context_included_count": len(context_chunks),
        "reference_law_hit": bool(expected_law_ids & context_law_ids),
    }


def _empty_trace(config: AdvancedRagConfig) -> dict[str, Any]:
    return {
        "retrieved_chunk_ids": [],
        "retrieved_law_ids": [],
        "context_chunk_ids": [],
        "retrieved_count": 0,
        "context_count": 0,
        "metadata_filters": config.active_static_filters,
        "retrieval_mode": "hybrid" if config.hybrid_enabled else "dense",
        "graph_expanded_law_ids": [],
        "graph_expanded_chunk_ids": [],
        "graph_relations_used": [],
        "reranked_chunk_ids": [],
        "rerank_scores": [],
        "context_included_count": 0,
        "reference_law_hit": False,
    }


def _expected_law_ids(record: dict[str, Any], resolver: OracleReferenceResolver) -> set[str]:
    out: set[str] = set()
    for reference in split_reference_values([str(value) for value in record.get("expected_references", [])]):
        try:
            out.add(resolver.resolve_reference(reference).law_id)
        except Exception:
            continue
    return out


def _failure_category(
    *,
    error: str | None,
    score: int | None,
    reference_law_hit: bool,
    answer: str | None,
) -> FailureCategory | None:
    answer_text = normalize_ws(str(answer or "")).lower()
    if not reference_law_hit and error and ("empty_retrieval" in error or "empty_context" in error):
        return "retrieval_miss"
    if error and ("judge_error" in error or "rerank" in error):
        return "judge_error"
    if error and ("structured_error" in error or "generation_error" in error or "citation_error" in error):
        return "generation_error"
    if score is not None and score > 0 and answer_text and not error:
        return None
    if not answer_text or "non lo so" in answer_text:
        return "abstention"
    if not reference_law_hit:
        return "retrieval_miss" if error and ("empty_retrieval" in error or "empty_context" in error) else "context_noise"
    return "unknown"


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


def _dedupe_chunks(chunks: list[RetrievedChunkRecord]) -> list[RetrievedChunkRecord]:
    out: list[RetrievedChunkRecord] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        out.append(chunk)
        seen.add(chunk.chunk_id)
    return out


def _unique(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _safe_config_dump(config: AdvancedRagConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    data.pop("api_key", None)
    data["api_key_present"] = bool(config.api_key)
    data["output_dir"] = config.output_dir
    return data


def _simple_config(config: AdvancedRagConfig) -> SimpleRagConfig:
    return SimpleRagConfig(
        evaluation_dir=config.evaluation_dir,
        index_dir=config.index_dir,
        index_manifest_path=config.index_manifest_path,
        collection_name=config.collection_name,
        env_file=config.env_file,
        api_url=config.api_url,
        api_key=config.api_key,
        base_url=config.base_url,
        chat_model=config.chat_model,
        judge_model=config.judge_model,
        timeout_seconds=config.timeout_seconds,
        start=config.start,
        benchmark_size=config.benchmark_size,
        smoke=config.smoke,
        retry_attempts=config.retry_attempts,
        max_concurrency=config.max_concurrency,
        static_filters=config.active_static_filters,
    )

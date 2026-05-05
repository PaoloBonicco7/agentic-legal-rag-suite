"""Interactive advanced graph-aware RAG helpers for local UI demos."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.env import load_env_file
from legal_rag.oracle_context_evaluation.llm import StructuredChatClient, UtopiaStructuredChatClient
from legal_rag.simple_rag.models import Citation, RetrievedChunkRecord

from .models import (
    AdvancedNoHintAnswerOutput,
    GraphRelationUsed,
    InteractiveRagConfig,
    InteractiveRagResult,
    InteractiveStepTiming,
)
from .prompts import build_no_hint_prompt
from .retrieval import (
    GraphIndex,
    connect_qdrant,
    dense_vector_name,
    expand_with_graph,
    load_index_manifest,
    manifest_disables_sparse,
    resolve_collection_name,
    search_dense,
    search_hybrid,
    sparse_vector_name,
)
from .runner import (
    _build_citations,
    _dedupe_chunks,
    build_advanced_query_embedder,
    build_context,
    rerank_candidates,
    resolve_answer_model,
    resolve_judge_model,
    resolve_utopia_runtime,
)

StepCallback = Callable[[str], None]


class InteractiveRagRuntime:
    """Reusable resources for single-question advanced RAG runs."""

    def __init__(
        self,
        config: InteractiveRagConfig | dict[str, Any] | None = None,
        *,
        client: StructuredChatClient | None = None,
        qdrant_client: QdrantClient | None = None,
        embedder: SupportsEmbedding | None = None,
    ) -> None:
        cfg = config if isinstance(config, InteractiveRagConfig) else InteractiveRagConfig.model_validate(config or {})
        if client is None:
            runtime_connection = resolve_utopia_runtime(cfg)
            self.llm_client = UtopiaStructuredChatClient(
                api_url=runtime_connection["api_url"],
                api_key=runtime_connection["api_key"],
                retry_attempts=cfg.retry_attempts,
            )
            self.connection = {key: value for key, value in runtime_connection.items() if key != "api_key"}
        else:
            load_env_file(cfg.env_file)
            self.llm_client = client
            self.connection = {"client": "injected"}

        answer_model = resolve_answer_model(cfg)
        judge_model = resolve_judge_model(cfg, answer_model)
        self.config = cfg.model_copy(update={"chat_model": answer_model, "judge_model": judge_model})
        self.index_manifest_path, self.index_manifest = load_index_manifest(self.config)
        self.collection_name = resolve_collection_name(self.config, self.index_manifest)
        self.owned_qdrant_client = qdrant_client is None
        self.qdrant_client = qdrant_client or connect_qdrant(self.config, self.index_manifest)
        self.embedder = embedder or build_advanced_query_embedder(self.config, self.index_manifest)
        self.graph = GraphIndex.from_dir(self.config.laws_dir)
        self.dense_vector_name = dense_vector_name(self.qdrant_client, collection_name=self.collection_name)
        self.sparse_vector_name = sparse_vector_name(self.qdrant_client, collection_name=self.collection_name)
        self.hybrid_available, self.hybrid_unavailable_reason = self._resolve_hybrid_availability()

    def answer_question(
        self,
        question: str,
        *,
        config: InteractiveRagConfig | dict[str, Any] | None = None,
        on_step: StepCallback | None = None,
    ) -> InteractiveRagResult:
        """Answer one free-form question and return the full retrieval trace."""
        started = perf_counter()
        cfg = self._effective_call_config(config)
        text = str(question or "").strip()
        if not text:
            return self._result(
                question=text,
                config=cfg,
                started=started,
                timing=InteractiveStepTiming(total_seconds=perf_counter() - started),
                error="empty_question",
            )
        if cfg.hybrid_enabled and not self.hybrid_available:
            return self._result(
                question=text,
                config=cfg,
                started=started,
                timing=InteractiveStepTiming(total_seconds=perf_counter() - started),
                error=f"hybrid_unavailable: {self.hybrid_unavailable_reason}",
            )

        retrieved: list[RetrievedChunkRecord] = []
        expanded: list[RetrievedChunkRecord] = []
        relations: list[GraphRelationUsed] = []
        reranked: list[RetrievedChunkRecord] = []
        rerank_scores: list[int] = []
        context_chunks: list[RetrievedChunkRecord] = []
        context_text = ""
        answer_text: str | None = None
        answer_rationale: str | None = None
        citations: list[Citation] = []
        invalid_citations: list[str] = []
        timing = InteractiveStepTiming()
        error: str | None = None

        try:
            _notify(on_step, "Retrieving initial candidates")
            step_started = perf_counter()
            retrieved = self._retrieve(text, cfg)
            timing.retrieval_seconds = perf_counter() - step_started

            _notify(on_step, "Expanding through explicit legal graph edges")
            step_started = perf_counter()
            if cfg.graph_expansion_enabled:
                expanded, relations = expand_with_graph(
                    self.qdrant_client,
                    collection_name=self.collection_name,
                    graph=self.graph,
                    seeds=retrieved[: cfg.graph_expansion_seed_k],
                    relation_types=cfg.graph_expansion_relation_types,
                    static_filters=cfg.active_static_filters,
                    max_chunks_per_law=cfg.max_chunks_per_expanded_law,
                )
            timing.graph_expansion_seconds = perf_counter() - step_started

            candidates = _dedupe_chunks([*retrieved, *expanded])
            _notify(on_step, "Reranking candidates")
            step_started = perf_counter()
            if cfg.rerank_enabled and candidates:
                reranked, rerank_scores = rerank_candidates(
                    llm_client=self.llm_client,
                    question=text,
                    candidates=candidates[: cfg.rerank_input_k],
                    config=cfg,
                )
            else:
                reranked = candidates
            timing.rerank_seconds = perf_counter() - step_started

            _notify(on_step, "Building bounded context")
            step_started = perf_counter()
            context_chunks, context_text = build_context(
                reranked,
                max_context_chunks=cfg.rerank_output_k,
                max_context_chars=cfg.max_context_chars,
            )
            timing.context_seconds = perf_counter() - step_started

            _notify(on_step, "Generating grounded answer")
            step_started = perf_counter()
            if not retrieved:
                error = "empty_retrieval"
            elif not context_chunks:
                error = "empty_context"
            else:
                answer_call = self.llm_client.structured_chat(
                    prompt=build_no_hint_prompt({"question": text}, context_text),
                    model=cfg.chat_model,
                    payload_schema=AdvancedNoHintAnswerOutput.model_json_schema(),
                    timeout_seconds=cfg.timeout_seconds,
                )
                answer = AdvancedNoHintAnswerOutput.model_validate(answer_call["structured"])
                answer_text = answer.answer_text
                answer_rationale = answer.short_rationale
                citations, invalid_citations = _build_citations(answer.citation_chunk_ids, context_chunks)
                if invalid_citations:
                    error = _join_error(error, f"citation_error: invalid_chunk_ids={invalid_citations}")
            timing.answer_seconds = perf_counter() - step_started
        except ValidationError as exc:
            error = _join_error(error, f"generation_error: {type(exc).__name__}: {exc}")
        except Exception as exc:
            error = _join_error(error, f"generation_error: {type(exc).__name__}: {exc}")
        finally:
            timing.total_seconds = perf_counter() - started

        return self._result(
            question=text,
            config=cfg,
            started=started,
            timing=timing,
            answer=answer_text,
            answer_rationale=answer_rationale,
            citations=citations,
            invalid_citation_chunk_ids=invalid_citations,
            retrieved=retrieved,
            expanded=expanded,
            graph_relations_used=relations,
            reranked=reranked,
            rerank_scores=rerank_scores if cfg.rerank_enabled else [],
            context_chunks=context_chunks,
            context_text=context_text,
            error=error,
        )

    def health(self) -> dict[str, Any]:
        """Return manifest and runtime details useful for UI diagnostics."""
        return {
            "collection_name": self.collection_name,
            "index_manifest_path": str(self.index_manifest_path),
            "index_schema_version": self.index_manifest.get("schema_version"),
            "index_ready_for_retrieval": self.index_manifest.get("ready_for_retrieval"),
            "embedding_model": getattr(self.embedder, "model_name", None),
            "chat_model": self.config.chat_model,
            "judge_model": self.config.resolved_judge_model,
            "dense_vector_name": self.dense_vector_name,
            "sparse_vector_name": self.sparse_vector_name,
            "hybrid_available": self.hybrid_available,
            "hybrid_unavailable_reason": self.hybrid_unavailable_reason,
            "connection": self.connection,
        }

    def close(self) -> None:
        """Close owned clients when the runtime is not cached anymore."""
        if self.owned_qdrant_client:
            close = getattr(self.qdrant_client, "close", None)
            if callable(close):
                close()

    def _retrieve(self, question: str, config: InteractiveRagConfig) -> list[RetrievedChunkRecord]:
        if config.hybrid_enabled:
            return search_hybrid(
                self.qdrant_client,
                collection_name=self.collection_name,
                embedder=self.embedder,
                query_text=question,
                limit=config.top_k,
                rrf_k=config.rrf_k,
                static_filters=config.active_static_filters,
                index_manifest=self.index_manifest,
            )
        return search_dense(
            self.qdrant_client,
            collection_name=self.collection_name,
            embedder=self.embedder,
            query_text=question,
            limit=config.top_k,
            static_filters=config.active_static_filters,
        )

    def _effective_call_config(self, config: InteractiveRagConfig | dict[str, Any] | None) -> InteractiveRagConfig:
        if config is None:
            return self.config
        cfg = config if isinstance(config, InteractiveRagConfig) else InteractiveRagConfig.model_validate(config)
        answer_model = resolve_answer_model(cfg)
        judge_model = resolve_judge_model(cfg, answer_model)
        return cfg.model_copy(update={"chat_model": answer_model, "judge_model": judge_model})

    def _resolve_hybrid_availability(self) -> tuple[bool, str | None]:
        if manifest_disables_sparse(self.index_manifest):
            return False, "index manifest declares sparse/hybrid support disabled"
        if self.sparse_vector_name is None:
            return False, "Qdrant collection does not expose sparse vectors"
        if not _embedder_supports_sparse(self.embedder):
            return False, "query embedder does not expose sparse embeddings"
        return True, None

    def _result(
        self,
        *,
        question: str,
        config: InteractiveRagConfig,
        started: float,
        timing: InteractiveStepTiming,
        answer: str | None = None,
        answer_rationale: str | None = None,
        citations: list[Citation] | None = None,
        invalid_citation_chunk_ids: list[str] | None = None,
        retrieved: list[RetrievedChunkRecord] | None = None,
        expanded: list[RetrievedChunkRecord] | None = None,
        graph_relations_used: list[GraphRelationUsed] | None = None,
        reranked: list[RetrievedChunkRecord] | None = None,
        rerank_scores: list[int] | None = None,
        context_chunks: list[RetrievedChunkRecord] | None = None,
        context_text: str = "",
        error: str | None = None,
    ) -> InteractiveRagResult:
        if timing.total_seconds <= 0:
            timing.total_seconds = perf_counter() - started
        return InteractiveRagResult(
            question=question,
            answer=answer,
            answer_rationale=answer_rationale,
            citations=citations or [],
            invalid_citation_chunk_ids=invalid_citation_chunk_ids or [],
            retrieved=retrieved or [],
            expanded=expanded or [],
            graph_relations_used=graph_relations_used or [],
            reranked=reranked or [],
            rerank_scores=rerank_scores or [],
            context_chunks=context_chunks or [],
            context_text=context_text,
            metadata_filters=config.active_static_filters,
            retrieval_mode="hybrid" if config.hybrid_enabled else "dense",
            hybrid_available=self.hybrid_available,
            hybrid_unavailable_reason=self.hybrid_unavailable_reason,
            dense_vector_name=self.dense_vector_name,
            sparse_vector_name=self.sparse_vector_name,
            collection_name=self.collection_name,
            flags={
                "metadata_filters_enabled": config.metadata_filters_enabled,
                "hybrid_enabled": config.hybrid_enabled,
                "graph_expansion_enabled": config.graph_expansion_enabled,
                "rerank_enabled": config.rerank_enabled,
            },
            parameters={
                "top_k": config.top_k,
                "rrf_k": config.rrf_k,
                "graph_expansion_seed_k": config.graph_expansion_seed_k,
                "max_chunks_per_expanded_law": config.max_chunks_per_expanded_law,
                "rerank_input_k": config.rerank_input_k,
                "rerank_output_k": config.rerank_output_k,
                "max_context_chars": config.max_context_chars,
            },
            timing=timing,
            error=error,
        )


def build_interactive_runtime(
    config: InteractiveRagConfig | dict[str, Any] | None = None,
    *,
    client: StructuredChatClient | None = None,
    qdrant_client: QdrantClient | None = None,
    embedder: SupportsEmbedding | None = None,
) -> InteractiveRagRuntime:
    """Build reusable resources for interactive advanced RAG."""
    return InteractiveRagRuntime(config, client=client, qdrant_client=qdrant_client, embedder=embedder)


def answer_interactive_question(
    question: str,
    config: InteractiveRagConfig | dict[str, Any] | None = None,
    *,
    client: StructuredChatClient | None = None,
    qdrant_client: QdrantClient | None = None,
    embedder: SupportsEmbedding | None = None,
    runtime: InteractiveRagRuntime | None = None,
    on_step: StepCallback | None = None,
) -> InteractiveRagResult:
    """Answer one question without writing batch run artifacts."""
    owned_runtime = runtime is None
    active_runtime = runtime or build_interactive_runtime(config, client=client, qdrant_client=qdrant_client, embedder=embedder)
    try:
        return active_runtime.answer_question(question, config=config, on_step=on_step)
    finally:
        if owned_runtime:
            active_runtime.close()


def _embedder_supports_sparse(embedder: SupportsEmbedding) -> bool:
    return callable(getattr(embedder, "embed_sparse_texts", None)) or callable(getattr(embedder, "sparse_embed_texts", None))


def _notify(callback: StepCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _join_error(current: str | None, new_error: str | None) -> str | None:
    if not new_error:
        return current
    return f"{current}; {new_error}" if current else new_error


__all__ = [
    "InteractiveRagRuntime",
    "StepCallback",
    "answer_interactive_question",
    "build_interactive_runtime",
]

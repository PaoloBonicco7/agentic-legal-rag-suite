from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from pydantic import ValidationError
from qdrant_client import QdrantClient

from legal_indexing.embeddings import SupportsEmbedding, build_embedder
from legal_indexing.io import DatasetValidationResult, validate_dataset
from legal_indexing.law_references import LawCatalog, build_law_catalog
from legal_indexing.qdrant_store import (
    collection_vector_capabilities,
    get_vector_size,
)

from .config import AdvancedRerankConfig, RagRuntimeConfig
from .context_builder import ContextBuildResult, build_context
from .graph_adapter import LegalGraphAdapter
from .index_contract import IndexContract, resolve_index_contract
from .llm import (
    SupportsInvoke,
    SupportsRunSync,
    build_chat_model,
    build_pydantic_ai_agent,
    is_empty_structured_answer,
    invoke_model,
    parse_structured_output,
    run_structured_with_agent,
)
from .metadata_filters import (
    MetadataFilterDecision,
    build_metadata_filter,
    is_relation_query,
    resolve_metadata_filter_decision,
)
from .prompts import build_rag_system_prompt, build_rag_user_prompt
from .qdrant_retrieval import (
    PayloadSchemaInspection,
    QdrantRetriever,
    RetrievedChunk,
    assert_required_payload_fields,
    build_article_filter,
    build_law_filter,
    build_view_filter,
    introspect_payload_schema,
    merge_filters,
    merge_retrieved,
    retrieve_multi_queries_hybrid,
    retrieve_multi_queries,
)
from .query_rewriting import QueryRewritePayload, rewrite_query
from .reranking import RerankRow, rerank_candidates
from .reporting import (
    provenance_rows,
    retrieval_preview,
    summarize_answer,
    summarize_context,
    summarize_filters,
)
from .schemas import RagAnswer


class RagState(TypedDict, total=False):
    question: str
    normalized_query: str
    rewritten_queries: list[str]
    metadata_filter_decision: dict[str, Any]
    metadata_query_filter_decision: dict[str, Any]
    relation_query: bool
    metadata_hard_law_filter_applied: bool
    metadata_hard_article_filter_applied: bool
    query_filter: Any
    filters_used: dict[str, Any]
    retrieval_batches: list[dict[str, Any]]
    retrieval_mode: str
    dense_retrieved_count: int
    sparse_retrieved_count: int
    fusion_overlap_count: int
    retrieved: list[RetrievedChunk]
    graph_expansion: dict[str, Any]
    reranked: list[dict[str, Any]]
    context: str
    context_summary: dict[str, Any]
    answer: dict[str, Any]
    provenance: list[dict[str, Any]]
    pipeline_errors: list[dict[str, Any]]
    trace: list[dict[str, Any]]


@dataclass
class RuntimeResources:
    config: RagRuntimeConfig
    dataset_validation: DatasetValidationResult
    index_contract: IndexContract
    payload_inspection: PayloadSchemaInspection
    client: QdrantClient
    retriever: QdrantRetriever
    graph_adapter: LegalGraphAdapter
    law_catalog: LawCatalog
    collection_vector_size: int
    dense_vector_name: str | None
    sparse_enabled: bool
    sparse_vector_name: str | None
    sparse_artifacts_path: str | None
    query_vector_size: int
    created_client: bool

    def close(self) -> None:
        if not self.created_client:
            return
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()


def _validate_dataset_or_raise(result: DatasetValidationResult) -> None:
    if result.errors:
        details = "\n".join([f"- {msg}" for msg in result.errors])
        raise RuntimeError(f"Dataset validation failed:\n{details}")


def _infer_query_vector_size(embedder: SupportsEmbedding) -> int:
    probe = embedder.embed_texts(["dimension probe text"])
    if not probe or not isinstance(probe[0], list):
        raise RuntimeError("Embedding probe returned invalid vector payload")
    size = len(probe[0])
    if size <= 0:
        raise RuntimeError("Embedding probe returned an empty vector")
    return size


def _append_trace(state: RagState, event: dict[str, Any]) -> list[dict[str, Any]]:
    trace = list(state.get("trace") or [])
    trace.append(event)
    return trace


def _append_pipeline_error(
    state: RagState,
    *,
    stage: str,
    error: str,
) -> list[dict[str, Any]]:
    errors = list(state.get("pipeline_errors") or [])
    errors.append({"stage": stage, "error": error})
    return errors


def _build_source_tags_map(state: RagState) -> dict[str, set[str]]:
    tags: dict[str, set[str]] = {}
    for batch in list(state.get("retrieval_batches") or []):
        if not isinstance(batch, dict):
            continue
        name = str(batch.get("name") or "retrieval").strip() or "retrieval"
        chunk_ids = batch.get("retrieved_chunk_ids") or []
        for chunk_id in chunk_ids:
            cid = str(chunk_id or "").strip()
            if not cid:
                continue
            tags.setdefault(cid, set()).add(name)
    return tags


def _build_provenance(
    retrieved: list[RetrievedChunk],
    citations: list[str],
    *,
    source_tags_by_chunk: dict[str, set[str]] | None = None,
    rerank_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cited_set = set(citations)
    source_tags_by_chunk = source_tags_by_chunk or {}
    rerank_rows = rerank_rows or {}

    rows: list[dict[str, Any]] = []
    for doc in retrieved:
        rerank_row = rerank_rows.get(doc.chunk_id) or {}
        rows.append(
            {
                "chunk_id": doc.chunk_id,
                "law_id": doc.law_id,
                "article_id": doc.article_id,
                "source_chunk_ids": list(doc.source_chunk_ids),
                "source_passage_ids": list(doc.source_passage_ids),
                "score": float(doc.score),
                "retrieval_source": sorted(source_tags_by_chunk.get(doc.chunk_id, set())),
                "rerank_score": rerank_row.get("final_score"),
                "cited": doc.chunk_id in cited_set,
            }
        )
    return rows


def _safe_parse_rag_answer(raw_text: str) -> RagAnswer:
    """Parse model output into RagAnswer without breaking pipeline execution."""
    try:
        return parse_structured_output(raw_text, RagAnswer)
    except ValidationError:
        return RagAnswer(answer="", citations=[], needs_more_context=True)


def _parse_metadata_decision(payload: dict[str, Any] | None) -> MetadataFilterDecision | None:
    if not isinstance(payload, dict):
        return None
    view_raw = str(payload.get("view") or "none").strip().lower()
    view = "none"
    if view_raw in {"none", "current", "historical"}:
        view = view_raw

    relation_types = tuple(
        [str(x).strip() for x in (payload.get("relation_types") or []) if str(x).strip()]
    )
    law_ids = tuple(
        [str(x).strip() for x in (payload.get("law_ids") or []) if str(x).strip()]
    )
    article_ids = tuple(
        [str(x).strip() for x in (payload.get("article_ids") or []) if str(x).strip()]
    )

    return MetadataFilterDecision(
        view=view,  # type: ignore[arg-type]
        law_status=(str(payload.get("law_status") or "").strip() or None),
        law_ids=law_ids,
        relation_types=relation_types,
        article_ids=article_ids,
        year_from=(int(payload["year_from"]) if payload.get("year_from") is not None else None),
        year_to=(int(payload["year_to"]) if payload.get("year_to") is not None else None),
        applied_heuristics=tuple(
            [str(x).strip() for x in (payload.get("applied_heuristics") or []) if str(x).strip()]
        ),
    )


def _is_legally_specific_query(query: str) -> bool:
    query_low = str(query or "").lower()
    specific_markers = (
        "art.",
        "articolo",
        "comma",
        "l.r.",
        "legge regionale",
        "d.lgs.",
        "law:",
        "#art:",
    )
    return any(marker in query_low for marker in specific_markers)


def _build_context_provenance_map(state: RagState) -> dict[str, dict[str, Any]]:
    source_tags = _build_source_tags_map(state)
    rerank_rows = {
        str(row.get("chunk_id") or ""): row
        for row in list(state.get("reranked") or [])
        if isinstance(row, dict)
    }

    out: dict[str, dict[str, Any]] = {}
    for chunk_id, tags in source_tags.items():
        out[chunk_id] = {
            "retrieval_source": ",".join(sorted(tags)),
        }
    for chunk_id, row in rerank_rows.items():
        cur = out.setdefault(chunk_id, {})
        cur["rerank_score"] = row.get("final_score")
    return out


def prepare_runtime(
    config: RagRuntimeConfig,
    *,
    client: QdrantClient | None = None,
    embedder: SupportsEmbedding | None = None,
) -> RuntimeResources:
    config = config.with_overrides()

    dataset_validation = validate_dataset(config.resolved_dataset_dir, strict=True)
    _validate_dataset_or_raise(dataset_validation)

    contract = resolve_index_contract(config)
    if (
        config.enforce_index_contract_coverage
        and contract.eval_reference_coverage is not None
        and contract.eval_reference_coverage < float(config.index_contract_min_eval_coverage)
    ):
        raise RuntimeError(
            "Index contract coverage check failed for runtime. "
            f"eval_reference_coverage={contract.eval_reference_coverage:.3f} "
            f"< min_required={config.index_contract_min_eval_coverage:.3f}. "
            "Select a run_id/collection aligned with evaluation references or rebuild indexing."
        )

    created_client = client is None
    if client is None:
        qdrant_url = str(config.qdrant_url or "").strip()
        if config.qdrant_prefer_remote and qdrant_url:
            client = QdrantClient(
                url=qdrant_url,
                api_key=(str(config.qdrant_api_key).strip() or None),
            )
        else:
            client = QdrantClient(path=str(contract.qdrant_path))

    assert client is not None
    if not client.collection_exists(collection_name=contract.collection_name):
        raise RuntimeError(
            f"Collection {contract.collection_name!r} does not exist in "
            f"{contract.qdrant_path}. Run notebook 04 indexing first or set explicit contract."
        )

    if embedder is None:
        embedder = build_embedder(config.to_indexing_config())

    vector_caps = collection_vector_capabilities(client, contract.collection_name)
    sparse_vector_name: str | None = None
    if vector_caps.sparse_enabled:
        if (
            contract.sparse_vector_name
            and contract.sparse_vector_name in set(vector_caps.sparse_vector_names)
        ):
            sparse_vector_name = contract.sparse_vector_name
        elif "bm25" in set(vector_caps.sparse_vector_names):
            sparse_vector_name = "bm25"
        elif vector_caps.sparse_vector_names:
            sparse_vector_name = vector_caps.sparse_vector_names[0]

    sparse_artifact_path = (
        contract.sparse_artifacts_path
        if (contract.sparse_artifacts_path is not None and sparse_vector_name is not None)
        else None
    )

    retriever = QdrantRetriever.from_sparse_artifact(
        client=client,
        collection_name=contract.collection_name,
        embedder=embedder,
        field_map=config.payload_fields,
        dense_vector_name=vector_caps.dense_vector_name,
        sparse_vector_name=sparse_vector_name,
        sparse_artifact_path=sparse_artifact_path,
    )

    collection_vector_size = int(get_vector_size(client, contract.collection_name))
    query_vector_size = _infer_query_vector_size(embedder)
    if query_vector_size != collection_vector_size:
        raise RuntimeError(
            "Embedding dimension mismatch between query embedder and Qdrant collection: "
            f"query_vector_size={query_vector_size}, collection_vector_size={collection_vector_size}, "
            f"collection={contract.collection_name!r}, run_id={contract.run_id!r}. "
            "Set RagRuntimeConfig.indexing_run_id/collection_name to the correct run, or rebuild index "
            "with the same embedding model used at query time."
        )

    inspection = introspect_payload_schema(
        client,
        collection_name=contract.collection_name,
        field_map=config.payload_fields,
        sample_size=config.payload_introspection_sample_size,
    )
    assert_required_payload_fields(inspection)

    graph_adapter = LegalGraphAdapter(config.resolved_dataset_dir)
    law_catalog = build_law_catalog(config.resolved_dataset_dir)

    return RuntimeResources(
        config=config,
        dataset_validation=dataset_validation,
        index_contract=contract,
        payload_inspection=inspection,
        client=client,
        retriever=retriever,
        graph_adapter=graph_adapter,
        law_catalog=law_catalog,
        collection_vector_size=collection_vector_size,
        dense_vector_name=vector_caps.dense_vector_name,
        sparse_enabled=bool(sparse_vector_name),
        sparse_vector_name=sparse_vector_name,
        sparse_artifacts_path=(str(sparse_artifact_path) if sparse_artifact_path else None),
        query_vector_size=query_vector_size,
        created_client=created_client,
    )


def _build_nodes(
    config: RagRuntimeConfig,
    resources: RuntimeResources,
    *,
    llm: SupportsInvoke | SupportsRunSync | None = None,
) -> dict[str, Any]:
    rag_system_prompt = build_rag_system_prompt(language=config.query_language)

    llm_model: SupportsInvoke | None = None
    answer_agent: SupportsRunSync | None = None
    rewrite_agent: SupportsRunSync | None = None

    if llm is not None:
        if hasattr(llm, "run_sync"):
            answer_agent = llm  # type: ignore[assignment]
            rewrite_agent = llm  # type: ignore[assignment]
        else:
            llm_model = llm  # type: ignore[assignment]
    else:
        try:
            answer_agent = build_pydantic_ai_agent(
                config,
                result_type=RagAnswer,
                system_prompt=rag_system_prompt,
            )
        except Exception:
            llm_model = build_chat_model(config)

        if config.advanced.rewrite.enabled and config.advanced.rewrite.use_llm:
            try:
                rewrite_agent = build_pydantic_ai_agent(
                    config,
                    result_type=QueryRewritePayload,
                    system_prompt=(
                        "Sei un assistente di query rewriting per retrieval legale. "
                        "Restituisci solo JSON strutturato."
                    ),
                )
            except Exception:
                rewrite_agent = None

    def normalize_query(state: RagState) -> RagState:
        question = str(state.get("question") or "").strip()
        if not question:
            raise ValueError("Question is empty")
        normalized = " ".join(question.split())
        return {
            "normalized_query": normalized,
            "trace": _append_trace(
                state,
                {
                    "node": "normalize_query",
                    "question_length": len(question),
                },
            ),
        }

    def retrieve_top_k(state: RagState) -> RagState:
        query = str(state.get("normalized_query") or "").strip()
        view_name = (config.view_filter or "none").strip().lower()
        query_filter = build_view_filter(config.payload_fields, view_name)
        retrieved = resources.retriever.query(
            query,
            top_k=config.top_k,
            query_filter=query_filter,
            score_threshold=config.retrieval_score_threshold,
            threshold_direction=config.score_threshold_direction,
        )
        return {
            "query_filter": query_filter,
            "filters_used": {
                "view": view_name,
                "metadata_mode": "naive",
            },
            "retrieved": retrieved,
            "retrieval_batches": [
                {
                    "name": "primary",
                    "query": query,
                    "top_k": int(config.top_k),
                    "retrieved_chunk_ids": [d.chunk_id for d in retrieved],
                    "retrieved_count": len(retrieved),
                }
            ],
            "trace": _append_trace(
                state,
                {
                    "node": "retrieve_top_k",
                    "top_k": config.top_k,
                    "retrieved_count": len(retrieved),
                    "view_filter": view_name,
                },
            ),
        }

    def rewrite_or_decompose_query(state: RagState) -> RagState:
        query = str(state.get("normalized_query") or "").strip()
        if not query:
            return {
                "rewritten_queries": [],
                "trace": _append_trace(state, {"node": "rewrite_or_decompose_query", "count": 0}),
            }

        rewrite_out = rewrite_query(
            query,
            config=config.advanced.rewrite,
            llm_model=llm_model,
            rewrite_agent=rewrite_agent,
        )
        rewritten_queries = rewrite_out.all_queries(
            max_subqueries=config.advanced.rewrite.max_subqueries
        )

        updates: RagState = {
            "rewritten_queries": rewritten_queries,
            "trace": _append_trace(
                state,
                {
                    "node": "rewrite_or_decompose_query",
                    "count": len(rewritten_queries),
                    "used_llm": rewrite_out.used_llm,
                    "fallback_used": rewrite_out.fallback_used,
                },
            ),
        }
        if rewrite_out.error:
            updates["pipeline_errors"] = _append_pipeline_error(
                state,
                stage="rewrite_or_decompose_query",
                error=rewrite_out.error,
            )
        return updates

    def build_metadata_filter_node(state: RagState) -> RagState:
        query = str(state.get("normalized_query") or "")
        resolved_refs = resources.law_catalog.resolve(query) if query else None
        decision = resolve_metadata_filter_decision(
            query,
            config=config.advanced.metadata_filtering,
            default_view=config.view_filter,
            resolved_references=resolved_refs,
        )
        relation_query = is_relation_query(query, decision=decision)
        query_filter_decision = decision
        if (
            relation_query
            and config.advanced.metadata_filtering.relax_law_article_filters_on_relation_queries
        ):
            query_filter_decision = MetadataFilterDecision(
                view=decision.view,
                law_status=decision.law_status,
                law_ids=tuple(),
                relation_types=decision.relation_types,
                article_ids=tuple(),
                year_from=decision.year_from,
                year_to=decision.year_to,
                applied_heuristics=decision.applied_heuristics,
            )
        query_filter = build_metadata_filter(config.payload_fields, query_filter_decision)

        return {
            "metadata_filter_decision": decision.to_dict(),
            "metadata_query_filter_decision": query_filter_decision.to_dict(),
            "relation_query": relation_query,
            "metadata_hard_law_filter_applied": bool(query_filter_decision.law_ids),
            "metadata_hard_article_filter_applied": bool(query_filter_decision.article_ids),
            "query_filter": query_filter,
            "filters_used": {
                "view": query_filter_decision.view,
                "law_status": query_filter_decision.law_status,
                "law_ids": list(query_filter_decision.law_ids),
                "relation_types": list(query_filter_decision.relation_types),
                "year_from": query_filter_decision.year_from,
                "year_to": query_filter_decision.year_to,
                "metadata_mode": config.advanced.metadata_filtering.mode,
                "relation_query": relation_query,
                "metadata_seed_law_ids": list(decision.law_ids),
                "metadata_seed_article_ids": list(decision.article_ids),
                "metadata_hard_law_filter_applied": bool(query_filter_decision.law_ids),
                "metadata_hard_article_filter_applied": bool(query_filter_decision.article_ids),
            },
            "trace": _append_trace(
                state,
                {
                    "node": "build_metadata_filter",
                    "mode": config.advanced.metadata_filtering.mode,
                    "relation_query": relation_query,
                    "heuristics": list(decision.applied_heuristics),
                    "hard_law_filter_applied": bool(query_filter_decision.law_ids),
                    "hard_article_filter_applied": bool(query_filter_decision.article_ids),
                    "resolved_law_refs": len((resolved_refs.law_ids if resolved_refs else tuple())),
                    "resolved_article_refs": len(
                        (resolved_refs.article_ids if resolved_refs else tuple())
                    ),
                },
            ),
        }

    def retrieve_multi(state: RagState) -> RagState:
        query = str(state.get("normalized_query") or "")
        rewritten_queries = list(state.get("rewritten_queries") or [])
        all_queries = rewritten_queries if rewritten_queries else [query]

        retrieval_mode = "dense_only"
        dense_retrieved_count = 0
        sparse_retrieved_count = 0
        fusion_overlap_count = 0
        hybrid_stats: list[dict[str, Any]] = []

        if config.advanced.hybrid.enabled:
            retrieved, batches, hybrid_stats = retrieve_multi_queries_hybrid(
                resources.retriever,
                queries=all_queries,
                top_k_primary=config.advanced.multi_retrieval.top_k_primary,
                top_k_secondary=config.advanced.multi_retrieval.top_k_secondary,
                query_filter=state.get("query_filter"),
                score_threshold=config.retrieval_score_threshold,
                threshold_direction=config.score_threshold_direction,
                dedupe_by_chunk_id=config.advanced.multi_retrieval.dedupe_by_chunk_id,
                hybrid_config=config.advanced.hybrid,
            )
            if hybrid_stats:
                dense_retrieved_count = int(
                    sum(int(x.get("dense_retrieved_count") or 0) for x in hybrid_stats)
                )
                sparse_retrieved_count = int(
                    sum(int(x.get("sparse_retrieved_count") or 0) for x in hybrid_stats)
                )
                fusion_overlap_count = int(
                    sum(int(x.get("overlap_count") or 0) for x in hybrid_stats)
                )
                modes = {str(x.get("retrieval_mode") or "") for x in hybrid_stats}
                if "hybrid" in modes:
                    retrieval_mode = "hybrid"
                elif "fallback_dense" in modes:
                    retrieval_mode = "fallback_dense"
        else:
            retrieved, batches = retrieve_multi_queries(
                resources.retriever,
                queries=all_queries,
                top_k_primary=config.advanced.multi_retrieval.top_k_primary,
                top_k_secondary=config.advanced.multi_retrieval.top_k_secondary,
                query_filter=state.get("query_filter"),
                score_threshold=config.retrieval_score_threshold,
                threshold_direction=config.score_threshold_direction,
                dedupe_by_chunk_id=config.advanced.multi_retrieval.dedupe_by_chunk_id,
            )
        if len(retrieved) > config.advanced.max_candidates:
            retrieved = retrieved[: config.advanced.max_candidates]

        return {
            "retrieved": retrieved,
            "retrieval_batches": [x.to_dict() for x in batches],
            "retrieval_mode": retrieval_mode,
            "dense_retrieved_count": dense_retrieved_count,
            "sparse_retrieved_count": sparse_retrieved_count,
            "fusion_overlap_count": fusion_overlap_count,
            "trace": _append_trace(
                state,
                {
                    "node": "retrieve_multi",
                    "queries": len(all_queries),
                    "retrieved_count": len(retrieved),
                    "retrieval_mode": retrieval_mode,
                    "dense_retrieved_count": dense_retrieved_count,
                    "sparse_retrieved_count": sparse_retrieved_count,
                    "fusion_overlap_count": fusion_overlap_count,
                    "hybrid_stats": hybrid_stats,
                },
            ),
        }

    def graph_expand(state: RagState) -> RagState:
        base_docs = list(state.get("retrieved") or [])
        if not config.advanced.graph_expansion.enabled or not base_docs:
            return {
                "graph_expansion": {
                    "enabled": bool(config.advanced.graph_expansion.enabled),
                    "graph_retrieved_count": 0,
                },
                "trace": _append_trace(
                    state,
                    {
                        "node": "graph_expand",
                        "enabled": bool(config.advanced.graph_expansion.enabled),
                        "graph_retrieved_count": 0,
                    },
                ),
            }

        normalized_query = str(state.get("normalized_query") or "")
        metadata_decision = _parse_metadata_decision(
            state.get("metadata_filter_decision")
            if isinstance(state.get("metadata_filter_decision"), dict)
            else None
        )
        relation_query = bool(state.get("relation_query"))
        if not relation_query:
            relation_query = is_relation_query(normalized_query, decision=metadata_decision)
        query_is_specific = _is_legally_specific_query(normalized_query)
        seed_law_ids = {d.law_id for d in base_docs if d.law_id}
        graph_cfg = config.advanced.graph_expansion
        dynamic_related_cap = int(graph_cfg.max_related_laws)
        dynamic_graph_top_k = int(graph_cfg.graph_retrieval_top_k)
        expansion_enabled = True
        reason = "default"
        specific_query_mode_applied = "full"
        if (
            relation_query
            and graph_cfg.force_on_relation_queries
        ):
            expansion_enabled = True
            reason = "forced_relation_query"
            specific_query_mode_applied = "full"
        elif query_is_specific and len(seed_law_ids) <= 1:
            specific_query_mode = str(graph_cfg.specific_query_mode or "minimal").strip().lower()
            specific_query_mode_applied = specific_query_mode
            if specific_query_mode == "disable":
                expansion_enabled = False
                reason = "gated_specific_query"
            elif specific_query_mode == "minimal":
                dynamic_related_cap = min(
                    dynamic_related_cap,
                    int(graph_cfg.specific_query_max_related_laws),
                )
                dynamic_graph_top_k = min(
                    dynamic_graph_top_k,
                    int(graph_cfg.specific_query_graph_retrieval_top_k),
                )
                reason = "gated_specific_query"
            else:
                reason = "gated_specific_query"

        if not expansion_enabled:
            return {
                "graph_expansion": {
                    "enabled": False,
                    "reason": reason,
                    "specific_query_mode_applied": specific_query_mode_applied,
                    "graph_retrieved_count": 0,
                },
                "trace": _append_trace(
                    state,
                    {
                        "node": "graph_expand",
                        "enabled": False,
                        "reason": reason,
                        "specific_query_mode_applied": specific_query_mode_applied,
                        "graph_retrieved_count": 0,
                        "relation_query": relation_query,
                    },
                ),
            }

        expansion = resources.graph_adapter.expand_from_retrieved(
            base_docs,
            max_related_laws=dynamic_related_cap,
        )

        graph_docs: list[RetrievedChunk] = []
        base_filter = state.get("query_filter")
        if metadata_decision is not None:
            graph_base_decision = MetadataFilterDecision(
                view=metadata_decision.view,
                law_status=metadata_decision.law_status,
                law_ids=tuple(),
                relation_types=metadata_decision.relation_types,
                article_ids=tuple(),
                year_from=metadata_decision.year_from,
                year_to=metadata_decision.year_to,
                applied_heuristics=metadata_decision.applied_heuristics,
            )
            base_filter = build_metadata_filter(config.payload_fields, graph_base_decision)

        if expansion.related_law_ids:
            law_filter = build_law_filter(config.payload_fields, expansion.related_law_ids)
            if config.advanced.hybrid.enabled:
                law_hybrid = resources.retriever.query_hybrid(
                    normalized_query,
                    top_k=dynamic_graph_top_k,
                    query_filter=merge_filters(base_filter, law_filter),
                    score_threshold=config.retrieval_score_threshold,
                    threshold_direction=config.score_threshold_direction,
                    hybrid_config=config.advanced.hybrid,
                )
                graph_docs = merge_retrieved(graph_docs, list(law_hybrid.retrieved))
            else:
                graph_docs = merge_retrieved(
                    graph_docs,
                    resources.retriever.query(
                        normalized_query,
                        top_k=dynamic_graph_top_k,
                        query_filter=merge_filters(base_filter, law_filter),
                        score_threshold=config.retrieval_score_threshold,
                        threshold_direction=config.score_threshold_direction,
                    ),
                )

        if (
            config.advanced.graph_expansion.include_related_articles
            and expansion.related_article_ids
        ):
            article_filter = build_article_filter(
                config.payload_fields,
                expansion.related_article_ids,
            )
            if config.advanced.hybrid.enabled:
                article_hybrid = resources.retriever.query_hybrid(
                    normalized_query,
                    top_k=dynamic_graph_top_k,
                    query_filter=merge_filters(base_filter, article_filter),
                    score_threshold=config.retrieval_score_threshold,
                    threshold_direction=config.score_threshold_direction,
                    hybrid_config=config.advanced.hybrid,
                )
                graph_docs = merge_retrieved(graph_docs, list(article_hybrid.retrieved))
            else:
                graph_docs = merge_retrieved(
                    graph_docs,
                    resources.retriever.query(
                        normalized_query,
                        top_k=dynamic_graph_top_k,
                        query_filter=merge_filters(base_filter, article_filter),
                        score_threshold=config.retrieval_score_threshold,
                        threshold_direction=config.score_threshold_direction,
                    ),
                )

        merged_docs = merge_retrieved(base_docs, graph_docs)
        if len(merged_docs) > config.advanced.max_candidates:
            merged_docs = merged_docs[: config.advanced.max_candidates]

        retrieval_batches = list(state.get("retrieval_batches") or [])
        retrieval_batches.append(
            {
                "name": "graph",
                "query": normalized_query,
                "top_k": dynamic_graph_top_k,
                "retrieved_chunk_ids": [d.chunk_id for d in graph_docs],
                "retrieved_count": len(graph_docs),
            }
        )

        return {
            "retrieved": merged_docs,
            "retrieval_batches": retrieval_batches,
            "graph_expansion": {
                **expansion.to_dict(),
                "enabled": True,
                "reason": reason,
                "specific_query_mode_applied": specific_query_mode_applied,
                "relation_query": relation_query,
                "graph_retrieved_count": len(graph_docs),
            },
            "trace": _append_trace(
                state,
                {
                    "node": "graph_expand",
                    "seed_count": len(base_docs),
                    "graph_retrieved_count": len(graph_docs),
                    "total_after_merge": len(merged_docs),
                    "reason": reason,
                    "specific_query_mode_applied": specific_query_mode_applied,
                    "relation_query": relation_query,
                },
            ),
        }

    def rerank_candidates_node(state: RagState) -> RagState:
        docs = list(state.get("retrieved") or [])
        if not docs:
            return {
                "reranked": [],
                "trace": _append_trace(
                    state,
                    {"node": "rerank_candidates", "candidates": 0, "enabled": False},
                ),
            }

        if not config.advanced.rerank.enabled:
            rows = [
                {
                    "chunk_id": d.chunk_id,
                    "retrieval_score": float(d.score),
                    "lexical_overlap": 0.0,
                    "sparse_score": 0.0,
                    "graph_bonus": 0.0,
                    "metadata_bonus": 0.0,
                    "final_score": float(d.score),
                    "source_tags": sorted(_build_source_tags_map(state).get(d.chunk_id, set())),
                }
                for d in docs
            ]
            return {
                "reranked": rows,
                "trace": _append_trace(
                    state,
                    {
                        "node": "rerank_candidates",
                        "candidates": len(docs),
                        "enabled": False,
                    },
                ),
            }

        metadata_decision = _parse_metadata_decision(
            state.get("metadata_filter_decision") if isinstance(state.get("metadata_filter_decision"), dict) else None
        )
        relation_query = bool(state.get("relation_query"))
        normalized_query = str(state.get("normalized_query") or "")
        if not relation_query:
            relation_query = is_relation_query(normalized_query, decision=metadata_decision)
        source_map = _build_source_tags_map(state)
        rerank_cfg = config.advanced.rerank
        if relation_query:
            rerank_cfg = AdvancedRerankConfig(
                enabled=rerank_cfg.enabled,
                weight_retrieval_score=1.0,
                weight_graph_bonus=0.35,
                weight_metadata_bonus=0.10,
                weight_lexical_overlap=0.10,
                weight_sparse_score=0.25,
                tie_breaker=rerank_cfg.tie_breaker,
            )

        rerank_out = rerank_candidates(
            normalized_query,
            docs,
            config=rerank_cfg,
            source_tags_by_chunk=source_map,
            metadata_decision=metadata_decision,
            relation_query=relation_query,
        )

        return {
            "retrieved": list(rerank_out.ordered_chunks),
            "reranked": [row.to_dict() for row in rerank_out.rows],
            "trace": _append_trace(
                state,
                {
                    "node": "rerank_candidates",
                    "candidates": len(docs),
                    "enabled": True,
                    "relation_query": relation_query,
                    "weights": {
                        "weight_retrieval_score": rerank_cfg.weight_retrieval_score,
                        "weight_graph_bonus": rerank_cfg.weight_graph_bonus,
                        "weight_metadata_bonus": rerank_cfg.weight_metadata_bonus,
                        "weight_lexical_overlap": rerank_cfg.weight_lexical_overlap,
                        "weight_sparse_score": rerank_cfg.weight_sparse_score,
                    },
                },
            ),
        }

    def build_context_node(state: RagState) -> RagState:
        docs = list(state.get("retrieved") or [])
        context_result: ContextBuildResult = build_context(
            docs,
            max_chunks=config.max_context_chunks,
            max_chars=config.max_context_chars,
            per_chunk_max_chars=config.per_chunk_max_chars,
            provenance_map=_build_context_provenance_map(state),
        )
        return {
            "context": context_result.context,
            "context_summary": context_result.to_dict(),
            "trace": _append_trace(
                state,
                {
                    "node": "build_context",
                    "included": len(context_result.included_chunk_ids),
                },
            ),
        }

    def generate_answer_structured(state: RagState) -> RagState:
        question = str(state.get("question") or "")
        context = str(state.get("context") or "")
        base_user_prompt = build_rag_user_prompt(question, context)
        guard_cfg = config.advanced.answer_guard
        retrieved = list(state.get("retrieved") or [])
        top_score = float(retrieved[0].score) if retrieved else 0.0
        retrieval_mode = str(state.get("retrieval_mode") or "dense_only")
        assertive_mode = bool(
            config.pipeline_mode == "advanced"
            and top_score >= 0.01
            and retrieval_mode in {"hybrid", "dense_only", "fallback_dense"}
        )
        if assertive_mode:
            base_user_prompt = (
                f"{base_user_prompt}\n\n"
                "Istruzione aggiuntiva: se nel contesto c'e' evidenza esplicita, rispondi in modo "
                "puntuale e sintetico; usa needs_more_context=true solo in assenza di evidenze chiave."
            )

        answer_model: RagAnswer = RagAnswer(answer="", citations=[], needs_more_context=True)
        raw_text = ""
        pipeline_errors = list(state.get("pipeline_errors") or [])
        answer_source = "model"
        was_empty_before_guard = False

        def _record_error(stage: str, error: str) -> None:
            pipeline_errors.append({"stage": stage, "error": error})

        def _invoke_answer(user_prompt: str, *, stage: str) -> tuple[RagAnswer, str]:
            if answer_agent is not None:
                model = run_structured_with_agent(user_prompt, answer_agent, RagAnswer)
                return model, model.model_dump_json()
            if llm_model is not None:
                full_prompt = (
                    f"{rag_system_prompt}\n\n"
                    "Rispondi in JSON valido con campi: answer, citations, needs_more_context.\n\n"
                    f"{user_prompt}"
                )
                raw = invoke_model(llm_model, full_prompt)
                return _safe_parse_rag_answer(raw), raw
            raise RuntimeError(f"No LLM backend available for answer generation ({stage})")

        try:
            answer_model, raw_text = _invoke_answer(base_user_prompt, stage="first_pass")
        except Exception as exc:
            _record_error("generate_answer_structured", f"{type(exc).__name__}: {exc}")
            answer_model = RagAnswer(answer="", citations=[], needs_more_context=True)
            raw_text = ""

        context_summary_payload = (
            state.get("context_summary") if isinstance(state.get("context_summary"), dict) else {}
        )
        context_ids = set((context_summary_payload or {}).get("included_chunk_ids") or [])
        if not context_ids:
            context_ids = {doc.chunk_id for doc in list(state.get("retrieved") or [])}
        context_included_count = (
            int(context_summary_payload.get("included_count"))
            if context_summary_payload.get("included_count") is not None
            else len(context_ids)
        )

        if is_empty_structured_answer(answer_model):
            was_empty_before_guard = True
            if guard_cfg.mark_empty_as_pipeline_error:
                _record_error("generate_answer_structured", "empty_answer_detected:first_pass")

            retry_count = guard_cfg.max_empty_retries if guard_cfg.retry_on_empty_answer else 0
            for attempt in range(1, retry_count + 1):
                retry_prompt = (
                    f"{base_user_prompt}\n\n"
                    "Vincolo obbligatorio di output: il campo `answer` non puo' essere vuoto. "
                    "Anche se `needs_more_context=true`, inserisci una risposta sintetica e non vuota."
                )
                try:
                    retry_model, retry_raw = _invoke_answer(
                        retry_prompt,
                        stage=f"retry_{attempt}",
                    )
                    answer_model = retry_model
                    raw_text = retry_raw
                except Exception as exc:
                    _record_error(
                        "generate_answer_structured_retry",
                        f"retry_{attempt}:{type(exc).__name__}: {exc}",
                    )
                    continue

                if not is_empty_structured_answer(answer_model):
                    answer_source = "retry"
                    break
                if guard_cfg.mark_empty_as_pipeline_error:
                    _record_error(
                        "generate_answer_structured_retry",
                        f"empty_answer_detected:retry_{attempt}",
                    )

            if is_empty_structured_answer(answer_model):
                answer_source = "fallback"
                fallback_message = (
                    guard_cfg.fallback_message_en
                    if str(config.query_language or "").strip().lower().startswith("en")
                    else guard_cfg.fallback_message_it
                )
                answer_model = RagAnswer(
                    answer=fallback_message,
                    citations=[],
                    needs_more_context=True,
                )

        strong_retrieval = (
            retrieval_mode == "hybrid"
            and len(retrieved) >= 12
            and int(context_included_count) >= 8
        )
        if (
            not is_empty_structured_answer(answer_model)
            and answer_model.needs_more_context
            and strong_retrieval
            and guard_cfg.retry_on_needs_more_context
        ):
            retry_count = int(guard_cfg.max_needs_more_retries)
            sample_ids = sorted(context_ids)[:5]
            sample_hint = ", ".join(sample_ids)
            for attempt in range(1, retry_count + 1):
                retry_prompt = (
                    f"{base_user_prompt}\n\n"
                    "Tentativo best-effort obbligatorio: con il contesto disponibile devi fornire comunque "
                    "una risposta utile e concreta.\n"
                    "Vincoli: `answer` non vuoto; in `citations` includi almeno un `chunk_id` presente nel "
                    "contesto; usa `needs_more_context=true` solo se restano lacune sostanziali.\n"
                    f"Esempi di chunk_id validi: {sample_hint}"
                )
                try:
                    retry_model, retry_raw = _invoke_answer(
                        retry_prompt,
                        stage=f"retry_needs_more_context_{attempt}",
                    )
                except Exception as exc:
                    _record_error(
                        "generate_answer_structured_retry_needs_more_context",
                        f"retry_{attempt}:{type(exc).__name__}: {exc}",
                    )
                    continue
                if is_empty_structured_answer(retry_model):
                    if guard_cfg.mark_empty_as_pipeline_error:
                        _record_error(
                            "generate_answer_structured_retry_needs_more_context",
                            f"empty_answer_detected:retry_{attempt}",
                        )
                    continue
                retry_citations = [cid for cid in retry_model.citations if cid in context_ids]
                if context_ids and not retry_citations:
                    _record_error(
                        "generate_answer_structured_retry_needs_more_context",
                        f"missing_valid_citation:retry_{attempt}",
                    )
                    continue
                answer_model = retry_model
                raw_text = retry_raw
                answer_source = "retry_needs_more_context"
                break

        citations = [cid for cid in answer_model.citations if cid in context_ids]

        answer_payload = RagAnswer(
            answer=answer_model.answer,
            citations=citations,
            needs_more_context=answer_model.needs_more_context,
        ).model_dump()
        answer_payload["answer_source"] = answer_source
        answer_payload["was_empty_before_guard"] = was_empty_before_guard
        answer_payload["raw"] = raw_text

        source_tags_by_chunk = _build_source_tags_map(state)
        rerank_rows = {
            str(row.get("chunk_id") or ""): row
            for row in list(state.get("reranked") or [])
            if isinstance(row, dict)
        }
        provenance = _build_provenance(
            retrieved,
            citations,
            source_tags_by_chunk=source_tags_by_chunk,
            rerank_rows=rerank_rows,
        )
        return {
            "answer": answer_payload,
            "provenance": provenance,
            "pipeline_errors": pipeline_errors,
            "trace": _append_trace(
                state,
                {
                    "node": "generate_answer_structured",
                    "citations_count": len(citations),
                    "answer_source": answer_source,
                    "was_empty_before_guard": was_empty_before_guard,
                },
            ),
        }

    return {
        "normalize_query": normalize_query,
        "retrieve_top_k": retrieve_top_k,
        "rewrite_or_decompose_query": rewrite_or_decompose_query,
        "build_metadata_filter": build_metadata_filter_node,
        "retrieve_multi": retrieve_multi,
        "graph_expand": graph_expand,
        "rerank_candidates": rerank_candidates_node,
        "build_context": build_context_node,
        "generate_answer_structured": generate_answer_structured,
    }


def build_rag_graph(
    config: RagRuntimeConfig,
    resources: RuntimeResources,
    *,
    llm: SupportsInvoke | SupportsRunSync | None = None,
) -> Any:
    from langgraph.graph import END, START, StateGraph

    nodes = _build_nodes(config, resources, llm=llm)

    graph = StateGraph(RagState)

    if config.pipeline_mode == "advanced":
        graph.add_node("normalize_query", nodes["normalize_query"])
        graph.add_node("rewrite_or_decompose_query", nodes["rewrite_or_decompose_query"])
        graph.add_node("build_metadata_filter", nodes["build_metadata_filter"])
        graph.add_node("retrieve_multi", nodes["retrieve_multi"])
        graph.add_node("graph_expand", nodes["graph_expand"])
        graph.add_node("rerank_candidates", nodes["rerank_candidates"])
        graph.add_node("build_context", nodes["build_context"])
        graph.add_node("generate_answer_structured", nodes["generate_answer_structured"])

        graph.add_edge(START, "normalize_query")
        graph.add_edge("normalize_query", "rewrite_or_decompose_query")
        graph.add_edge("rewrite_or_decompose_query", "build_metadata_filter")
        graph.add_edge("build_metadata_filter", "retrieve_multi")
        graph.add_edge("retrieve_multi", "graph_expand")
        graph.add_edge("graph_expand", "rerank_candidates")
        graph.add_edge("rerank_candidates", "build_context")
        graph.add_edge("build_context", "generate_answer_structured")
        graph.add_edge("generate_answer_structured", END)
    else:
        graph.add_node("normalize_query", nodes["normalize_query"])
        graph.add_node("retrieve_top_k", nodes["retrieve_top_k"])
        graph.add_node("build_context", nodes["build_context"])
        graph.add_node("generate_answer_structured", nodes["generate_answer_structured"])

        graph.add_edge(START, "normalize_query")
        graph.add_edge("normalize_query", "retrieve_top_k")
        graph.add_edge("retrieve_top_k", "build_context")
        graph.add_edge("build_context", "generate_answer_structured")
        graph.add_edge("generate_answer_structured", END)

    return graph.compile()


def _run_linear(
    config: RagRuntimeConfig,
    resources: RuntimeResources,
    *,
    question: str,
    llm: SupportsInvoke | SupportsRunSync | None = None,
) -> RagState:
    nodes = _build_nodes(config, resources, llm=llm)
    if config.pipeline_mode == "advanced":
        order = [
            "normalize_query",
            "rewrite_or_decompose_query",
            "build_metadata_filter",
            "retrieve_multi",
            "graph_expand",
            "rerank_candidates",
            "build_context",
            "generate_answer_structured",
        ]
    else:
        order = [
            "normalize_query",
            "retrieve_top_k",
            "build_context",
            "generate_answer_structured",
        ]

    state: RagState = {
        "question": question,
        "trace": [],
        "pipeline_errors": [],
    }
    for name in order:
        updates = dict(nodes[name](state))
        state.update(updates)
    return state


def _run_linear_order(
    config: RagRuntimeConfig,
    resources: RuntimeResources,
    *,
    question: str,
    order: list[str],
    llm: SupportsInvoke | SupportsRunSync | None = None,
) -> RagState:
    nodes = _build_nodes(config, resources, llm=llm)
    state: RagState = {
        "question": question,
        "trace": [],
        "pipeline_errors": [],
    }
    for name in order:
        updates = dict(nodes[name](state))
        state.update(updates)
    return state


def _build_retrieval_context_graph(
    config: RagRuntimeConfig,
    resources: RuntimeResources,
    *,
    llm: SupportsInvoke | SupportsRunSync | None = None,
    pipeline_mode: str = "naive",
) -> Any:
    from langgraph.graph import END, START, StateGraph

    nodes = _build_nodes(config, resources, llm=llm)
    graph = StateGraph(RagState)
    if pipeline_mode == "advanced":
        graph.add_node("normalize_query", nodes["normalize_query"])
        graph.add_node("rewrite_or_decompose_query", nodes["rewrite_or_decompose_query"])
        graph.add_node("build_metadata_filter", nodes["build_metadata_filter"])
        graph.add_node("retrieve_multi", nodes["retrieve_multi"])
        graph.add_node("graph_expand", nodes["graph_expand"])
        graph.add_node("rerank_candidates", nodes["rerank_candidates"])
        graph.add_node("build_context", nodes["build_context"])

        graph.add_edge(START, "normalize_query")
        graph.add_edge("normalize_query", "rewrite_or_decompose_query")
        graph.add_edge("rewrite_or_decompose_query", "build_metadata_filter")
        graph.add_edge("build_metadata_filter", "retrieve_multi")
        graph.add_edge("retrieve_multi", "graph_expand")
        graph.add_edge("graph_expand", "rerank_candidates")
        graph.add_edge("rerank_candidates", "build_context")
        graph.add_edge("build_context", END)
    else:
        graph.add_node("normalize_query", nodes["normalize_query"])
        graph.add_node("retrieve_top_k", nodes["retrieve_top_k"])
        graph.add_node("build_context", nodes["build_context"])

        graph.add_edge(START, "normalize_query")
        graph.add_edge("normalize_query", "retrieve_top_k")
        graph.add_edge("retrieve_top_k", "build_context")
        graph.add_edge("build_context", END)
    return graph.compile()


def build_rag_retrieval_context_graph(
    config: RagRuntimeConfig,
    resources: RuntimeResources,
    *,
    llm: SupportsInvoke | SupportsRunSync | None = None,
    pipeline_mode: str = "naive",
) -> Any:
    if pipeline_mode not in {"naive", "advanced"}:
        raise ValueError("pipeline_mode must be one of: naive, advanced")
    effective_cfg = resources.config.with_overrides(pipeline_mode=pipeline_mode)
    return _build_retrieval_context_graph(
        effective_cfg,
        resources,
        llm=llm,
        pipeline_mode=pipeline_mode,
    )


def run_rag_retrieval_context(
    config: RagRuntimeConfig,
    question: str,
    *,
    resources: RuntimeResources | None = None,
    llm: SupportsInvoke | SupportsRunSync | None = None,
    pipeline_mode: str = "naive",
    compiled_app: Any | None = None,
) -> dict[str, Any]:
    """Run retrieval/context-only stages (no answer generation)."""

    created_resources = resources is None
    runtime = resources or prepare_runtime(config)
    try:
        if pipeline_mode not in {"naive", "advanced"}:
            raise ValueError("pipeline_mode must be one of: naive, advanced")
        config = runtime.config.with_overrides(pipeline_mode=pipeline_mode)
        effective_llm = llm
        if effective_llm is None and pipeline_mode == "naive":
            # _build_nodes builds answer node dependencies eagerly.
            # Inject a no-op invoke model to avoid requiring runtime LLM setup
            # when we only need retrieval/context stages.
            class _NoopInvoke:
                def invoke(self, prompt: str) -> str:  # pragma: no cover - defensive
                    return "{}"

            effective_llm = _NoopInvoke()  # type: ignore[assignment]
        retrieval_order = (
            [
                "normalize_query",
                "rewrite_or_decompose_query",
                "build_metadata_filter",
                "retrieve_multi",
                "graph_expand",
                "rerank_candidates",
                "build_context",
            ]
            if pipeline_mode == "advanced"
            else ["normalize_query", "retrieve_top_k", "build_context"]
        )
        try:
            app = compiled_app or _build_retrieval_context_graph(
                config,
                runtime,
                llm=effective_llm,
                pipeline_mode=pipeline_mode,
            )
            state: RagState = app.invoke(
                {"question": question, "trace": [], "pipeline_errors": []}
            )
        except Exception as graph_exc:
            try:
                state = _run_linear_order(
                    config,
                    runtime,
                    question=question,
                    llm=effective_llm,
                    order=retrieval_order,
                )
                state["pipeline_errors"] = _append_pipeline_error(
                    state,
                    stage="langgraph_runtime",
                    error=f"{type(graph_exc).__name__}: {graph_exc}",
                )
            except Exception:
                raise graph_exc

        return {
            "question": question,
            "pipeline_mode": pipeline_mode,
            "stage": "retrieval_context_only",
            "state": state,
            "retrieved_preview": retrieval_preview(list(state.get("retrieved") or [])),
            "filters_summary": summarize_filters(state),
            "context_summary": summarize_context(state),
            "retrieval_batches": list(state.get("retrieval_batches") or []),
            "retrieval_mode": str(state.get("retrieval_mode") or "dense_only"),
            "pipeline_errors": list(state.get("pipeline_errors") or []),
            "index_contract": runtime.index_contract.to_dict(),
            "payload_inspection": runtime.payload_inspection.to_dict(),
            "vector_sizes": {
                "collection_vector_size": runtime.collection_vector_size,
                "query_vector_size": runtime.query_vector_size,
                "dense_vector_name": runtime.dense_vector_name,
                "sparse_enabled": runtime.sparse_enabled,
                "sparse_vector_name": runtime.sparse_vector_name,
                "sparse_artifacts_path": runtime.sparse_artifacts_path,
            },
            "dataset_validation": {
                "is_valid": runtime.dataset_validation.is_valid,
                "counts": dict(runtime.dataset_validation.counts),
                "errors": list(runtime.dataset_validation.errors),
                "warnings": list(runtime.dataset_validation.warnings),
            },
        }
    finally:
        if created_resources:
            runtime.close()


def run_rag_question(
    config: RagRuntimeConfig,
    question: str,
    *,
    resources: RuntimeResources | None = None,
    llm: SupportsInvoke | SupportsRunSync | None = None,
    compiled_app: Any | None = None,
) -> dict[str, Any]:
    created_resources = resources is None
    runtime = resources or prepare_runtime(config)
    try:
        config = runtime.config
        try:
            app = compiled_app or build_rag_graph(config, runtime, llm=llm)
            state: RagState = app.invoke(
                {"question": question, "trace": [], "pipeline_errors": []}
            )
        except Exception as graph_exc:
            try:
                state = _run_linear(config, runtime, question=question, llm=llm)
                state["pipeline_errors"] = _append_pipeline_error(
                    state,
                    stage="langgraph_runtime",
                    error=f"{type(graph_exc).__name__}: {graph_exc}",
                )
            except Exception:
                raise graph_exc

        return {
            "question": question,
            "pipeline_mode": config.pipeline_mode,
            "state": state,
            "retrieved_preview": retrieval_preview(list(state.get("retrieved") or [])),
            "filters_summary": summarize_filters(state),
            "context_summary": summarize_context(state),
            "answer_summary": summarize_answer(state),
            "provenance_rows": provenance_rows(state),
            "rewritten_queries": list(state.get("rewritten_queries") or []),
            "retrieval_batches": list(state.get("retrieval_batches") or []),
            "retrieval_mode": str(state.get("retrieval_mode") or "dense_only"),
            "dense_retrieved_count": int(state.get("dense_retrieved_count") or 0),
            "sparse_retrieved_count": int(state.get("sparse_retrieved_count") or 0),
            "fusion_overlap_count": int(state.get("fusion_overlap_count") or 0),
            "graph_expansion": state.get("graph_expansion") or {},
            "reranked": list(state.get("reranked") or []),
            "pipeline_errors": list(state.get("pipeline_errors") or []),
            "index_contract": runtime.index_contract.to_dict(),
            "payload_inspection": runtime.payload_inspection.to_dict(),
            "vector_sizes": {
                "collection_vector_size": runtime.collection_vector_size,
                "query_vector_size": runtime.query_vector_size,
                "dense_vector_name": runtime.dense_vector_name,
                "sparse_enabled": runtime.sparse_enabled,
                "sparse_vector_name": runtime.sparse_vector_name,
                "sparse_artifacts_path": runtime.sparse_artifacts_path,
            },
            "dataset_validation": {
                "is_valid": runtime.dataset_validation.is_valid,
                "counts": dict(runtime.dataset_validation.counts),
                "errors": list(runtime.dataset_validation.errors),
                "warnings": list(runtime.dataset_validation.warnings),
            },
        }
    finally:
        if created_resources:
            runtime.close()

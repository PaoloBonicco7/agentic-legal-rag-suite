from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

from legal_indexing.pipeline import run_indexing_pipeline
from legal_indexing.rag_runtime.config import (
    AdvancedGraphExpansionConfig,
    AdvancedMetadataFilteringConfig,
    AdvancedRagConfig,
    AdvancedRewriteConfig,
    RagRuntimeConfig,
)
from legal_indexing.rag_runtime.graph_adapter import GraphExpansionResult
from legal_indexing.rag_runtime.langgraph_app import (
    _build_nodes,
    prepare_runtime,
    run_rag_question,
    run_rag_retrieval_context,
)
from legal_indexing.rag_runtime.qdrant_retrieval import RetrievedChunk
from legal_indexing.settings import IndexingConfig, make_chunking_profile


class HashEmbedder:
    def __init__(self, size: int = 8) -> None:
        self._size = size

    @property
    def model_name(self) -> str:
        return "hash-test-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._size)])
        return out


class CitationEchoLLM:
    _chunk_re = re.compile(r"chunk_id=([^\s]+)")

    def invoke(self, prompt: str) -> str:
        match = self._chunk_re.search(prompt or "")
        cited = match.group(1) if match else ""
        payload = {
            "answer": "Risposta sintetica basata sul contesto recuperato.",
            "citations": [cited] if cited else [],
            "needs_more_context": False,
        }
        return json.dumps(payload, ensure_ascii=False)


class EmptyAnswerLLM:
    def invoke(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(
            {
                "answer": "   ",
                "citations": [],
                "needs_more_context": True,
            },
            ensure_ascii=False,
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dataset(root: Path) -> Path:
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)

    laws = [
        {
            "law_id": "law:test",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge test",
            "status": "current",
        }
    ]
    articles = [
        {
            "article_id": "law:test#art:1",
            "law_id": "law:test",
            "article_label_norm": "1",
            "is_abrogated": False,
        }
    ]
    chunks = [
        {
            "chunk_id": "law:test#art:1#p:intro#chunk:0",
            "passage_id": "law:test#art:1#p:intro",
            "article_id": "law:test#art:1",
            "law_id": "law:test",
            "chunk_seq": 0,
            "text": "Articolo 1: disciplina generale della procedura.",
            "text_for_embedding": "Articolo 1: disciplina generale della procedura.",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge test",
            "law_status": "current",
            "article_label_norm": "1",
            "article_is_abrogated": False,
            "passage_label": "intro",
            "related_law_ids": [],
            "relation_types": [],
            "inbound_law_ids": [],
            "outbound_law_ids": [],
            "status_confidence": 0.9,
            "status_evidence": [],
            "index_views": ["historical", "current"],
        },
        {
            "chunk_id": "law:test#art:1#p:c1#chunk:0",
            "passage_id": "law:test#art:1#p:c1",
            "article_id": "law:test#art:1",
            "law_id": "law:test",
            "chunk_seq": 0,
            "text": "Comma 1: obbligo di presentare la domanda entro 30 giorni.",
            "text_for_embedding": "Comma 1: obbligo di presentare la domanda entro 30 giorni.",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge test",
            "law_status": "current",
            "article_label_norm": "1",
            "article_is_abrogated": False,
            "passage_label": "c1",
            "related_law_ids": [],
            "relation_types": [],
            "inbound_law_ids": [],
            "outbound_law_ids": [],
            "status_confidence": 0.9,
            "status_evidence": [],
            "index_views": ["historical", "current"],
        },
    ]
    manifest = {
        "schema_version": "laws-graph-pipeline-v1",
        "run_id": "rag_test_run",
        "ready_to_embedding": True,
        "counts": {
            "laws": len(laws),
            "articles": len(articles),
            "notes": 0,
            "edges": 0,
            "events": 0,
            "chunks": len(chunks),
        },
        "hashes": {
            "chunks": "hash_rag_test_chunks",
            "laws": "h1",
            "articles": "h2",
            "notes": "h3",
            "edges": "h4",
            "events": "h5",
        },
    }

    _write_jsonl(ds / "laws.jsonl", laws)
    _write_jsonl(ds / "articles.jsonl", articles)
    _write_jsonl(ds / "notes.jsonl", [])
    _write_jsonl(ds / "edges.jsonl", [])
    _write_jsonl(ds / "events.jsonl", [])
    _write_jsonl(ds / "chunks.jsonl", chunks)
    (ds / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ds


def test_langgraph_baseline_state_transitions_and_output(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"
    embedder = HashEmbedder(size=8)

    indexing_cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="hash-test-embedder",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=40,
            overlap_words_split=5,
        ),
        run_id="rag_indexing_run",
    )
    summary = run_indexing_pipeline(indexing_cfg, embedder=embedder)

    rag_cfg = RagRuntimeConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        indexing_artifacts_root=artifacts_root,
        collection_name=summary.collection_name,
        llm_provider="disabled",
        top_k=3,
        payload_introspection_sample_size=16,
    )
    resources = prepare_runtime(rag_cfg, embedder=embedder)
    try:
        result = run_rag_question(
            rag_cfg,
            "Qual e l'obbligo previsto dal comma 1?",
            resources=resources,
            llm=CitationEchoLLM(),
        )
    finally:
        resources.close()

    state = result["state"]
    assert state["normalized_query"] == "Qual e l'obbligo previsto dal comma 1?"
    assert len(state["retrieved"]) >= 1
    assert isinstance(state["context"], str) and state["context"].strip()
    assert state["answer"]["answer"]
    assert isinstance(state["provenance"], list)

    trace_nodes = [item["node"] for item in state["trace"]]
    assert trace_nodes == [
        "normalize_query",
        "retrieve_top_k",
        "build_context",
        "generate_answer_structured",
    ]

    answer_summary = result["answer_summary"]
    assert "answer" in answer_summary
    assert isinstance(answer_summary["citations"], list)
    assert "retrieved_preview" in result and result["retrieved_preview"]
    assert "provenance_rows" in result and result["provenance_rows"]


def test_langgraph_retrieval_context_only_runs_without_answer_stage(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"
    embedder = HashEmbedder(size=8)

    indexing_cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="hash-test-embedder",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=40,
            overlap_words_split=5,
        ),
        run_id="rag_indexing_run_retrieval_only",
    )
    summary = run_indexing_pipeline(indexing_cfg, embedder=embedder)

    rag_cfg = RagRuntimeConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        indexing_artifacts_root=artifacts_root,
        collection_name=summary.collection_name,
        llm_provider="disabled",
        top_k=3,
        payload_introspection_sample_size=16,
    )
    resources = prepare_runtime(rag_cfg, embedder=embedder)
    try:
        result = run_rag_retrieval_context(
            rag_cfg,
            "Qual e l'obbligo previsto dal comma 1?",
            resources=resources,
        )
    finally:
        resources.close()

    state = result["state"]
    assert state["normalized_query"] == "Qual e l'obbligo previsto dal comma 1?"
    assert len(state["retrieved"]) >= 1
    assert isinstance(state["context"], str) and state["context"].strip()
    assert "answer" not in state
    assert "provenance" not in state

    trace_nodes = [item["node"] for item in state["trace"]]
    assert trace_nodes == [
        "normalize_query",
        "retrieve_top_k",
        "build_context",
    ]
    assert result["stage"] == "retrieval_context_only"
    assert "retrieved_preview" in result and result["retrieved_preview"]
    assert "context_summary" in result and result["context_summary"]


def test_langgraph_advanced_retrieval_context_only_runs_without_answer_stage(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"
    embedder = HashEmbedder(size=8)

    indexing_cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="hash-test-embedder",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=40,
            overlap_words_split=5,
        ),
        run_id="rag_indexing_run_retrieval_only_advanced",
    )
    summary = run_indexing_pipeline(indexing_cfg, embedder=embedder)

    rag_cfg = RagRuntimeConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        indexing_artifacts_root=artifacts_root,
        collection_name=summary.collection_name,
        llm_provider="disabled",
        pipeline_mode="advanced",
        top_k=3,
        payload_introspection_sample_size=16,
        advanced=AdvancedRagConfig(
            rewrite=AdvancedRewriteConfig(enabled=False),
        ),
    )
    resources = prepare_runtime(rag_cfg, embedder=embedder)
    try:
        result = run_rag_retrieval_context(
            rag_cfg,
            "Qual e l'obbligo previsto dal comma 1?",
            resources=resources,
            llm=CitationEchoLLM(),
            pipeline_mode="advanced",
        )
    finally:
        resources.close()

    state = result["state"]
    assert isinstance(state["context"], str) and state["context"].strip()
    assert "answer" not in state
    assert "provenance" not in state
    trace_nodes = [item["node"] for item in state["trace"]]
    assert trace_nodes == [
        "normalize_query",
        "rewrite_or_decompose_query",
        "build_metadata_filter",
        "retrieve_multi",
        "graph_expand",
        "rerank_candidates",
        "build_context",
    ]
    assert result["pipeline_mode"] == "advanced"


def test_langgraph_advanced_pipeline_state_transitions(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"
    embedder = HashEmbedder(size=8)

    indexing_cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="hash-test-embedder",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=40,
            overlap_words_split=5,
        ),
        run_id="rag_indexing_run_advanced",
    )
    summary = run_indexing_pipeline(indexing_cfg, embedder=embedder)

    rag_cfg = RagRuntimeConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        indexing_artifacts_root=artifacts_root,
        collection_name=summary.collection_name,
        llm_provider="disabled",
        pipeline_mode="advanced",
        top_k=3,
        payload_introspection_sample_size=16,
    )
    resources = prepare_runtime(rag_cfg, embedder=embedder)
    try:
        result = run_rag_question(
            rag_cfg,
            "Qual e l'obbligo previsto dal comma 1?",
            resources=resources,
            llm=CitationEchoLLM(),
        )
    finally:
        resources.close()

    state = result["state"]
    trace_nodes = [item["node"] for item in state["trace"]]
    assert trace_nodes == [
        "normalize_query",
        "rewrite_or_decompose_query",
        "build_metadata_filter",
        "retrieve_multi",
        "graph_expand",
        "rerank_candidates",
        "build_context",
        "generate_answer_structured",
    ]
    assert result["pipeline_mode"] == "advanced"
    assert isinstance(result["rewritten_queries"], list)
    assert isinstance(result["retrieval_batches"], list)
    assert result["retrieval_mode"] in {"hybrid", "dense_only", "fallback_dense"}
    assert isinstance(result["dense_retrieved_count"], int)
    assert isinstance(result["sparse_retrieved_count"], int)
    assert isinstance(result["fusion_overlap_count"], int)
    assert isinstance(result["graph_expansion"], dict)
    assert isinstance(result["reranked"], list)
    assert isinstance(result["pipeline_errors"], list)


def test_langgraph_advanced_guard_replaces_empty_answer(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"
    embedder = HashEmbedder(size=8)

    indexing_cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="hash-test-embedder",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=40,
            overlap_words_split=5,
        ),
        run_id="rag_indexing_run_advanced_guard",
    )
    summary = run_indexing_pipeline(indexing_cfg, embedder=embedder)

    rag_cfg = RagRuntimeConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        indexing_artifacts_root=artifacts_root,
        collection_name=summary.collection_name,
        llm_provider="disabled",
        pipeline_mode="advanced",
        top_k=3,
        payload_introspection_sample_size=16,
    )
    resources = prepare_runtime(rag_cfg, embedder=embedder)
    try:
        result = run_rag_question(
            rag_cfg,
            "Qual e l'obbligo previsto dal comma 1?",
            resources=resources,
            llm=EmptyAnswerLLM(),
        )
    finally:
        resources.close()

    answer = result["answer_summary"]
    assert answer["answer"].strip()
    assert answer["answer_source"] == "fallback"
    assert answer["was_empty_before_guard"] is True
    assert any(
        "empty_answer_detected" in str(err.get("error") or "")
        for err in result["pipeline_errors"]
    )


def _test_chunk(*, chunk_id: str, law_id: str, article_id: str, text: str = "testo") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.5,
        text=text,
        point_id=chunk_id,
        payload={
            "chunk_id": chunk_id,
            "law_id": law_id,
            "article_id": article_id,
            "law_status": "current",
            "relation_types": ["AMENDS"],
            "source_chunk_ids": [chunk_id],
            "source_passage_ids": [f"{article_id}#p:intro"],
        },
    )


def _law_ids_from_filter(query_filter: object | None) -> set[str]:
    if query_filter is None:
        return set()
    out: set[str] = set()
    must = getattr(query_filter, "must", None) or []
    for cond in must:
        key = getattr(cond, "key", None)
        if key != "law_id":
            continue
        match = getattr(cond, "match", None)
        if match is None:
            continue
        any_values = getattr(match, "any", None)
        if any_values is not None:
            for value in any_values:
                cur = str(value).strip()
                if cur:
                    out.add(cur)
        value = getattr(match, "value", None)
        if value is not None:
            cur = str(value).strip()
            if cur:
                out.add(cur)
    return out


class _GraphRetrieverStub:
    def __init__(self, related_doc: RetrievedChunk) -> None:
        self.related_doc = related_doc
        self.last_graph_law_ids: set[str] = set()
        self.last_graph_top_k: int | None = None

    def query_hybrid(self, query: str, **kwargs):  # type: ignore[no-untyped-def]
        _ = query
        query_filter = kwargs.get("query_filter")
        self.last_graph_top_k = int(kwargs.get("top_k") or 0)
        law_ids = _law_ids_from_filter(query_filter)
        self.last_graph_law_ids = set(law_ids)
        if "law:b" in law_ids and "law:a" not in law_ids:
            return SimpleNamespace(retrieved=(self.related_doc,))
        return SimpleNamespace(retrieved=tuple())

    def query(self, query: str, **kwargs):  # type: ignore[no-untyped-def]
        _ = query
        query_filter = kwargs.get("query_filter")
        self.last_graph_top_k = int(kwargs.get("top_k") or 0)
        law_ids = _law_ids_from_filter(query_filter)
        self.last_graph_law_ids = set(law_ids)
        if "law:b" in law_ids and "law:a" not in law_ids:
            return [self.related_doc]
        return []


class _GraphAdapterStub:
    def __init__(self, related_law_id: str = "law:b") -> None:
        self.related_law_id = related_law_id

    def expand_from_retrieved(self, retrieved: list[RetrievedChunk], *, max_related_laws: int) -> GraphExpansionResult:
        _ = max_related_laws
        seed_law_ids = tuple(sorted({doc.law_id for doc in retrieved if doc.law_id}))
        seed_chunk_ids = tuple(sorted({doc.chunk_id for doc in retrieved if doc.chunk_id}))
        return GraphExpansionResult(
            seed_chunk_ids=seed_chunk_ids,
            seed_law_ids=seed_law_ids,
            seed_article_ids=tuple(),
            seed_passage_ids=tuple(),
            related_law_ids=(self.related_law_id,),
            related_article_ids=tuple(),
            related_chunk_ids=tuple(),
            edge_hits=1,
            event_hits=0,
        )


class _LawCatalogStub:
    def resolve(self, query: str):  # type: ignore[no-untyped-def]
        _ = query
        return SimpleNamespace(law_ids=tuple(), article_ids=tuple())


class _JsonLLMStub:
    def invoke(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(
            {"answer": "ok", "citations": [], "needs_more_context": False},
            ensure_ascii=False,
        )


def _build_graph_test_nodes(
    *,
    relax_law_article_filters: bool = True,
    graph_expansion: AdvancedGraphExpansionConfig | None = None,
) -> tuple[dict[str, object], _GraphRetrieverStub]:
    related_doc = _test_chunk(
        chunk_id="law:b#art:1#chunk:0",
        law_id="law:b",
        article_id="law:b#art:1",
        text="La legge B modifica la legge A.",
    )
    retriever = _GraphRetrieverStub(related_doc=related_doc)
    resources = SimpleNamespace(
        retriever=retriever,
        graph_adapter=_GraphAdapterStub(),
        law_catalog=_LawCatalogStub(),
    )
    config = RagRuntimeConfig(
        llm_provider="disabled",
        pipeline_mode="advanced",
        advanced=AdvancedRagConfig(
            rewrite=AdvancedRewriteConfig(enabled=False),
            metadata_filtering=AdvancedMetadataFilteringConfig(
                mode="hybrid",
                enable_heuristics=True,
                relax_law_article_filters_on_relation_queries=relax_law_article_filters,
            ),
            graph_expansion=graph_expansion
            or AdvancedGraphExpansionConfig(
                enabled=True,
                force_on_relation_queries=True,
            ),
        ),
    )
    nodes = _build_nodes(config, resources, llm=_JsonLLMStub())
    return nodes, retriever


def test_graph_expand_keeps_working_when_metadata_has_law_ids() -> None:
    nodes, retriever = _build_graph_test_nodes(relax_law_article_filters=False)
    seed_doc = _test_chunk(
        chunk_id="law:a#art:1#chunk:0",
        law_id="law:a",
        article_id="law:a#art:1",
        text="La legge A e la norma di riferimento.",
    )
    state = {"question": "Quali leggi hanno modificato law:a?", "trace": [], "pipeline_errors": []}
    state.update(nodes["normalize_query"](state))
    state.update(nodes["build_metadata_filter"](state))
    assert state["metadata_filter_decision"]["law_ids"] == ["law:a"]
    state["retrieved"] = [seed_doc]
    state["retrieval_batches"] = [
        {
            "name": "primary",
            "query": state["normalized_query"],
            "top_k": 8,
            "retrieved_chunk_ids": [seed_doc.chunk_id],
            "retrieved_count": 1,
        }
    ]

    updates = nodes["graph_expand"](state)
    assert updates["graph_expansion"]["graph_retrieved_count"] > 0
    assert "law:a" not in retriever.last_graph_law_ids
    assert "law:b" in retriever.last_graph_law_ids


def test_query_with_non_legal_digits_does_not_disable_graph() -> None:
    nodes, _ = _build_graph_test_nodes(relax_law_article_filters=True)
    seed_doc = _test_chunk(
        chunk_id="law:a#art:1#chunk:0",
        law_id="law:a",
        article_id="law:a#art:1",
        text="Procedura contributi 12345.",
    )
    state = {
        "question": "Quali effetti produce la procedura 12345 per i contributi?",
        "trace": [],
        "pipeline_errors": [],
        "normalized_query": "Quali effetti produce la procedura 12345 per i contributi?",
        "retrieved": [seed_doc],
        "retrieval_batches": [],
    }
    updates = nodes["graph_expand"](state)
    assert updates["graph_expansion"]["enabled"] is True
    assert updates["graph_expansion"]["reason"] != "gated_specific_query"


def test_relation_query_forces_graph_and_retrieves_related_laws() -> None:
    nodes, _ = _build_graph_test_nodes(relax_law_article_filters=True)
    seed_doc = _test_chunk(
        chunk_id="law:a#art:1#chunk:0",
        law_id="law:a",
        article_id="law:a#art:1",
        text="La legge A disciplina la materia.",
    )
    state = {"question": "Quali leggi hanno modificato law:a?", "trace": [], "pipeline_errors": []}
    state.update(nodes["normalize_query"](state))
    state.update(nodes["build_metadata_filter"](state))
    state["retrieved"] = [seed_doc]
    state["retrieval_batches"] = []
    updates = nodes["graph_expand"](state)
    assert updates["graph_expansion"]["enabled"] is True
    assert updates["graph_expansion"]["reason"] == "forced_relation_query"
    assert updates["graph_expansion"]["graph_retrieved_count"] > 0


def test_specific_query_mode_disable_keeps_graph_off() -> None:
    nodes, _ = _build_graph_test_nodes(
        relax_law_article_filters=True,
        graph_expansion=AdvancedGraphExpansionConfig(
            enabled=True,
            force_on_relation_queries=False,
            specific_query_mode="disable",
        ),
    )
    seed_doc = _test_chunk(
        chunk_id="law:a#art:1#chunk:0",
        law_id="law:a",
        article_id="law:a#art:1",
        text="La legge A disciplina la materia.",
    )
    state = {
        "question": "Qual e il contenuto dell'art. 1 della law:a?",
        "trace": [],
        "pipeline_errors": [],
        "normalized_query": "Qual e il contenuto dell'art. 1 della law:a?",
        "retrieved": [seed_doc],
        "retrieval_batches": [],
    }
    updates = nodes["graph_expand"](state)
    assert updates["graph_expansion"]["enabled"] is False
    assert updates["graph_expansion"]["reason"] == "gated_specific_query"
    assert updates["graph_expansion"]["specific_query_mode_applied"] == "disable"


def test_specific_query_mode_minimal_uses_reduced_top_k() -> None:
    nodes, retriever = _build_graph_test_nodes(
        relax_law_article_filters=True,
        graph_expansion=AdvancedGraphExpansionConfig(
            enabled=True,
            force_on_relation_queries=False,
            graph_retrieval_top_k=6,
            specific_query_mode="minimal",
            specific_query_graph_retrieval_top_k=2,
            specific_query_max_related_laws=2,
        ),
    )
    seed_doc = _test_chunk(
        chunk_id="law:a#art:1#chunk:0",
        law_id="law:a",
        article_id="law:a#art:1",
        text="La legge A disciplina la materia.",
    )
    state = {
        "question": "Qual e il contenuto dell'art. 1 della law:a?",
        "trace": [],
        "pipeline_errors": [],
        "normalized_query": "Qual e il contenuto dell'art. 1 della law:a?",
        "retrieved": [seed_doc],
        "retrieval_batches": [],
    }
    updates = nodes["graph_expand"](state)
    assert updates["graph_expansion"]["enabled"] is True
    assert updates["graph_expansion"]["reason"] == "gated_specific_query"
    assert updates["graph_expansion"]["specific_query_mode_applied"] == "minimal"
    assert updates["graph_expansion"]["graph_retrieved_count"] > 0
    assert retriever.last_graph_top_k == 2


def test_specific_query_mode_full_keeps_default_top_k() -> None:
    nodes, retriever = _build_graph_test_nodes(
        relax_law_article_filters=True,
        graph_expansion=AdvancedGraphExpansionConfig(
            enabled=True,
            force_on_relation_queries=False,
            graph_retrieval_top_k=6,
            specific_query_mode="full",
            specific_query_graph_retrieval_top_k=2,
            specific_query_max_related_laws=2,
        ),
    )
    seed_doc = _test_chunk(
        chunk_id="law:a#art:1#chunk:0",
        law_id="law:a",
        article_id="law:a#art:1",
        text="La legge A disciplina la materia.",
    )
    state = {
        "question": "Qual e il contenuto dell'art. 1 della law:a?",
        "trace": [],
        "pipeline_errors": [],
        "normalized_query": "Qual e il contenuto dell'art. 1 della law:a?",
        "retrieved": [seed_doc],
        "retrieval_batches": [],
    }
    updates = nodes["graph_expand"](state)
    assert updates["graph_expansion"]["enabled"] is True
    assert updates["graph_expansion"]["reason"] == "gated_specific_query"
    assert updates["graph_expansion"]["specific_query_mode_applied"] == "full"
    assert retriever.last_graph_top_k == 6

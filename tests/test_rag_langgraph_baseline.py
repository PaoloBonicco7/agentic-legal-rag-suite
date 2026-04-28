from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from legal_indexing.pipeline import run_indexing_pipeline
from legal_indexing.rag_runtime.config import RagRuntimeConfig
from legal_indexing.rag_runtime.langgraph_app import (
    prepare_runtime,
    run_rag_question,
    run_rag_retrieval_context,
)
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

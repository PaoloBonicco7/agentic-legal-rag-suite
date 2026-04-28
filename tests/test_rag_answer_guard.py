from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from legal_indexing.pipeline import run_indexing_pipeline
from legal_indexing.rag_runtime.config import (
    AdvancedAnswerGuardConfig,
    AdvancedMultiRetrievalConfig,
    AdvancedRagConfig,
    RagRuntimeConfig,
)
from legal_indexing.rag_runtime.langgraph_app import (
    _build_nodes,
    prepare_runtime,
    run_rag_question,
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        }
    ]
    manifest = {
        "schema_version": "laws-graph-pipeline-v1",
        "run_id": "rag_answer_guard_test",
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
            "chunks": "hash_rag_answer_guard",
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
    (ds / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ds


class EmptyThenValidLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        _ = prompt
        self.calls += 1
        if self.calls == 1:
            payload = {
                "answer": "   ",
                "citations": [],
                "needs_more_context": True,
            }
        else:
            payload = {
                "answer": "Risposta di retry affidabile.",
                "citations": [],
                "needs_more_context": False,
            }
        return json.dumps(payload, ensure_ascii=False)


class AlwaysEmptyLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        _ = prompt
        self.calls += 1
        return json.dumps(
            {
                "answer": " ",
                "citations": [],
                "needs_more_context": True,
            },
            ensure_ascii=False,
        )


def _build_runtime(tmp_path: Path, *, run_id: str, query_language: str = "it") -> tuple[RagRuntimeConfig, Any]:
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
        run_id=run_id,
    )
    summary = run_indexing_pipeline(indexing_cfg, embedder=embedder)

    rag_cfg = RagRuntimeConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        indexing_artifacts_root=artifacts_root,
        collection_name=summary.collection_name,
        llm_provider="disabled",
        pipeline_mode="advanced",
        query_language=query_language,
        top_k=3,
        payload_introspection_sample_size=16,
    )
    resources = prepare_runtime(rag_cfg, embedder=embedder)
    return rag_cfg, resources


def test_answer_guard_retries_when_first_pass_is_empty(tmp_path: Path) -> None:
    rag_cfg, resources = _build_runtime(tmp_path, run_id="answer_guard_retry")
    llm = EmptyThenValidLLM()
    try:
        result = run_rag_question(
            rag_cfg,
            "Qual e l'obbligo previsto dal comma 1?",
            resources=resources,
            llm=llm,
        )
    finally:
        resources.close()

    answer = result["answer_summary"]
    assert answer["answer"] == "Risposta di retry affidabile."
    assert answer["answer_source"] == "retry"
    assert answer["was_empty_before_guard"] is True
    assert answer["needs_more_context"] is False
    assert llm.calls >= 2


def test_answer_guard_fallback_when_retry_is_still_empty(tmp_path: Path) -> None:
    rag_cfg, resources = _build_runtime(
        tmp_path,
        run_id="answer_guard_fallback",
        query_language="en",
    )
    llm = AlwaysEmptyLLM()
    try:
        result = run_rag_question(
            rag_cfg,
            "What is the obligation in article 1?",
            resources=resources,
            llm=llm,
        )
    finally:
        resources.close()

    answer = result["answer_summary"]
    assert answer["answer"].strip()
    assert answer["answer"] == rag_cfg.advanced.answer_guard.fallback_message_en
    assert answer["answer_source"] == "fallback"
    assert answer["was_empty_before_guard"] is True
    assert answer["needs_more_context"] is True
    assert answer["citations"] == []


class NeedsMoreThenResolvedLLM:
    _chunk_re = re.compile(r"chunk_id=([^\s]+)")

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        match = self._chunk_re.search(prompt or "")
        citation = match.group(1) if match else "c0"
        if self.calls == 1:
            payload = {
                "answer": "Serve più contesto, ma provo a sintetizzare.",
                "citations": [citation],
                "needs_more_context": True,
            }
        else:
            payload = {
                "answer": "Risposta best-effort con evidenza sufficiente.",
                "citations": [citation],
                "needs_more_context": False,
            }
        return json.dumps(payload, ensure_ascii=False)


def _retrieved_chunks(n: int) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for i in range(n):
        chunk_id = f"law:test:{i}#chunk:0"
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                score=0.5 - (i * 0.001),
                text=f"Testo normativo {i}",
                point_id=chunk_id,
                payload={
                    "chunk_id": chunk_id,
                    "law_id": f"law:test:{i % 4}",
                    "article_id": f"law:test:{i % 4}#art:{i % 3}",
                    "source_chunk_ids": [chunk_id],
                    "source_passage_ids": [f"law:test:{i % 4}#art:{i % 3}#p:intro"],
                },
            )
        )
    return out


def test_answer_guard_retries_on_needs_more_context_when_retrieval_is_strong() -> None:
    llm = NeedsMoreThenResolvedLLM()
    cfg = RagRuntimeConfig(
        llm_provider="disabled",
        pipeline_mode="advanced",
        advanced=AdvancedRagConfig(
            multi_retrieval=AdvancedMultiRetrievalConfig(top_k_primary=12, top_k_secondary=6),
            answer_guard=AdvancedAnswerGuardConfig(
                retry_on_empty_answer=True,
                max_empty_retries=1,
                retry_on_needs_more_context=True,
                max_needs_more_retries=1,
            ),
        ),
    )
    nodes = _build_nodes(
        cfg,
        resources=SimpleNamespace(
            retriever=SimpleNamespace(),
            graph_adapter=SimpleNamespace(),
            law_catalog=SimpleNamespace(),
        ),
        llm=llm,
    )
    retrieved = _retrieved_chunks(12)
    context_ids = [doc.chunk_id for doc in retrieved[:8]]
    context_text = "\n\n".join([f"chunk_id={cid}\nTesto di supporto." for cid in context_ids])
    state = {
        "question": "Qual e la disciplina applicabile?",
        "context": context_text,
        "retrieved": retrieved,
        "retrieval_mode": "hybrid",
        "context_summary": {
            "included_count": 8,
            "included_chunk_ids": context_ids,
            "total_candidates": 12,
        },
        "pipeline_errors": [],
        "trace": [],
        "reranked": [],
        "retrieval_batches": [],
    }

    updates = nodes["generate_answer_structured"](state)
    answer = updates["answer"]
    assert answer["answer_source"] == "retry_needs_more_context"
    assert answer["needs_more_context"] is False
    assert llm.calls >= 2

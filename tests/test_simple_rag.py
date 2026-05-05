from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.oracle_context_evaluation.io import write_json, write_jsonl
from legal_rag.simple_rag import (
    SIMPLE_RAG_PROMPT_VERSION,
    SIMPLE_RAG_SCHEMA_VERSION,
    RetrievedChunkRecord,
    SimpleRagConfig,
    build_context,
    build_query_embedder,
    run_mcq,
    run_simple_rag,
    resolve_index_manifest_path,
    search_dense,
)
from legal_rag.simple_rag.retrieval import connect_qdrant


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.lower()
            if "second" in normalized or "historical" in normalized:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0, 0.0])
        return vectors


class FakeStructuredClient:
    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        citation = _first_context_chunk_id(prompt)
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "citation_chunk_ids": [citation], "short_rationale": "fake"}}
        if "answer_text" in properties:
            return {"structured": {"answer_text": "Risposta fake", "citation_chunk_ids": [citation], "short_rationale": "fake"}}
        if "score" in properties:
            return {"structured": {"score": 2, "explanation": "Correct fake answer."}}
        raise AssertionError(f"Unexpected schema: {payload_schema}")


class InvalidCitationClient(FakeStructuredClient):
    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "citation_chunk_ids": ["missing-chunk"]}}
        if "answer_text" in properties:
            return {"structured": {"answer_text": "Risposta fake", "citation_chunk_ids": ["missing-chunk"]}}
        return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)


class InvalidJudgeClient(FakeStructuredClient):
    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        if "score" in properties:
            return {"structured": {"score": 3, "explanation": ""}}
        return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)


class ObservedSlowClient(FakeStructuredClient):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.02)
            return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)
        finally:
            with self._lock:
                self._active -= 1


def _first_context_chunk_id(prompt: str) -> str:
    match = re.search(r"chunk_id: ([^\s]+)", prompt)
    return match.group(1) if match else "c1"


def _make_clean_inputs(tmp_path: Path, *, count: int = 2) -> Path:
    evaluation_dir = tmp_path / "evaluation_clean"
    evaluation_dir.mkdir()
    mcq_records = []
    no_hint_records = []
    for idx in range(1, count + 1):
        qid = f"eval-{idx:04d}"
        stem = "Second question?" if idx == 2 else f"Question {idx}?"
        correct_label = "A" if idx != 2 else "B"
        mcq_records.append(
            {
                "qid": qid,
                "source_position": idx,
                "level": "L1",
                "question_stem": stem,
                "options": {"A": "Answer A", "B": "Answer B", "C": "C", "D": "D", "E": "E", "F": "F"},
                "correct_label": correct_label,
                "correct_answer": "Answer A",
                "expected_references": ["Law - Art. 1"],
            }
        )
        no_hint_records.append(
            {
                "qid": qid,
                "source_position": idx,
                "level": "L1",
                "question": stem,
                "correct_answer": "Answer A",
                "expected_references": ["Law - Art. 1"],
                "linked_mcq_qid": qid,
            }
        )
    write_jsonl(evaluation_dir / "questions_mcq.jsonl", mcq_records)
    write_jsonl(evaluation_dir / "questions_no_hint.jsonl", no_hint_records)
    write_json(evaluation_dir / "evaluation_manifest.json", {"schema_version": "evaluation-dataset-v1"})
    return evaluation_dir


def _write_index_manifest(
    tmp_path: Path,
    *,
    run_id: str = "20260504_000000",
    collection_name: str = "test_collection",
    created_at: str = "2026-05-04T00:00:00Z",
) -> Path:
    run_dir = tmp_path / "indexing_runs" / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "index_manifest.json"
    write_json(
        path,
        {
            "schema_version": "indexing-contract-v1",
            "created_at": created_at,
            "run_id": run_id,
            "collection_name": collection_name,
            "ready_for_retrieval": True,
            "embedding": {"provider": "utopia", "model": "fake-embedding", "configured_model": "fake-embedding", "mode": "auto"},
            "config": {"embedding_provider": "utopia", "embedding_model": "fake-embedding"},
        },
    )
    return path


def _payload(chunk_id: str, *, text: str, law_status: str = "current") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "law_id": f"law-{chunk_id}",
        "article_id": f"law-{chunk_id}#art:1",
        "text": text,
        "law_title": f"Law {chunk_id}",
        "law_status": law_status,
        "article_status": "current",
        "index_views": [law_status],
    }


def _make_qdrant(collection_name: str = "test_collection") -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            qmodels.PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload=_payload("c1", text="Current legal text.")),
            qmodels.PointStruct(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload=_payload("c2", text="Historical legal text.", law_status="historical")),
        ],
        wait=True,
    )
    return client


def _make_named_qdrant(collection_name: str = "named_collection") -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            qmodels.PointStruct(id=1, vector={"dense": [1.0, 0.0, 0.0, 0.0]}, payload=_payload("c1", text="Current legal text.")),
            qmodels.PointStruct(id=2, vector={"dense": [0.0, 1.0, 0.0, 0.0]}, payload=_payload("c2", text="Historical legal text.")),
        ],
        wait=True,
    )
    return client


def test_latest_index_manifest_uses_most_recent_run_not_lexicographic_name(tmp_path: Path) -> None:
    older = _write_index_manifest(tmp_path, run_id="notebook_demo_20260505_084835", created_at="2026-05-05T08:48:35Z")
    newer = _write_index_manifest(tmp_path, run_id="20260505_084836", created_at="2026-05-05T09:09:12Z")
    os.utime(older.parent, (100.0, 100.0))
    os.utime(newer.parent, (200.0, 200.0))

    resolved = resolve_index_manifest_path(str(tmp_path / "indexing_runs" / "<latest>" / "index_manifest.json"))

    assert resolved == newer


def test_latest_index_manifest_prefers_real_run_over_newer_demo(tmp_path: Path) -> None:
    demo = _write_index_manifest(
        tmp_path,
        run_id="notebook_demo_20260505_999999",
        collection_name="legal_chunks_notebook_demo",
        created_at="2026-05-05T23:59:59Z",
    )
    real = _write_index_manifest(
        tmp_path,
        run_id="20260505_084836",
        collection_name="legal_chunks",
        created_at="2026-05-05T09:09:12Z",
    )
    os.utime(demo.parent, (300.0, 300.0))
    os.utime(real.parent, (100.0, 100.0))

    resolved = resolve_index_manifest_path(str(tmp_path / "indexing_runs" / "<latest>" / "index_manifest.json"))

    assert resolved == real


def test_qdrant_connection_uses_nested_manifest_url(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr("legal_rag.simple_rag.retrieval.QdrantClient", FakeQdrantClient)

    connect_qdrant(
        SimpleRagConfig(index_dir="local-path"),
        {"qdrant": {"url": "http://127.0.0.1:6333", "path": "server-storage"}, "config": {"index_dir": "config-path"}},
    )

    assert calls == {"url": "http://127.0.0.1:6333"}


def test_query_embedder_accepts_current_index_manifest_shape(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_build_embedder(config: Any) -> FakeEmbedder:
        captured.update(config.model_dump())
        return FakeEmbedder()

    monkeypatch.setattr("legal_rag.simple_rag.runner.build_embedder", fake_build_embedder)

    embedder = build_query_embedder(
        SimpleRagConfig(env_file=None, api_key="secret"),
        {
            "embedding": {
                "backend": "utopia",
                "resolved_model": "SLURM.nomic-embed-text:latest",
                "vector_size": 768,
            },
            "config": {
                "batch_size": 128,
                "embedding_timeout_seconds": 120.0,
                "utopia_embed_api_mode": "ollama",
                "utopia_embed_url": "https://utopia.hpc4ai.unito.it/ollama/api/embeddings",
            },
        },
    )

    assert isinstance(embedder, FakeEmbedder)
    assert captured["embedding_backend"] == "utopia"
    assert captured["embedding_model"] == "SLURM.nomic-embed-text:latest"
    assert captured["batch_size"] == 128
    assert captured["hybrid_enabled"] is False
    assert captured["embedding_api_key"] == "secret"


def test_dense_search_supports_named_dense_vectors() -> None:
    rows = search_dense(
        _make_named_qdrant(),
        collection_name="named_collection",
        embedder=FakeEmbedder(),
        query_text="first question",
        limit=1,
        static_filters={},
    )

    assert [row.chunk_id for row in rows] == ["c1"]


def test_run_simple_rag_exports_contract_files(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path)
    index_manifest = _write_index_manifest(tmp_path)
    output_dir = tmp_path / "simple_rag"

    manifest = run_simple_rag(
        SimpleRagConfig(
            evaluation_dir=str(evaluation_dir),
            index_manifest_path=str(index_manifest),
            output_dir=str(output_dir),
            api_key="secret-value",
            chat_model="fake-answer",
            judge_model="fake-judge",
            max_concurrency=1,
        ),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeEmbedder(),
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "simple_rag_manifest.json",
        "mcq_results.jsonl",
        "no_hint_results.jsonl",
        "simple_rag_summary.json",
        "quality_report.md",
    }
    assert manifest["schema_version"] == SIMPLE_RAG_SCHEMA_VERSION
    assert manifest["prompt_version"] == SIMPLE_RAG_PROMPT_VERSION
    assert "api_key" not in manifest["config"]
    assert manifest["config"]["api_key_present"] is True
    assert manifest["models"]["answer_model"] == "fake-answer"
    assert manifest["models"]["judge_model"] == "fake-judge"
    assert set(manifest["source_hashes"]) == {"questions_mcq", "questions_no_hint", "evaluation_manifest", "index_manifest"}
    assert set(manifest["output_hashes"]) == {"mcq_results", "no_hint_results", "simple_rag_summary", "quality_report"}

    rows = [
        json.loads(line)
        for line in (output_dir / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["retrieved_count"] == 2
    assert rows[0]["context_count"] == 2
    assert rows[0]["retrieved_chunk_ids"][0] == "c1"
    assert rows[0]["citations"][0]["chunk_id"] == "c1"
    assert rows[0]["score"] == 1
    assert rows[1]["score"] == 0


def test_context_builder_enforces_chunk_and_character_budgets() -> None:
    chunks = [
        RetrievedChunkRecord(chunk_id="c1", score=1.0, text="abcdef", payload={"law_id": "l1", "article_id": "a1"}),
        RetrievedChunkRecord(chunk_id="c2", score=0.5, text="second", payload={"law_id": "l2", "article_id": "a2"}),
    ]

    selected, context_text = build_context(chunks, max_context_chunks=1, max_context_chars=3)

    assert [chunk.chunk_id for chunk in selected] == ["c1"]
    assert "abc" in context_text
    assert "abcdef" not in context_text
    assert "c2" not in context_text


def test_invalid_citation_is_recorded_as_error(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path, count=1)
    index_manifest = _write_index_manifest(tmp_path)
    output_dir = tmp_path / "invalid_citation"

    run_simple_rag(
        SimpleRagConfig(
            evaluation_dir=str(evaluation_dir),
            index_manifest_path=str(index_manifest),
            output_dir=str(output_dir),
            chat_model="fake",
            max_concurrency=1,
        ),
        client=InvalidCitationClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeEmbedder(),
    )

    mcq_row = json.loads((output_dir / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    no_hint_row = json.loads((output_dir / "no_hint_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "citation_error" in mcq_row["error"]
    assert "citation_error" in no_hint_row["error"]
    assert mcq_row["citations"] == []


def test_empty_retrieval_is_counted_separately(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path, count=1)
    index_manifest = _write_index_manifest(tmp_path)
    output_dir = tmp_path / "empty_retrieval"

    manifest = run_simple_rag(
        SimpleRagConfig(
            evaluation_dir=str(evaluation_dir),
            index_manifest_path=str(index_manifest),
            output_dir=str(output_dir),
            chat_model="fake",
            static_filters={"law_status": "missing"},
            max_concurrency=1,
        ),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeEmbedder(),
    )

    row = json.loads((output_dir / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["retrieved_count"] == 0
    assert row["context_count"] == 0
    assert row["error"] == "empty_retrieval"
    assert manifest["counts"]["empty_retrieval"] == 2


def test_static_filters_are_applied_to_retrieval(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path, count=1)
    index_manifest = _write_index_manifest(tmp_path)
    output_dir = tmp_path / "filtered"

    run_simple_rag(
        SimpleRagConfig(
            evaluation_dir=str(evaluation_dir),
            index_manifest_path=str(index_manifest),
            output_dir=str(output_dir),
            chat_model="fake",
            static_filters={"law_status": "historical"},
            top_k=1,
            max_concurrency=1,
        ),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeEmbedder(),
    )

    row = json.loads((output_dir / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["retrieved_chunk_ids"] == ["c2"]
    assert row["retrieved_law_ids"] == ["law-c2"]


def test_invalid_judge_result_is_recorded_as_error(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path, count=1)
    index_manifest = _write_index_manifest(tmp_path)
    output_dir = tmp_path / "invalid_judge"

    run_simple_rag(
        SimpleRagConfig(
            evaluation_dir=str(evaluation_dir),
            index_manifest_path=str(index_manifest),
            output_dir=str(output_dir),
            chat_model="fake",
            max_concurrency=1,
        ),
        client=InvalidJudgeClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeEmbedder(),
    )

    row = json.loads((output_dir / "no_hint_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["predicted_answer"] == "Risposta fake"
    assert row["judge_score"] is None
    assert "judge_error" in row["error"]


def test_mcq_calls_are_parallelized_but_output_order_is_stable(tmp_path: Path) -> None:
    records = [
        {
            "qid": f"eval-{idx:04d}",
            "level": "L1",
            "question_stem": f"Question {idx}?",
            "options": {"A": "Answer A", "B": "Answer B", "C": "C", "D": "D", "E": "E", "F": "F"},
            "correct_label": "A",
        }
        for idx in range(1, 5)
    ]
    llm_client = ObservedSlowClient()

    rows = run_mcq(
        records=records,
        llm_client=llm_client,
        qdrant_client=_make_qdrant(),
        embedder=FakeEmbedder(),
        collection_name="test_collection",
        config=SimpleRagConfig(chat_model="fake", max_concurrency=4),
    )

    assert llm_client.max_active > 1
    assert [row["qid"] for row in rows] == [record["qid"] for record in records]
    assert all(row["score"] == 1 for row in rows)

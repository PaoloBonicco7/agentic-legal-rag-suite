from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.advanced_graph_rag import (
    ADVANCED_RAG_PROMPT_VERSION,
    ADVANCED_RAG_SCHEMA_VERSION,
    AdvancedNoHintAnswerOutput,
    AdvancedRagConfig,
    run_advanced_graph_rag,
)
from legal_rag.advanced_graph_rag.retrieval import GraphIndex, connect_qdrant, embed_sparse_query, expand_with_graph
from legal_rag.oracle_context_evaluation.io import sha256_file, write_json, write_jsonl
from legal_rag.simple_rag.models import RetrievedChunkRecord


LAW_1 = "vda:lr:2000-01-01:1"
LAW_2 = "vda:lr:2000-01-01:2"
LAW_3 = "vda:lr:2000-01-01:3"


class FakeHybridEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_sparse_texts(self, texts: list[str]) -> list[dict[str, list[float] | list[int]]]:
        return [{"indices": [10], "values": [1.0]} for _ in texts]


class DenseOnlyEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class TupleSparseEmbedder(FakeHybridEmbedder):
    def embed_sparse_texts(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [([10, 20], [1.0, 0.5]) for _ in texts]


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
        if "scores" in properties:
            scores = []
            for chunk_id in _context_chunk_ids(prompt):
                scores.append({"chunk_id": chunk_id, "score": 2 if chunk_id == "c3" else 1})
            return {"structured": {"scores": scores}}
        citation = _first_context_chunk_id(prompt)
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "citation_chunk_ids": [citation], "short_rationale": "fake"}}
        if "answer_text" in properties:
            return {"structured": {"answer_text": "Risposta fake", "citation_chunk_ids": [citation], "short_rationale": "fake"}}
        if "score" in properties:
            return {"structured": {"score": 2, "explanation": "Correct fake answer."}}
        raise AssertionError(f"Unexpected schema: {payload_schema}")


class InvalidRerankClient(FakeStructuredClient):
    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if "scores" in payload_schema.get("properties", {}):
            return {"structured": {"scores": [{"chunk_id": "c1", "score": 3}]}}
        return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)


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
            return {"structured": {"answer_label": "A", "citation_chunk_ids": ["missing-citation"], "short_rationale": "fake"}}
        if "answer_text" in properties:
            return {"structured": {"answer_text": "Risposta fake", "citation_chunk_ids": ["missing-citation"], "short_rationale": "fake"}}
        return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)


class SlowRecordingClient(FakeStructuredClient):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_active_calls = 0

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            time.sleep(0.05)
            return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)
        finally:
            with self._lock:
                self._active_calls -= 1


def _context_chunk_ids(prompt: str) -> list[str]:
    return re.findall(r"chunk_id: ([^\s]+)", prompt)


def _first_context_chunk_id(prompt: str) -> str:
    ids = _context_chunk_ids(prompt)
    return ids[0] if ids else "c1"


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    evaluation_dir = tmp_path / "evaluation_clean"
    evaluation_dir.mkdir()
    mcq_records = [
        {
            "qid": "eval-0001",
            "source_position": 1,
            "level": "L1",
            "question_stem": "Graph question?",
            "options": {"A": "Answer A", "B": "Answer B", "C": "C", "D": "D", "E": "E", "F": "F"},
            "correct_label": "A",
            "correct_answer": "Answer A",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 1"],
        }
    ]
    no_hint_records = [
        {
            "qid": "eval-0001",
            "source_position": 1,
            "level": "L1",
            "question": "Graph question?",
            "correct_answer": "Answer A",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 1"],
            "linked_mcq_qid": "eval-0001",
        }
    ]
    write_jsonl(evaluation_dir / "questions_mcq.jsonl", mcq_records)
    write_jsonl(evaluation_dir / "questions_no_hint.jsonl", no_hint_records)
    write_json(evaluation_dir / "evaluation_manifest.json", {"schema_version": "evaluation-dataset-v1"})

    laws_dir = tmp_path / "laws_dataset_clean"
    laws_dir.mkdir()
    write_jsonl(
        laws_dir / "laws.jsonl",
        [
            {"law_id": LAW_1, "law_title": "Legge regionale 1 gennaio 2000, n. 1"},
            {"law_id": LAW_2, "law_title": "Legge regionale 1 gennaio 2000, n. 2"},
            {"law_id": LAW_3, "law_title": "Legge regionale 1 gennaio 2000, n. 3"},
        ],
    )
    write_jsonl(
        laws_dir / "articles.jsonl",
        [
            {"law_id": LAW_1, "article_id": f"{LAW_1}#art:1", "article_label_norm": "1", "article_text": "Answer A."},
            {"law_id": LAW_2, "article_id": f"{LAW_2}#art:1", "article_label_norm": "1", "article_text": "Seed law."},
            {"law_id": LAW_3, "article_id": f"{LAW_3}#art:1", "article_label_norm": "1", "article_text": "Expanded law."},
        ],
    )
    write_jsonl(
        laws_dir / "edges.jsonl",
        [
            {
                "edge_id": "e1",
                "src_law_id": LAW_2,
                "dst_law_id": LAW_3,
                "relation_type": "REFERENCES",
            }
        ],
    )
    write_jsonl(
        laws_dir / "chunks.jsonl",
        [
            _chunk_payload("c1", law_id=LAW_1, text="Answer A legal text."),
            _chunk_payload("c2", law_id=LAW_2, text="Seed legal text."),
            _chunk_payload("c3", law_id=LAW_3, text="Expanded legal text."),
        ],
    )
    write_json(laws_dir / "manifest.json", {"ready_for_indexing": True})

    index_manifest = _write_index_manifest(tmp_path, evaluation_dir=evaluation_dir)
    simple_manifest = _write_simple_manifest(tmp_path, evaluation_dir=evaluation_dir, index_manifest=index_manifest)
    return evaluation_dir, laws_dir, index_manifest, simple_manifest


def _write_index_manifest(tmp_path: Path, *, evaluation_dir: Path, collection_name: str = "advanced_collection") -> Path:
    run_dir = tmp_path / "indexing_runs" / "20260504_000000"
    run_dir.mkdir(parents=True)
    path = run_dir / "index_manifest.json"
    write_json(
        path,
        {
            "schema_version": "indexing-contract-v1",
            "collection_name": collection_name,
            "ready_for_retrieval": True,
            "hybrid_enabled": True,
            "embedding": {"provider": "utopia", "model": "fake-embedding", "configured_model": "fake-embedding", "mode": "auto"},
            "config": {"embedding_provider": "utopia", "embedding_model": "fake-embedding", "hybrid_enabled": True},
        },
    )
    return path


def _write_simple_manifest(tmp_path: Path, *, evaluation_dir: Path, index_manifest: Path) -> Path:
    output_dir = tmp_path / "simple"
    output_dir.mkdir()
    path = output_dir / "simple_rag_manifest.json"
    write_json(
        path,
        {
            "schema_version": "simple-rag-v1",
            "source_hashes": {
                "questions_mcq": sha256_file(evaluation_dir / "questions_mcq.jsonl"),
                "questions_no_hint": sha256_file(evaluation_dir / "questions_no_hint.jsonl"),
                "evaluation_manifest": sha256_file(evaluation_dir / "evaluation_manifest.json"),
                "index_manifest": sha256_file(index_manifest),
            },
        },
    )
    return path


def _chunk_payload(
    chunk_id: str,
    *,
    law_id: str,
    text: str,
    law_status: str = "current",
    article_label_norm: str = "1",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "law_id": law_id,
        "article_id": f"{law_id}#art:{article_label_norm}",
        "article_label_norm": article_label_norm,
        "text": text,
        "law_title": f"Law {chunk_id}",
        "law_status": law_status,
        "article_status": "current",
        "index_views": ["current"],
        "relation_types": [],
    }


def _make_qdrant(*, sparse: bool = True, collection_name: str = "advanced_collection") -> QdrantClient:
    client = QdrantClient(":memory:")
    if sparse:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )
        vectors = [
            {"dense": [1.0, 0.0, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[10], values=[1.0])},
            {"dense": [0.95, 0.05, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[20], values=[1.0])},
            {"dense": [0.0, 1.0, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[30], values=[1.0])},
        ]
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE),
        )
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    client.upsert(
        collection_name=collection_name,
        points=[
            qmodels.PointStruct(id=1, vector=vectors[0], payload=_chunk_payload("c1", law_id=LAW_1, text="Answer A legal text.")),
            qmodels.PointStruct(id=2, vector=vectors[1], payload=_chunk_payload("c2", law_id=LAW_2, text="Seed legal text.")),
            qmodels.PointStruct(id=3, vector=vectors[2], payload=_chunk_payload("c3", law_id=LAW_3, text="Expanded legal text.")),
        ],
        wait=True,
    )
    return client


def _add_qdrant_chunk(
    client: QdrantClient,
    *,
    point_id: int,
    chunk_id: str,
    law_id: str,
    text: str,
    vector: list[float] | None = None,
    article_label_norm: str = "1",
) -> None:
    client.upsert(
        collection_name="advanced_collection",
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector={
                    "dense": vector or [0.0, 1.0, 0.0, 0.0],
                    "sparse": qmodels.SparseVector(indices=[30 + point_id], values=[1.0]),
                },
                payload=_chunk_payload(chunk_id, law_id=law_id, text=text, article_label_norm=article_label_norm),
            )
        ],
        wait=True,
    )


def _config(tmp_path: Path, evaluation_dir: Path, laws_dir: Path, index_manifest: Path, simple_manifest: Path, **overrides: Any) -> AdvancedRagConfig:
    data = {
        "evaluation_dir": str(evaluation_dir),
        "laws_dir": str(laws_dir),
        "index_manifest_path": str(index_manifest),
        "simple_rag_manifest_path": str(simple_manifest),
        "output_root": str(tmp_path / "advanced_runs"),
        "run_name": "test",
        "chat_model": "fake-answer",
        "judge_model": "fake-judge",
        "max_concurrency": 1,
        "top_k": 2,
        "rerank_input_k": 4,
        "rerank_output_k": 2,
    }
    data.update(overrides)
    return AdvancedRagConfig.model_validate(data)


def test_qdrant_connection_uses_nested_manifest_url(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr("legal_rag.advanced_graph_rag.retrieval.QdrantClient", FakeQdrantClient)

    connect_qdrant(
        AdvancedRagConfig(index_dir="local-path"),
        {"qdrant": {"url": "http://127.0.0.1:6333", "path": "server-storage"}, "config": {"index_dir": "config-path"}},
    )

    assert calls == {"url": "http://127.0.0.1:6333"}


def test_qdrant_connection_uses_nested_config_url(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr("legal_rag.advanced_graph_rag.retrieval.QdrantClient", FakeQdrantClient)

    connect_qdrant(AdvancedRagConfig(index_dir="local-path"), {"config": {"qdrant_url": "http://localhost:6333"}})

    assert calls == {"url": "http://localhost:6333"}


def test_qdrant_connection_uses_manifest_path_fallback(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr("legal_rag.advanced_graph_rag.retrieval.QdrantClient", FakeQdrantClient)

    connect_qdrant(AdvancedRagConfig(index_dir="local-path"), {"qdrant": {"path": "manifest-path"}, "config": {"index_dir": "config-path"}})

    assert calls == {"path": "manifest-path"}


def test_qdrant_preflight_fails_fast_when_collection_is_missing(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    with pytest.raises(RuntimeError, match="collection not found"):
        run_advanced_graph_rag(
            _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest),
            client=FakeStructuredClient(),
            qdrant_client=_make_qdrant(collection_name="other_collection"),
            embedder=FakeHybridEmbedder(),
        )

    assert not (tmp_path / "advanced_runs" / "test").exists()


def test_qdrant_preflight_fails_fast_when_collection_is_empty(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name="advanced_collection",
        vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
    )

    with pytest.raises(RuntimeError, match="collection is empty"):
        run_advanced_graph_rag(
            _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest),
            client=FakeStructuredClient(),
            qdrant_client=qdrant,
            embedder=FakeHybridEmbedder(),
        )

    assert not (tmp_path / "advanced_runs" / "test").exists()


def test_run_advanced_graph_rag_exports_contract_files_and_traces(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    manifest = run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    output_dir = tmp_path / "advanced_runs" / "test"
    assert {path.name for path in output_dir.iterdir()} == {
        "advanced_rag_manifest.json",
        "mcq_results.jsonl",
        "no_hint_results.jsonl",
        "advanced_rag_summary.json",
        "advanced_diagnostics.json",
        "quality_report.md",
    }
    assert manifest["schema_version"] == ADVANCED_RAG_SCHEMA_VERSION
    assert manifest["prompt_version"] == ADVANCED_RAG_PROMPT_VERSION
    row = json.loads((output_dir / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["metadata_filters"] == {"law_status": "current"}
    assert row["retrieval_mode"] == "hybrid"
    assert row["graph_expanded_law_ids"] == [LAW_3]
    assert row["graph_expanded_chunk_ids"] == ["c3"]
    assert row["graph_relations_used"] == [{"source_law_id": LAW_2, "target_law_id": LAW_3, "relation_type": "REFERENCES"}]
    assert row["reranked_chunk_ids"][:2] == ["c3", "c1"]
    assert row["rerank_scores"] == [2, 1]
    assert row["context_included_count"] == 2
    assert row["reference_law_hit"] is True
    assert row["failure_category"] is None
    no_hint_row = json.loads((output_dir / "no_hint_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert no_hint_row["context_sufficient"] == "yes"
    assert manifest["diagnostics"]["context_sufficient_counts"] == {"yes": 1}


def test_embed_sparse_query_accepts_tuple_sparse_vectors() -> None:
    sparse = embed_sparse_query(TupleSparseEmbedder(), "question")

    assert sparse.indices == [10, 20]
    assert sparse.values == [1.0, 0.5]


def test_run_advanced_graph_rag_reports_progress(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    events: list[dict[str, Any]] = []

    run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest, run_name="progress"),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
        progress_callback=events.append,
    )

    assert events[0] == {"event": "setup_finished", "mcq": 1, "no_hint": 1, "total": 2}
    row_events = [event for event in events if event["event"] == "row_finished"]
    assert len(row_events) == 2
    assert {event["run"] for event in row_events} == {"mcq", "no_hint"}
    assert all(event["completed"] == 1 and event["total"] == 1 for event in row_events)
    assert {event["event"] for event in events if event["event"] == "run_finished"} == {"run_finished"}


def test_feature_flags_disable_observable_effects(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="flags",
            metadata_filters_enabled=False,
            hybrid_enabled=False,
            graph_expansion_enabled=False,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "flags" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["metadata_filters"] == {}
    assert row["retrieval_mode"] == "dense"
    assert row["graph_expanded_law_ids"] == []
    assert row["graph_expanded_chunk_ids"] == []
    assert row["graph_relations_used"] == []
    assert row["rerank_scores"] == []
    assert row["reranked_chunk_ids"] == row["retrieved_chunk_ids"]


def test_parallel_dataset_runs_overlap_model_calls(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    client = SlowRecordingClient()

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="parallel_datasets",
            max_concurrency=1,
            parallel_datasets_enabled=True,
        ),
        client=client,
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    assert client.max_active_calls >= 2


def test_hybrid_enabled_requires_sparse_collection_and_embedder(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    with pytest.raises(RuntimeError, match="sparse"):
        run_advanced_graph_rag(
            _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest),
            client=FakeStructuredClient(),
            qdrant_client=_make_qdrant(sparse=False),
            embedder=DenseOnlyEmbedder(),
        )


def test_invalid_rerank_score_is_recorded_as_judge_style_error(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest, run_name="invalid_rerank"),
        client=InvalidRerankClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "invalid_rerank" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["score"] is None
    assert "rerank_error" in row["error"]
    assert row["failure_category"] == "judge_error"


def test_graph_expansion_deduplicates_edges_and_caps_chunks_per_target_law(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    write_jsonl(
        laws_dir / "edges.jsonl",
        [
            {"edge_id": "e1", "src_law_id": LAW_2, "dst_law_id": LAW_3, "relation_type": "REFERENCES"},
            {"edge_id": "e1-duplicate", "src_law_id": LAW_2, "dst_law_id": LAW_3, "relation_type": "REFERENCES"},
            {"edge_id": "e2", "src_law_id": LAW_2, "dst_law_id": LAW_3, "relation_type": "AMENDS"},
        ],
    )
    write_jsonl(
        laws_dir / "chunks.jsonl",
        [
            _chunk_payload("c1", law_id=LAW_1, text="Answer A legal text."),
            _chunk_payload("c2", law_id=LAW_2, text="Seed legal text."),
            _chunk_payload("c3", law_id=LAW_3, text="Expanded legal text."),
            _chunk_payload("c4", law_id=LAW_3, text="Second expanded legal text."),
        ],
    )
    qdrant = _make_qdrant()
    _add_qdrant_chunk(qdrant, point_id=4, chunk_id="c4", law_id=LAW_3, text="Second expanded legal text.")

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="graph_cap",
            max_chunks_per_expanded_law=1,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=qdrant,
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "graph_cap" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(row["graph_expanded_chunk_ids"]) == 1
    assert set(row["graph_expanded_chunk_ids"]) <= {"c3", "c4"}
    assert row["graph_relations_used"] == [{"source_law_id": LAW_2, "target_law_id": LAW_3, "relation_type": "REFERENCES"}]


def test_graph_expansion_ignores_qdrant_chunks_missing_from_clean_chunks_jsonl(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    write_jsonl(
        laws_dir / "chunks.jsonl",
        [
            _chunk_payload("c1", law_id=LAW_1, text="Answer A legal text."),
            _chunk_payload("c2", law_id=LAW_2, text="Seed legal text."),
            _chunk_payload("c3", law_id=LAW_3, text="Expanded legal text."),
        ],
    )
    qdrant = _make_qdrant()
    _add_qdrant_chunk(qdrant, point_id=4, chunk_id="orphan-c4", law_id=LAW_3, text="Orphan expanded text.")

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="graph_allowed_chunks",
            max_chunks_per_expanded_law=3,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=qdrant,
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "graph_allowed_chunks" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["graph_expanded_chunk_ids"] == ["c3"]
    assert "orphan-c4" not in row["retrieved_chunk_ids"]


def test_graph_expansion_defaults_exclude_obsolete_relation_types(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    write_jsonl(
        laws_dir / "edges.jsonl",
        [
            {"edge_id": "e1", "src_law_id": LAW_2, "dst_law_id": LAW_3, "relation_type": "ABROGATED_BY"},
        ],
    )

    run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest, rerank_enabled=False),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "test" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["graph_expanded_chunk_ids"] == []
    assert row["graph_relations_used"] == []


def test_graph_expansion_respects_min_edge_confidence(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    write_jsonl(
        laws_dir / "edges.jsonl",
        [
            {"edge_id": "e1", "src_law_id": LAW_2, "dst_law_id": LAW_3, "relation_type": "AMENDS", "confidence": 0.69},
        ],
    )

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            min_edge_confidence=0.7,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "test" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["graph_expanded_chunk_ids"] == []


def test_graph_expansion_respects_global_expanded_chunk_cap(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    law_4 = "vda:lr:2000-01-01:4"
    write_jsonl(
        laws_dir / "edges.jsonl",
        [
            {"edge_id": "e1", "src_law_id": LAW_2, "dst_law_id": LAW_3, "relation_type": "REFERENCES"},
            {"edge_id": "e2", "src_law_id": LAW_2, "dst_law_id": law_4, "relation_type": "AMENDS"},
        ],
    )
    write_jsonl(
        laws_dir / "chunks.jsonl",
        [
            _chunk_payload("c1", law_id=LAW_1, text="Answer A legal text."),
            _chunk_payload("c2", law_id=LAW_2, text="Seed legal text."),
            _chunk_payload("c3", law_id=LAW_3, text="Expanded legal text."),
            _chunk_payload("c4", law_id=law_4, text="Extra expanded legal text."),
        ],
    )
    qdrant = _make_qdrant()
    _add_qdrant_chunk(qdrant, point_id=4, chunk_id="c4", law_id=law_4, text="Extra expanded legal text.")

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            max_chunks_per_expanded_law=2,
            max_expanded_chunks_total=1,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=qdrant,
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "test" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["graph_expanded_chunk_ids"] == ["c3"]


def test_graph_expansion_uses_dst_article_label_when_available(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    write_jsonl(
        laws_dir / "edges.jsonl",
        [
            {
                "edge_id": "e1",
                "src_law_id": LAW_2,
                "dst_law_id": LAW_3,
                "dst_article_label_norm": "2",
                "relation_type": "REFERENCES",
            },
        ],
    )
    write_jsonl(
        laws_dir / "chunks.jsonl",
        [
            _chunk_payload("c1", law_id=LAW_1, text="Answer A legal text."),
            _chunk_payload("c2", law_id=LAW_2, text="Seed legal text."),
            _chunk_payload("c3", law_id=LAW_3, text="Article one text.", article_label_norm="1"),
            _chunk_payload("c4", law_id=LAW_3, text="Article two text.", article_label_norm="2"),
        ],
    )
    qdrant = _make_qdrant()
    _add_qdrant_chunk(qdrant, point_id=4, chunk_id="c4", law_id=LAW_3, text="Article two text.", article_label_norm="2")

    run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest, rerank_enabled=False),
        client=FakeStructuredClient(),
        qdrant_client=qdrant,
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "test" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["graph_expanded_chunk_ids"] == ["c4"]


def test_graph_expansion_can_use_clean_chunks_without_qdrant_client() -> None:
    graph = GraphIndex(
        edges=[
            {
                "edge_id": "e1",
                "src_law_id": LAW_2,
                "dst_law_id": LAW_3,
                "dst_article_label_norm": "2",
                "relation_type": "REFERENCES",
            }
        ],
        chunks=[
            _chunk_payload("c3", law_id=LAW_3, text="Article one text.", article_label_norm="1"),
            _chunk_payload("c4", law_id=LAW_3, text="Article two text.", article_label_norm="2"),
        ],
    )
    seed = RetrievedChunkRecord(
        chunk_id="seed",
        score=1.0,
        text="Seed",
        payload=_chunk_payload("seed", law_id=LAW_2, text="Seed"),
    )

    expanded, relations = expand_with_graph(
        None,
        collection_name="unused",
        graph=graph,
        seeds=[seed],
        relation_types=["REFERENCES"],
        static_filters={"law_status": "current"},
        max_chunks_per_law=2,
        max_chunks_total=2,
        min_edge_confidence=0.0,
    )

    assert [chunk.chunk_id for chunk in expanded] == ["c4"]
    assert [relation.to_json_record() for relation in relations] == [
        {"source_law_id": LAW_2, "target_law_id": LAW_3, "relation_type": "REFERENCES"}
    ]


def test_graph_expansion_semantically_ranks_chunks_within_law(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    write_jsonl(
        laws_dir / "chunks.jsonl",
        [
            _chunk_payload("c1", law_id=LAW_1, text="Answer A legal text."),
            _chunk_payload("c2", law_id=LAW_2, text="Seed legal text."),
            _chunk_payload("c3", law_id=LAW_3, text="Less similar target text."),
            _chunk_payload("c4", law_id=LAW_3, text="More similar target text."),
        ],
    )
    qdrant = _make_qdrant()
    _add_qdrant_chunk(qdrant, point_id=4, chunk_id="c4", law_id=LAW_3, text="More similar target text.", vector=[0.9, 0.1, 0.0, 0.0])

    run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            max_chunks_per_expanded_law=1,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=qdrant,
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "test" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["graph_expanded_chunk_ids"] == ["c4"]


def test_no_hint_answer_output_accepts_context_sufficient() -> None:
    output = AdvancedNoHintAnswerOutput.model_validate(
        {
            "answer_text": "Risposta parziale",
            "context_sufficient": "partial",
            "citation_chunk_ids": ["c1"],
        }
    )

    assert output.to_json_record()["context_sufficient"] == "partial"


def test_invalid_citation_marks_row_as_generation_failure_even_when_answer_is_correct(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest, run_name="invalid_citation"),
        client=InvalidCitationClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    row = json.loads((tmp_path / "advanced_runs" / "invalid_citation" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["score"] == 1
    assert "citation_error" in row["error"]
    assert row["failure_category"] == "generation_error"


class QueryRewritingClient(FakeStructuredClient):
    """FakeStructuredClient that also serves query rewriting payloads."""

    def __init__(self) -> None:
        self.call_count_by_schema: dict[str, int] = {}

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        props = payload_schema.get("properties", {})
        if "queries" in props:
            self.call_count_by_schema["multi_query"] = self.call_count_by_schema.get("multi_query", 0) + 1
            return {"structured": {"queries": ["Domanda variante uno", "Domanda variante due", "Domanda variante tre"]}}
        if "rewritten_query" in props:
            self.call_count_by_schema["rewrite"] = self.call_count_by_schema.get("rewrite", 0) + 1
            return {"structured": {"rewritten_query": "Domanda riformulata"}}
        if "hypothetical_answer" in props:
            self.call_count_by_schema["hyde"] = self.call_count_by_schema.get("hyde", 0) + 1
            return {"structured": {"hypothetical_answer": "Risposta ipotetica."}}
        return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)


def test_skip_baseline_validation_allows_mismatched_index_hash(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    # Stamp the simple manifest with a bogus index_manifest hash so the default guard would fail.
    payload = json.loads(simple_manifest.read_text(encoding="utf-8"))
    payload["source_hashes"]["index_manifest"] = "deadbeef" * 8
    simple_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not comparable"):
        run_advanced_graph_rag(
            _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest, run_name="baseline_check_on"),
            client=FakeStructuredClient(), qdrant_client=_make_qdrant(), embedder=FakeHybridEmbedder(),
        )

    # With skip_baseline_validation=True the pipeline runs despite the hash mismatch.
    manifest = run_advanced_graph_rag(
        _config(tmp_path, evaluation_dir, laws_dir, index_manifest, simple_manifest,
                run_name="baseline_check_off", skip_baseline_validation=True),
        client=FakeStructuredClient(), qdrant_client=_make_qdrant(), embedder=FakeHybridEmbedder(),
    )
    assert manifest["summary"]["mcq"]["processed"] == 1


def test_query_rewriting_disabled_is_no_op(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    client = QueryRewritingClient()

    manifest = run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="qr_disabled",
            query_rewriting_cache_dir=str(tmp_path / "cache"),
        ),
        client=client,
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    block = manifest["query_rewriting"]
    assert block["enabled"] is False
    assert block["strategy"] == "none"
    assert block["cache_path"] is None
    assert block["cache_hits"] == 0
    assert block["cache_misses"] == 0
    assert block["failures"] == 0
    # No multi_query / rewrite / hyde calls should have happened.
    assert client.call_count_by_schema == {}


def test_query_rewriting_multi_query_produces_block_and_uses_cache(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)
    client = QueryRewritingClient()

    manifest = run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="qr_multi",
            query_rewriting_enabled=True,
            query_rewriting_strategy="multi_query",
            query_rewriting_n=3,
            query_rewriting_cache_dir=str(tmp_path / "cache"),
        ),
        client=client,
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    block = manifest["query_rewriting"]
    assert block["enabled"] is True
    assert block["strategy"] == "multi_query"
    assert block["n"] == 3
    assert block["failures"] == 0
    # MCQ + no_hint share the same question stem so cache hits once, misses once.
    assert block["cache_misses"] == 1
    assert block["cache_hits"] == 1
    assert client.call_count_by_schema.get("multi_query") == 1
    cache_path = Path(block["cache_path"])
    assert cache_path.exists() and cache_path.stat().st_size > 0


def test_query_rewriting_rewrite_strategy_falls_back_on_failure(tmp_path: Path) -> None:
    evaluation_dir, laws_dir, index_manifest, simple_manifest = _make_inputs(tmp_path)

    class FailingRewriteClient(FakeStructuredClient):
        def structured_chat(self, *, prompt, model, payload_schema, timeout_seconds):
            if "rewritten_query" in payload_schema.get("properties", {}):
                raise RuntimeError("simulated_rewrite_failure")
            return super().structured_chat(prompt=prompt, model=model, payload_schema=payload_schema, timeout_seconds=timeout_seconds)

    manifest = run_advanced_graph_rag(
        _config(
            tmp_path,
            evaluation_dir,
            laws_dir,
            index_manifest,
            simple_manifest,
            run_name="qr_failing",
            query_rewriting_enabled=True,
            query_rewriting_strategy="rewrite",
            query_rewriting_cache_dir=str(tmp_path / "cache"),
        ),
        client=FailingRewriteClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    block = manifest["query_rewriting"]
    # Both rows trigger a fresh rewrite attempt (no cache available) and both fail.
    assert block["failures"] == 2
    assert block["cache_hits"] == 0
    assert block["cache_misses"] == 0
    # The pipeline still produces results because we fall back to the original question.
    row = json.loads((tmp_path / "advanced_runs" / "qr_failing" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["retrieved_chunk_ids"]


def _mk_chunk(chunk_id: str) -> RetrievedChunkRecord:
    return RetrievedChunkRecord(chunk_id=chunk_id, score=1.0, text=chunk_id, payload={"chunk_id": chunk_id})


def test_fuse_rrf_promotes_chunks_present_in_multiple_queries() -> None:
    """RRF cross-query fusion must rank a chunk seen in many queries above one seen in a single query."""
    from legal_rag.advanced_graph_rag.runner import _fuse_rrf

    # Common chunk appears at rank 3 in every query; rare chunk is rank 1 only in the first.
    common = "chunk_common"
    rare = "chunk_rare_first_only"
    batches = [
        [_mk_chunk(rare), _mk_chunk("filler_a"), _mk_chunk(common)],
        [_mk_chunk("filler_b"), _mk_chunk("filler_c"), _mk_chunk(common)],
        [_mk_chunk("filler_d"), _mk_chunk("filler_e"), _mk_chunk(common)],
    ]
    fused = _fuse_rrf(batches, rrf_k=60, limit=5)
    ids = [chunk.chunk_id for chunk in fused]
    assert ids[0] == common, f"common chunk should win RRF; got {ids}"
    assert rare in ids, "rare chunk must still survive the fusion"
    assert ids.index(common) < ids.index(rare)


def test_fuse_rrf_falls_back_to_first_seen_when_single_batch() -> None:
    """With one query the result must preserve the upstream ranking (no reshuffling)."""
    from legal_rag.advanced_graph_rag.runner import _fuse_rrf

    batches = [[_mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c")]]
    fused = _fuse_rrf(batches, rrf_k=60, limit=2)
    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]


def test_apply_query_rewriting_always_includes_original_question(tmp_path: Path) -> None:
    """When multi-query rewrites succeed, the original question must remain the first variant."""
    from legal_rag.advanced_graph_rag.runner import QueryRewriteStats, apply_query_rewriting
    from legal_rag.retrieval_evaluation.query_rewriting import QueryRewriteCache

    config = AdvancedRagConfig(
        run_name="rewrite_includes_original",
        query_rewriting_enabled=True,
        query_rewriting_strategy="multi_query",
        query_rewriting_n=3,
    )
    original = "Che cos'è il GAP?"
    rewrites = [
        "Cos'è il gioco d'azzardo patologico?",
        "GAP definizione legge regionale",
        "ludopatia significato",
    ]
    cache = QueryRewriteCache(tmp_path / "rewrite_cache.jsonl")
    cache.set(
        QueryRewriteCache.make_key(
            question=original,
            strategy="multi_query",
            model=config.resolved_query_rewriting_model,
            prompt_version=config.query_rewriting_prompt_version,
        ),
        rewrites,
    )

    queries = apply_query_rewriting(
        question=original,
        config=config,
        llm_client=None,  # type: ignore[arg-type]
        cache=cache,
        stats=QueryRewriteStats(),
    )
    assert queries[0] == original
    assert all(rewrite in queries for rewrite in rewrites)
    assert len(queries) == len(rewrites) + 1


def test_advanced_config_max_context_chunks_falls_back_to_rerank_output_k() -> None:
    """effective_max_context_chunks returns max_context_chunks when set, else rerank_output_k."""
    default_cfg = AdvancedRagConfig(run_name="ctx_default", rerank_output_k=10)
    assert default_cfg.effective_max_context_chunks == 10
    explicit_cfg = AdvancedRagConfig(run_name="ctx_explicit", rerank_output_k=10, max_context_chunks=3)
    assert explicit_cfg.effective_max_context_chunks == 3

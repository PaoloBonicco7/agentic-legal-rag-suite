from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.advanced_graph_rag import (
    ADVANCED_RAG_PROMPT_VERSION,
    ADVANCED_RAG_SCHEMA_VERSION,
    AdvancedRagConfig,
    run_advanced_graph_rag,
)
from legal_rag.oracle_context_evaluation.io import sha256_file, write_json, write_jsonl
from legal_rag.advanced_graph_rag.retrieval import connect_qdrant


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


def _chunk_payload(chunk_id: str, *, law_id: str, text: str, law_status: str = "current") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "law_id": law_id,
        "article_id": f"{law_id}#art:1",
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


def _add_qdrant_chunk(client: QdrantClient, *, point_id: int, chunk_id: str, law_id: str, text: str) -> None:
    client.upsert(
        collection_name="advanced_collection",
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector={"dense": [0.0, 1.0, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[30 + point_id], values=[1.0])},
                payload=_chunk_payload(chunk_id, law_id=law_id, text=text),
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
    assert row["graph_expanded_chunk_ids"] == ["c3"]
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

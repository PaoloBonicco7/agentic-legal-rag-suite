from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.advanced_graph_rag import InteractiveRagConfig, answer_interactive_question
from legal_rag.oracle_context_evaluation.io import write_json, write_jsonl

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
            return {
                "structured": {
                    "scores": [
                        {"chunk_id": chunk_id, "score": 2 if chunk_id == "c3" else 1}
                        for chunk_id in _context_chunk_ids(prompt)
                    ]
                }
            }
        if "answer_text" in properties:
            return {
                "structured": {
                    "answer_text": "Risposta fake",
                    "citation_chunk_ids": [_first_context_chunk_id(prompt)],
                    "short_rationale": "fake",
                }
            }
        raise AssertionError(f"Unexpected schema: {payload_schema}")


def _context_chunk_ids(prompt: str) -> list[str]:
    return re.findall(r"chunk_id: ([^\s]+)", prompt)


def _first_context_chunk_id(prompt: str) -> str:
    ids = _context_chunk_ids(prompt)
    return ids[0] if ids else "c1"


def _make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    laws_dir = tmp_path / "laws_dataset_clean"
    laws_dir.mkdir()
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
    index_manifest = tmp_path / "index_manifest.json"
    write_json(
        index_manifest,
        {
            "schema_version": "indexing-contract-v1",
            "collection_name": "advanced_collection",
            "ready_for_retrieval": True,
            "hybrid_enabled": True,
            "embedding": {
                "provider": "utopia",
                "model": "fake-embedding",
                "configured_model": "fake-embedding",
                "mode": "auto",
            },
            "config": {"embedding_provider": "utopia", "embedding_model": "fake-embedding", "hybrid_enabled": True},
        },
    )
    return laws_dir, index_manifest


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


def _config(tmp_path: Path, laws_dir: Path, index_manifest: Path, **overrides: Any) -> InteractiveRagConfig:
    data = {
        "laws_dir": str(laws_dir),
        "index_manifest_path": str(index_manifest),
        "output_root": str(tmp_path / "advanced_runs"),
        "run_name": "interactive",
        "chat_model": "fake-answer",
        "judge_model": "fake-judge",
        "top_k": 2,
        "rerank_input_k": 4,
        "rerank_output_k": 2,
    }
    data.update(overrides)
    return InteractiveRagConfig.model_validate(data)


def test_interactive_question_returns_full_trace_without_writing_run_artifacts(tmp_path: Path) -> None:
    laws_dir, index_manifest = _make_inputs(tmp_path)
    output_root = tmp_path / "advanced_runs"

    result = answer_interactive_question(
        "Graph question?",
        _config(tmp_path, laws_dir, index_manifest, hybrid_enabled=True),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    assert result.error is None
    assert result.answer == "Risposta fake"
    assert [chunk.chunk_id for chunk in result.retrieved] == ["c1", "c2"]
    assert [chunk.chunk_id for chunk in result.expanded] == ["c3"]
    assert result.graph_relations_used[0].to_json_record() == {
        "source_law_id": LAW_2,
        "target_law_id": LAW_3,
        "relation_type": "REFERENCES",
    }
    assert [chunk.chunk_id for chunk in result.reranked] == ["c3", "c1"]
    assert result.rerank_scores == [2, 1]
    assert [chunk.chunk_id for chunk in result.context_chunks] == ["c3", "c1"]
    assert result.citations[0].chunk_id == "c3"
    assert not output_root.exists()


def test_interactive_flags_disable_observable_steps(tmp_path: Path) -> None:
    laws_dir, index_manifest = _make_inputs(tmp_path)

    result = answer_interactive_question(
        "Graph question?",
        _config(
            tmp_path,
            laws_dir,
            index_manifest,
            metadata_filters_enabled=False,
            hybrid_enabled=False,
            graph_expansion_enabled=False,
            rerank_enabled=False,
        ),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(),
        embedder=FakeHybridEmbedder(),
    )

    assert result.error is None
    assert result.metadata_filters == {}
    assert result.retrieval_mode == "dense"
    assert result.expanded == []
    assert result.graph_relations_used == []
    assert result.rerank_scores == []
    assert [chunk.chunk_id for chunk in result.reranked] == [chunk.chunk_id for chunk in result.retrieved]


def test_interactive_hybrid_reports_clear_error_when_sparse_is_unavailable(tmp_path: Path) -> None:
    laws_dir, index_manifest = _make_inputs(tmp_path)

    result = answer_interactive_question(
        "Graph question?",
        _config(tmp_path, laws_dir, index_manifest, hybrid_enabled=True),
        client=FakeStructuredClient(),
        qdrant_client=_make_qdrant(sparse=False),
        embedder=DenseOnlyEmbedder(),
    )

    assert result.error is not None
    assert "hybrid_unavailable" in result.error
    assert "sparse" in result.error
    assert result.retrieved == []

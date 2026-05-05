from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from legal_rag.indexing import (
    IndexingConfig,
    content_hash_for_text,
    point_id_from_chunk_id,
    run_indexing_pipeline,
    validate_clean_dataset,
)
from legal_rag.indexing.cli import main as indexing_main


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            base = float(len(text) or 1)
            vectors.append([base, base / 2.0, 1.0, 0.5])
        return vectors

    def embed_sparse_texts(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [([1, 2], [1.0, float(len(text) % 7 + 1)]) for text in texts]


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _chunk(chunk_id: str, *, text: str, law_id: str = "vda:lr:2000-01-01:1") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "passage_id": f"{law_id}#art:1#p:c1",
        "article_id": f"{law_id}#art:1",
        "law_id": law_id,
        "chunk_seq": 0,
        "text": text,
        "text_for_embedding": f"[LR 2000-01-01 n.1] Test law | Art. 1 | c1 |\n\n{text}",
        "law_date": "2000-01-01",
        "law_number": 1,
        "law_title": "Legge regionale 1 gennaio 2000, n. 1 - Test",
        "law_status": "current",
        "article_status": "current",
        "article_label_norm": "1",
        "passage_label": "c1",
        "structure_path": "",
        "source_file": "0001_LR-1-gennaio-2000-n1.html",
        "index_views": ["current", "historical"],
        "related_law_ids": [],
        "inbound_law_ids": [],
        "outbound_law_ids": [],
        "relation_types": ["REFERENCES"],
    }


def _write_dataset(root: Path, chunks: list[dict[str, object]], *, ready: bool = True) -> None:
    root.mkdir()
    _write_jsonl(root / "laws.jsonl", [{"law_id": "vda:lr:2000-01-01:1"}])
    _write_jsonl(root / "articles.jsonl", [{"article_id": "vda:lr:2000-01-01:1#art:1"}])
    _write_jsonl(root / "edges.jsonl", [{"edge_id": "e1"}])
    _write_jsonl(root / "chunks.jsonl", chunks)
    _write_json(
        root / "manifest.json",
        {
            "ready_for_indexing": ready,
            "source_hash": "source-hash",
            "counts": {"laws": 1, "articles": 1, "edges": 1, "chunks": len(chunks)},
            "output_hashes": {"chunks": "chunks-hash"},
        },
    )


def test_validate_clean_dataset_rejects_missing_ready_flag(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    _write_dataset(dataset, [_chunk("c1", text="Testo.")], ready=False)

    result = validate_clean_dataset(dataset)

    assert result.ok is False
    assert any("ready_for_indexing" in error for error in result.errors)


def test_validate_clean_dataset_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    _write_dataset(dataset, [_chunk("c1", text="Uno."), _chunk("c1", text="Due.")])

    result = validate_clean_dataset(dataset)

    assert result.ok is False
    assert result.duplicate_chunk_ids == ("c1",)


def test_hashing_helpers_are_stable() -> None:
    assert content_hash_for_text(" testo \n") == content_hash_for_text("testo")
    assert point_id_from_chunk_id("chunk-1") == point_id_from_chunk_id("chunk-1")
    assert point_id_from_chunk_id("chunk-1") != point_id_from_chunk_id("chunk-2")


def test_run_indexing_pipeline_creates_qdrant_contract_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    chunks = [_chunk("c1", text="Contributi regionali."), _chunk("c2", text="Formazione professionale.")]
    _write_dataset(dataset, chunks)
    client = QdrantClient(":memory:")

    manifest = run_indexing_pipeline(
        IndexingConfig(
            clean_dataset_dir=str(dataset),
            runs_dir=str(tmp_path / "runs"),
            collection_name="test_collection",
            run_id="run1",
            embedding_backend="local",
            embedding_model="fake-embedding",
            diagnostic_queries=["contributi"],
        ),
        embedder=FakeEmbedder(),
        client=client,
    )

    assert manifest["ready_for_retrieval"] is True
    assert manifest["indexed_count"] == 2
    assert manifest["collection_points_count"] == 2
    assert manifest["skipped_count"] == 0
    assert manifest["quality_gates"]["filter_validation_queryable"] is True
    run_dir = tmp_path / "runs" / "run1"
    assert (run_dir / "index_manifest.json").exists()
    assert (run_dir / "payload_profile.json").exists()
    assert (run_dir / "index_quality_report.md").exists()
    assert (run_dir / "sample_retrieval_report.json").exists()
    stored = json.loads((run_dir / "index_manifest.json").read_text(encoding="utf-8"))
    assert stored["source_hash"] == "source-hash"
    assert "law_id" in stored["payload_indexes"]


def test_run_indexing_pipeline_reuse_skips_unchanged_points(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    chunks = [_chunk("c1", text="Contributi regionali.")]
    _write_dataset(dataset, chunks)
    client = QdrantClient(":memory:")
    base_config = {
        "clean_dataset_dir": str(dataset),
        "runs_dir": str(tmp_path / "runs"),
        "collection_name": "reuse_collection",
        "embedding_backend": "local",
        "embedding_model": "fake-embedding",
        "diagnostic_queries": ["contributi"],
    }

    run_indexing_pipeline(IndexingConfig(**base_config, run_id="run1"), embedder=FakeEmbedder(), client=client)
    manifest = run_indexing_pipeline(
        IndexingConfig(**base_config, run_id="run2"),
        embedder=FakeEmbedder(),
        client=client,
    )

    assert manifest["indexed_count"] == 1
    assert manifest["skipped_count"] == 1
    assert manifest["upserted_count"] == 0


def test_indexing_cli_smoke_with_injected_pipeline(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run(config: IndexingConfig) -> dict[str, object]:
        assert config.sample_size == 1
        assert config.chunk_selection_mode == "sample"
        return {
            "ready_for_retrieval": True,
            "collection_name": "cli_collection",
            "indexed_count": 1,
            "run_id": "cli",
        }

    monkeypatch.setattr("legal_rag.indexing.cli.run_indexing_pipeline", fake_run)

    assert indexing_main(["--chunk-selection-mode", "sample", "--sample-size", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ready_for_retrieval"] is True
    assert out["collection_name"] == "cli_collection"

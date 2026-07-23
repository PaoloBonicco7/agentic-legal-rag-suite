from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from legal_rag.indexing import (
    IndexingConfig,
    REQUIRED_PAYLOAD_FIELDS,
    content_hash_for_text,
    point_id_from_chunk_id,
    run_indexing_pipeline,
    validate_clean_dataset,
)
from legal_rag.indexing.cli import main as indexing_main
from legal_rag.indexing.qdrant_store import (
    PreparedPoint,
    ensure_collection,
    restore_collection_indexing_threshold,
    upload_point_batch,
)
from legal_rag.indexing.io import sha256_file
from legal_rag.laws_preprocessing.inventory import build_corpus_registry, compute_source_hash


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
        "passage_status": "current",
        "content_availability": "substantive",
        "status_event_ids": [],
        "status_rule_ids": ["default-current"],
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
    source_dir = root.parent / "laws_html"
    source_dir.mkdir(exist_ok=True)
    (source_dir / "0001_LR-1-gennaio-2000-n1.html").write_text(
        "<html><body>Test law</body></html>\n",
        encoding="utf-8",
    )
    registry, _ = build_corpus_registry(source_dir)
    outputs = {
        "laws": "laws.jsonl",
        "articles": "articles.jsonl",
        "edges": "edges.jsonl",
        "chunks": "chunks.jsonl",
    }
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "laws-preprocessing-v2",
            "status_rules_version": "legal-status-rules-v1",
            "ready_for_indexing": ready,
            "source_dir": str(source_dir),
            "source_hash": compute_source_hash(list(registry.by_law_id.values())),
            "counts": {"laws": 1, "articles": 1, "edges": 1, "chunks": len(chunks)},
            "outputs": outputs,
            "output_hashes": {
                name: sha256_file(root / filename)
                for name, filename in outputs.items()
            },
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


def test_validate_clean_dataset_rejects_stale_declared_hash(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    _write_dataset(dataset, [_chunk("c1", text="Uno.")])
    (dataset / "chunks.jsonl").write_text("", encoding="utf-8")

    result = validate_clean_dataset(dataset)

    assert result.ok is False
    assert any("chunks.jsonl hash mismatch" in error for error in result.errors)


def test_hashing_helpers_are_stable() -> None:
    assert content_hash_for_text(" testo \n") == content_hash_for_text("testo")
    assert point_id_from_chunk_id("chunk-1") == point_id_from_chunk_id("chunk-1")
    assert point_id_from_chunk_id("chunk-1") != point_id_from_chunk_id("chunk-2")


def test_run_indexing_pipeline_creates_qdrant_contract_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    chunks = [_chunk("c1", text="Contributi regionali."), _chunk("c2", text="Formazione professionale.")]
    _write_dataset(dataset, chunks)
    client = QdrantClient(":memory:")
    progress_events: list[dict[str, object]] = []

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
        progress_callback=progress_events.append,
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
    assert stored["schema_version"] == "indexing-contract-v2"
    assert stored["preprocessing_schema_version"] == "laws-preprocessing-v2"
    assert stored["status_rules_version"] == "legal-status-rules-v1"
    assert stored["dataset_manifest_hash"] == sha256_file(dataset / "manifest.json")
    assert stored["chunks_hash"] == sha256_file(dataset / "chunks.jsonl")
    assert stored["qdrant"]["mode"] == "local"
    assert stored["inserted_count"] == 2
    assert stored["payload_updated_count"] == 0
    assert all(stored["identity_validation"].values())
    assert all(stored["distribution_reconciliation"].values())
    assert "law_id" in stored["payload_indexes"]
    assert "passage_status" in stored["payload_indexes"]
    assert "content_availability" in stored["payload_indexes"]
    record = client.retrieve(
        collection_name="test_collection",
        ids=[point_id_from_chunk_id("c1")],
        with_payload=True,
        with_vectors=False,
    )[0]
    assert REQUIRED_PAYLOAD_FIELDS <= set(record.payload or {})
    assert [event["event"] for event in progress_events] == [
        "dataset_ready",
        "embedder_ready",
        "embedding_probe_finished",
        "collection_ready",
        "sync_started",
        "batch_finished",
        "run_finished",
    ]


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


def test_run_indexing_pipeline_updates_metadata_without_reembedding(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    chunks = [_chunk("c1", text="Contributi regionali.")]
    _write_dataset(dataset, chunks)
    client = QdrantClient(":memory:")
    base_config = {
        "clean_dataset_dir": str(dataset),
        "runs_dir": str(tmp_path / "runs"),
        "collection_name": "metadata_collection",
        "embedding_backend": "local",
        "embedding_model": "fake-embedding",
        "diagnostic_queries": ["contributi"],
    }
    run_indexing_pipeline(
        IndexingConfig(**base_config, run_id="run1"),
        embedder=FakeEmbedder(),
        client=client,
    )

    chunks[0]["article_status"] = "partial"
    chunks[0]["index_views"] = ["current", "historical", "not_explicitly_past"]
    _write_jsonl(dataset / "chunks.jsonl", chunks)
    manifest_path = dataset / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest["output_hashes"]["chunks"] = sha256_file(dataset / "chunks.jsonl")
    _write_json(manifest_path, source_manifest)

    manifest = run_indexing_pipeline(
        IndexingConfig(**base_config, run_id="run2"),
        embedder=FakeEmbedder(),
        client=client,
    )

    assert manifest["embedded_count"] == 0
    assert manifest["inserted_count"] == 0
    assert manifest["vector_updated_count"] == 0
    assert manifest["payload_updated_count"] == 1
    assert manifest["skipped_count"] == 0
    point_id = point_id_from_chunk_id("c1")
    record = client.retrieve(
        collection_name="metadata_collection",
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )[0]
    assert record.payload["article_status"] == "partial"
    assert record.payload["dataset_chunks_hash"] == sha256_file(dataset / "chunks.jsonl")


def test_run_indexing_pipeline_rejects_changed_source_corpus(tmp_path: Path) -> None:
    dataset = tmp_path / "laws_dataset_clean"
    _write_dataset(dataset, [_chunk("c1", text="Contributi regionali.")])
    source_file = tmp_path / "laws_html" / "0001_LR-1-gennaio-2000-n1.html"
    source_file.write_text("<html><body>Changed source</body></html>\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Source corpus hash mismatch"):
        run_indexing_pipeline(
            IndexingConfig(
                clean_dataset_dir=str(dataset),
                runs_dir=str(tmp_path / "runs"),
                collection_name="changed_source",
                run_id="run1",
                embedding_backend="local",
                embedding_model="fake-embedding",
            ),
            embedder=FakeEmbedder(),
            client=QdrantClient(":memory:"),
        )


def test_ensure_collection_applies_qdrant_server_tuning() -> None:
    class FakeQdrantClient:
        def __init__(self) -> None:
            self.created: dict[str, object] = {}
            self.updated: dict[str, object] = {}

        def collection_exists(self, *, collection_name: str) -> bool:
            return False

        def create_collection(self, **kwargs: object) -> None:
            self.created = kwargs

        def create_payload_index(self, **kwargs: object) -> None:
            return None

        def update_collection(self, **kwargs: object) -> None:
            self.updated = kwargs

    client = FakeQdrantClient()
    config = IndexingConfig(
        collection_name="server_collection",
        qdrant_shard_number=4,
        qdrant_bulk_indexing_threshold_kb=10_000_000,
        qdrant_restore_indexing_threshold_kb=20_000,
    )

    created, removed_count, statuses = ensure_collection(
        client,  # type: ignore[arg-type]
        config,
        collection_name="server_collection",
        vector_size=4,
    )
    restored = restore_collection_indexing_threshold(
        client,  # type: ignore[arg-type]
        config,
        collection_name="server_collection",
    )

    assert created is True
    assert removed_count == 0
    assert statuses["law_id"] == "created"
    assert statuses["passage_status"] == "created"
    assert statuses["content_availability"] == "created"
    assert client.created["shard_number"] == 4
    assert client.created["optimizers_config"].indexing_threshold == 10_000_000  # type: ignore[union-attr]
    assert restored is True
    assert client.updated["optimizers_config"].indexing_threshold == 20_000  # type: ignore[union-attr]


def test_upload_point_batch_can_use_parallel_upload_points() -> None:
    class FakeQdrantClient:
        def __init__(self) -> None:
            self.upload_calls: list[dict[str, object]] = []
            self.upsert_calls: list[dict[str, object]] = []

        def upload_points(self, **kwargs: object) -> None:
            self.upload_calls.append(kwargs)

        def upsert(self, **kwargs: object) -> None:
            self.upsert_calls.append(kwargs)

    client = FakeQdrantClient()
    point = PreparedPoint(
        chunk_id="c1",
        point_id="8d927ca1-6c96-5639-b69d-75bc1dd7ab35",
        embedding_text="test",
        payload={"chunk_id": "c1"},
        content_hash="hash",
    )

    upload_point_batch(
        client,  # type: ignore[arg-type]
        collection_name="server_collection",
        points=[point],
        vectors=[[1.0, 0.0, 0.0, 0.0]],
        sparse_vectors=[([1], [1.0])],
        max_retries=3,
        upload_batch_size=128,
        upload_parallel=4,
    )

    assert not client.upsert_calls
    assert client.upload_calls[0]["batch_size"] == 128
    assert client.upload_calls[0]["parallel"] == 4
    assert client.upload_calls[0]["max_retries"] == 3
    assert client.upload_calls[0]["wait"] is True


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


def test_indexing_cli_parses_qdrant_server_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(config: IndexingConfig) -> dict[str, object]:
        captured.update(config.model_dump())
        return {
            "ready_for_retrieval": True,
            "collection_name": "cli_collection",
            "indexed_count": 1,
            "run_id": "cli",
        }

    monkeypatch.setattr("legal_rag.indexing.cli.run_indexing_pipeline", fake_run)

    assert indexing_main(
        [
            "--qdrant-url",
            "http://127.0.0.1:6333",
            "--qdrant-api-key",
            "secret",
            "--qdrant-upload-parallel",
            "4",
            "--qdrant-shard-number",
            "4",
            "--qdrant-bulk-indexing-threshold-kb",
            "10000000",
            "--qdrant-restore-indexing-threshold-kb",
            "20000",
        ]
    ) == 0

    assert captured["qdrant_url"] == "http://127.0.0.1:6333"
    assert captured["qdrant_api_key"] == "secret"
    assert captured["qdrant_upload_parallel"] == 4
    assert captured["qdrant_shard_number"] == 4
    assert captured["qdrant_bulk_indexing_threshold_kb"] == 10_000_000
    assert captured["qdrant_restore_indexing_threshold_kb"] == 20_000

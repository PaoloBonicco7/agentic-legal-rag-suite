from __future__ import annotations

import json
from pathlib import Path

from legal_indexing.rag_runtime.config import RagRuntimeConfig
from legal_indexing.rag_runtime.index_contract import resolve_index_contract


def _write_indexing_summary(
    artifacts_root: Path,
    run_id: str,
    *,
    collection_name: str,
    qdrant_path: Path,
    hybrid_index: dict | None = None,
    index_contract: dict | None = None,
    collection: dict | None = None,
) -> Path:
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "indexing_summary.json"
    payload = {
        "summary": {
            "run_id": run_id,
            "collection_name": collection_name,
            "qdrant_path": str(qdrant_path),
        },
        "hybrid_index": hybrid_index or {},
        "index_contract": index_contract or {},
        "collection": collection or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_index_contract_explicit_collection_has_priority(tmp_path: Path) -> None:
    cfg = RagRuntimeConfig(
        collection_name="laws_collection_explicit",
        qdrant_path=tmp_path / "qdrant",
        indexing_artifacts_root=tmp_path / "artifacts",
    )

    contract = resolve_index_contract(cfg)
    assert contract.source == "explicit"
    assert contract.collection_name == "laws_collection_explicit"
    assert contract.run_id is None
    assert contract.indexing_summary_path is None
    assert contract.qdrant_path == (tmp_path / "qdrant").resolve()


def test_index_contract_resolves_latest_artifact_when_run_not_specified(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    qdrant_a = tmp_path / "qdrant_a"
    qdrant_b = tmp_path / "qdrant_b"
    _write_indexing_summary(
        artifacts_root,
        "20260223_100000",
        collection_name="collection_a",
        qdrant_path=qdrant_a,
    )
    _write_indexing_summary(
        artifacts_root,
        "20260223_110000",
        collection_name="collection_b",
        qdrant_path=qdrant_b,
    )

    cfg = RagRuntimeConfig(
        indexing_artifacts_root=artifacts_root,
        qdrant_path=tmp_path / "fallback_qdrant",
    )
    contract = resolve_index_contract(cfg)

    assert contract.source == "artifact"
    assert contract.run_id == "20260223_110000"
    assert contract.collection_name == "collection_b"
    assert contract.qdrant_path == qdrant_b.resolve()
    assert contract.indexing_summary_path == (
        artifacts_root / "20260223_110000" / "indexing_summary.json"
    )


def test_index_contract_resolves_selected_run_id(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    qdrant_a = tmp_path / "qdrant_a"
    qdrant_b = tmp_path / "qdrant_b"
    _write_indexing_summary(
        artifacts_root,
        "20260223_120000",
        collection_name="collection_a",
        qdrant_path=qdrant_a,
    )
    _write_indexing_summary(
        artifacts_root,
        "20260223_130000",
        collection_name="collection_b",
        qdrant_path=qdrant_b,
    )

    cfg = RagRuntimeConfig(
        indexing_artifacts_root=artifacts_root,
        indexing_run_id="20260223_120000",
        qdrant_path=tmp_path / "fallback_qdrant",
    )
    contract = resolve_index_contract(cfg)

    assert contract.source == "artifact"
    assert contract.run_id == "20260223_120000"
    assert contract.collection_name == "collection_a"
    assert contract.qdrant_path == qdrant_a.resolve()


def test_index_contract_default_prefers_timestamp_runs_over_named_smoke(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    qdrant_ts = tmp_path / "qdrant_ts"
    qdrant_smoke = tmp_path / "qdrant_smoke"
    _write_indexing_summary(
        artifacts_root,
        "20260223_120000",
        collection_name="collection_ts",
        qdrant_path=qdrant_ts,
    )
    _write_indexing_summary(
        artifacts_root,
        "smoke_local_20260223_04_force",
        collection_name="collection_smoke",
        qdrant_path=qdrant_smoke,
    )

    cfg = RagRuntimeConfig(
        indexing_artifacts_root=artifacts_root,
        qdrant_path=tmp_path / "fallback_qdrant",
    )
    contract = resolve_index_contract(cfg)

    assert contract.source == "artifact"
    assert contract.run_id == "20260223_120000"
    assert contract.collection_name == "collection_ts"


def test_index_contract_reads_hybrid_capabilities_from_summary(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    qdrant = tmp_path / "qdrant"
    sparse_artifact = tmp_path / "artifacts" / "20260223_140000" / "sparse_encoder.json"
    _write_indexing_summary(
        artifacts_root,
        "20260223_140000",
        collection_name="collection_hybrid",
        qdrant_path=qdrant,
        hybrid_index={
            "dense_vector_size": 768,
            "sparse_enabled": True,
            "sparse_vector_name": "bm25",
            "sparse_artifact_path": str(sparse_artifact),
        },
    )

    cfg = RagRuntimeConfig(
        indexing_artifacts_root=artifacts_root,
        qdrant_path=qdrant,
    )
    contract = resolve_index_contract(cfg)
    assert contract.collection_name == "collection_hybrid"
    assert contract.dense_vector_size == 768
    assert contract.sparse_enabled is True
    assert contract.sparse_vector_name == "bm25"
    assert contract.sparse_artifacts_path == sparse_artifact.resolve()


def test_index_contract_reads_eval_coverage_fields(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    qdrant = tmp_path / "qdrant"
    _write_indexing_summary(
        artifacts_root,
        "20260223_150000",
        collection_name="collection_eval",
        qdrant_path=qdrant,
        index_contract={
            "eval_reference_coverage": 0.98,
            "missing_references_sample": ["Legge regionale 1 gennaio 2024, n. 1"],
            "payload_field_coverage": {"law_id": 1.0, "article_id": 1.0},
        },
    )

    cfg = RagRuntimeConfig(indexing_artifacts_root=artifacts_root, qdrant_path=qdrant)
    contract = resolve_index_contract(cfg)
    assert contract.eval_reference_coverage == 0.98
    assert contract.missing_references_sample == ("Legge regionale 1 gennaio 2024, n. 1",)
    assert contract.payload_field_coverage == {"law_id": 1.0, "article_id": 1.0}


def test_index_contract_reads_collection_points_count(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    qdrant = tmp_path / "qdrant"
    _write_indexing_summary(
        artifacts_root,
        "20260223_160000",
        collection_name="collection_points",
        qdrant_path=qdrant,
        collection={
            "collection_name": "collection_points",
            "points_count_exact": 1234,
        },
    )

    cfg = RagRuntimeConfig(indexing_artifacts_root=artifacts_root, qdrant_path=qdrant)
    contract = resolve_index_contract(cfg)
    assert contract.collection_points_count == 1234

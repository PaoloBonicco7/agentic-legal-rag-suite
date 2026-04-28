from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient

from .artifacts import ensure_run_dir, write_json, write_jsonl
from .chunk_refinement import RefinedChunk, refine_chunks_with_diagnostics
from .embeddings import SupportsEmbedding, build_embedder
from .hashing import content_hash_for_text, payload_hash, point_id_from_chunk_id
from .io import (
    DatasetValidationResult,
    dataset_hash_from_manifest,
    load_dataset_bundle,
    validate_dataset,
)
from .law_references import (
    EvalCoverageReport,
    LawCatalog,
    build_law_catalog,
    compute_eval_reference_coverage,
    load_eval_reference_texts,
)
from .metadata import refined_chunk_payload
from .qdrant_store import (
    CollectionVectorCapabilities,
    PreparedPoint,
    SyncStats,
    build_collection_name,
    collection_vector_capabilities,
    collection_stats,
    ensure_collection,
    get_vector_size,
    sync_points_incremental,
    validate_filtered_query,
    validate_no_duplicate_chunk_ids,
)
from .settings import IndexingConfig
from .sparse import SparseEncoder, SparseVectorData, build_sparse_encoder


@dataclass(frozen=True)
class IndexingRunSummary:
    run_id: str
    artifacts_dir: Path
    collection_name: str
    qdrant_path: Path

    total_passages: int
    total_refined_chunks: int

    total_embedded: int
    skipped_unchanged: int
    failures: int

    collection_points_count: int
    duplicate_chunk_ids_ok: bool
    filter_validation_ok: bool
    sparse_enabled: bool
    sparse_vector_name: str | None
    sparse_vocab_size: int
    sparse_artifact_path: Path | None
    eval_reference_coverage: float | None
    eval_references_total: int
    eval_references_resolved: int
    payload_field_coverage: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifacts_dir": str(self.artifacts_dir),
            "collection_name": self.collection_name,
            "qdrant_path": str(self.qdrant_path),
            "total_passages": self.total_passages,
            "total_refined_chunks": self.total_refined_chunks,
            "total_embedded": self.total_embedded,
            "skipped_unchanged": self.skipped_unchanged,
            "failures": self.failures,
            "collection_points_count": self.collection_points_count,
            "duplicate_chunk_ids_ok": self.duplicate_chunk_ids_ok,
            "filter_validation_ok": self.filter_validation_ok,
            "sparse_enabled": self.sparse_enabled,
            "sparse_vector_name": self.sparse_vector_name,
            "sparse_vocab_size": self.sparse_vocab_size,
            "sparse_artifact_path": (
                str(self.sparse_artifact_path) if self.sparse_artifact_path else None
            ),
            "eval_reference_coverage": self.eval_reference_coverage,
            "eval_references_total": self.eval_references_total,
            "eval_references_resolved": self.eval_references_resolved,
            "payload_field_coverage": dict(self.payload_field_coverage),
        }


def _select_passages(passages: list[Any], subset_limit: int | None) -> list[Any]:
    if subset_limit is None:
        return passages
    if subset_limit <= 0:
        return []
    return passages[:subset_limit]


def _build_prepared_points(
    refined_chunks: list[RefinedChunk], *, dataset_hash: str, profile_id: str, embedding_model: str
) -> list[PreparedPoint]:
    points: list[PreparedPoint] = []
    seen_chunk_ids: set[str] = set()

    for chunk in refined_chunks:
        if chunk.chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate refined chunk_id: {chunk.chunk_id}")
        seen_chunk_ids.add(chunk.chunk_id)

        c_hash = content_hash_for_text(chunk.text_for_embedding)
        payload = refined_chunk_payload(
            chunk,
            dataset_hash=dataset_hash,
            chunking_profile_id=profile_id,
            embedding_model=embedding_model,
            content_hash=c_hash,
            payload_hash=None,
        )
        p_hash = payload_hash(payload)
        payload["payload_hash"] = p_hash

        points.append(
            PreparedPoint(
                chunk_id=chunk.chunk_id,
                point_id=point_id_from_chunk_id(chunk.chunk_id),
                embedding_text=chunk.text_for_embedding,
                payload=payload,
                content_hash=c_hash,
            )
        )

    return points


def _choose_filter_sample(chunks: list[RefinedChunk]) -> RefinedChunk | None:
    for chunk in chunks:
        if "current" in chunk.index_views and chunk.law_status == "current":
            return chunk
    if chunks:
        return chunks[0]
    return None


def _validate_dataset_or_raise(result: DatasetValidationResult, *, strict: bool) -> None:
    if result.errors and strict:
        details = "\n".join(f"- {msg}" for msg in result.errors)
        raise RuntimeError(f"Dataset validation failed:\n{details}")


def _payload_field_coverage(
    points: list[PreparedPoint], *, required_fields: Sequence[str]
) -> dict[str, float]:
    if not points:
        return {field: 0.0 for field in required_fields}

    total = float(len(points))
    counts: dict[str, int] = {field: 0 for field in required_fields}
    for point in points:
        payload = point.payload
        for field in required_fields:
            value = payload.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            counts[field] += 1
    return {field: counts[field] / total for field in required_fields}


def _build_eval_coverage_report(config: IndexingConfig, catalog: LawCatalog) -> EvalCoverageReport:
    refs = load_eval_reference_texts(
        [
            config.resolved_eval_questions_csv,
            config.resolved_eval_questions_no_hint_csv,
        ]
    )
    return compute_eval_reference_coverage(catalog=catalog, references=refs)


def run_indexing_pipeline(
    config: IndexingConfig,
    *,
    embedder: SupportsEmbedding | None = None,
    client: QdrantClient | None = None,
) -> IndexingRunSummary:
    config = config.with_overrides()
    run_id = config.effective_run_id

    artifacts_dir = ensure_run_dir(config.resolved_artifacts_root, run_id)
    write_json(artifacts_dir / "config.json", config.to_dict())

    validation = validate_dataset(config.resolved_dataset_dir, strict=config.strict_validation)
    write_json(artifacts_dir / "dataset_validation.json", validation)
    _validate_dataset_or_raise(validation, strict=config.strict_validation)

    bundle = load_dataset_bundle(config.resolved_dataset_dir)
    dataset_hash = dataset_hash_from_manifest(bundle.manifest)

    selected_passages = _select_passages(bundle.passages, config.subset_limit)
    refined_chunks, chunking_diag = refine_chunks_with_diagnostics(
        selected_passages,
        bundle.article_order_by_id,
        config.chunking_profile,
    )

    write_json(artifacts_dir / "chunking_stats.json", chunking_diag)
    write_json(
        artifacts_dir / "chunk_examples.json",
        {
            "merge_examples": list(chunking_diag.merge_examples),
            "split_examples": list(chunking_diag.split_examples),
            "sample_refined_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "law_id": c.law_id,
                    "article_id": c.article_id,
                    "source_passage_ids": list(c.source_passage_ids),
                    "source_passage_labels": list(c.source_passage_labels),
                    "article_chunk_order": c.article_chunk_order,
                    "text_preview": c.text[:220],
                }
                for c in refined_chunks[:20]
            ],
        },
    )

    points = _build_prepared_points(
        refined_chunks,
        dataset_hash=dataset_hash,
        profile_id=config.chunking_profile.profile_id,
        embedding_model=config.embedding_model,
    )
    payload_coverage = _payload_field_coverage(
        points,
        required_fields=(
            "chunk_id",
            "law_id",
            "article_id",
            "law_year",
            "law_number",
            "law_status",
            "relation_types",
            "source_passage_ids",
            "source_chunk_ids",
        ),
    )
    law_catalog = build_law_catalog(config.resolved_dataset_dir)
    eval_coverage = _build_eval_coverage_report(config, law_catalog)
    write_json(artifacts_dir / "index_contract_report.json", eval_coverage.to_dict())

    if (
        config.index_contract_enforce_eval_coverage
        and eval_coverage.coverage is not None
        and eval_coverage.coverage < float(config.index_contract_min_eval_coverage)
    ):
        raise RuntimeError(
            "Index contract check failed: evaluation reference coverage below threshold. "
            f"coverage={eval_coverage.coverage:.3f}, "
            f"threshold={config.index_contract_min_eval_coverage:.3f}. "
            "Rebuild dataset/index to include missing laws referenced by evaluation."
        )

    qdrant_path = config.resolved_qdrant_path
    qdrant_path.mkdir(parents=True, exist_ok=True)

    created_client = client is None
    if client is None:
        client = QdrantClient(path=str(qdrant_path))

    assert client is not None

    try:
        collection_name = build_collection_name(config, dataset_hash=dataset_hash)
        warnings: list[str] = []

        if embedder is None:
            embedder = build_embedder(config)

        if client.collection_exists(collection_name=collection_name):
            vector_size = get_vector_size(client, collection_name)
        else:
            if not points:
                raise RuntimeError("Cannot create a new collection with zero chunks to infer vector size")
            probe = embedder.embed_texts([points[0].embedding_text])
            if len(probe) != 1:
                raise RuntimeError("Embedding probe returned unexpected vector count")
            vector_size = len(probe[0])

        sparse_encoder: SparseEncoder | None = None
        sparse_vectors_by_chunk: dict[str, SparseVectorData] | None = None
        sparse_artifact_path: Path | None = None
        sparse_build_error: str | None = None
        if config.sparse_enabled:
            try:
                sparse_encoder = build_sparse_encoder(
                    [p.embedding_text for p in points],
                    min_token_len=config.sparse_min_token_len,
                    stopwords_lang=config.sparse_stopwords_lang,
                )
                sparse_vectors_by_chunk = {
                    p.chunk_id: sparse_encoder.transform(p.embedding_text) for p in points
                }
                if config.sparse_store_artifacts:
                    sparse_artifact_path = artifacts_dir / "sparse_encoder.json"
                    sparse_encoder.save_json(sparse_artifact_path)
            except Exception as exc:
                sparse_build_error = f"{type(exc).__name__}: {exc}"
                if config.strict_validation:
                    raise RuntimeError(f"Sparse encoder build failed: {sparse_build_error}")
                warnings.append(
                    f"Sparse encoder build failed; fallback dense-only: {sparse_build_error}"
                )
                sparse_encoder = None
                sparse_vectors_by_chunk = None
                sparse_artifact_path = None

        collection_cfg = config
        if config.sparse_enabled and sparse_encoder is None:
            collection_cfg = config.with_overrides(sparse_enabled=False)

        ensure_collection(
            client,
            collection_cfg,
            collection_name=collection_name,
            vector_size=vector_size,
        )
        vector_caps: CollectionVectorCapabilities = collection_vector_capabilities(
            client, collection_name
        )

        sparse_write_enabled = (
            sparse_vectors_by_chunk is not None
            and vector_caps.sparse_enabled
            and config.sparse_vector_name in set(vector_caps.sparse_vector_names)
        )
        if config.sparse_enabled and not sparse_write_enabled:
            warnings.append(
                "Sparse channel requested but unavailable on collection; running dense-only."
            )

        sync_stats: SyncStats = sync_points_incremental(
            client,
            collection_name=collection_name,
            points=points,
            embedder=embedder,
            force_reembed=bool(config.force_reembed),
            embed_batch_size=max(1, int(config.embedding_batch_size)),
            dense_vector_name=vector_caps.dense_vector_name,
            sparse_vector_name=(config.sparse_vector_name if sparse_write_enabled else None),
            sparse_vectors_by_chunk=(sparse_vectors_by_chunk if sparse_write_enabled else None),
        )

        failure_rows = [
            {"chunk_id": f.chunk_id, "stage": f.stage, "error": f.error}
            for f in sync_stats.failures
        ]
        write_jsonl(artifacts_dir / "failures.jsonl", failure_rows)

        collection_info = collection_stats(client, collection_name)
        write_json(artifacts_dir / "collection_info.json", collection_info)

        duplicate_check = validate_no_duplicate_chunk_ids(client, collection_name)
        write_json(artifacts_dir / "duplicates_validation.json", duplicate_check)

        filter_sample = _choose_filter_sample(refined_chunks)
        if filter_sample is None:
            filter_validation = {
                "ok": True,
                "reason": "No refined chunks available; filter validation skipped",
                "matches": [],
            }
        else:
            filter_validation = validate_filtered_query(
                client,
                collection_name=collection_name,
                sample_point_id=point_id_from_chunk_id(filter_sample.chunk_id),
                law_id=filter_sample.law_id,
            )
        write_json(artifacts_dir / "filter_validation.json", filter_validation)

        summary = IndexingRunSummary(
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
            total_passages=len(selected_passages),
            total_refined_chunks=len(refined_chunks),
            total_embedded=sync_stats.embedded,
            skipped_unchanged=sync_stats.skipped,
            failures=sync_stats.failure_count,
            collection_points_count=int(collection_info.get("points_count_exact") or 0),
            duplicate_chunk_ids_ok=bool(duplicate_check.get("ok")),
            filter_validation_ok=bool(filter_validation.get("ok")),
            sparse_enabled=bool(sparse_write_enabled),
            sparse_vector_name=(config.sparse_vector_name if sparse_write_enabled else None),
            sparse_vocab_size=int(sparse_encoder.vocab_size) if sparse_encoder else 0,
            sparse_artifact_path=sparse_artifact_path,
            eval_reference_coverage=eval_coverage.coverage,
            eval_references_total=eval_coverage.references_total,
            eval_references_resolved=eval_coverage.references_resolved,
            payload_field_coverage=payload_coverage,
        )

        write_json(
            artifacts_dir / "indexing_summary.json",
            {
                "summary": summary,
                "sync_stats": {
                    "total_chunks": sync_stats.total_chunks,
                    "to_process": sync_stats.to_process,
                    "embedded": sync_stats.embedded,
                    "skipped": sync_stats.skipped,
                    "upserted": sync_stats.upserted,
                    "failures": sync_stats.failure_count,
                },
                "collection": collection_info,
                "duplicates_validation": duplicate_check,
                "filter_validation": filter_validation,
                "hybrid_index": {
                    "dense_vector_size": vector_caps.dense_vector_size,
                    "dense_vector_name": vector_caps.dense_vector_name,
                    "sparse_enabled": bool(sparse_write_enabled),
                    "sparse_vector_name": (
                        config.sparse_vector_name if sparse_write_enabled else None
                    ),
                    "sparse_vector_names_in_collection": list(vector_caps.sparse_vector_names),
                    "sparse_vocab_size": (
                        int(sparse_encoder.vocab_size) if sparse_encoder else 0
                    ),
                    "sparse_doc_count": (
                        int(sparse_encoder.doc_count) if sparse_encoder else 0
                    ),
                    "sparse_artifact_path": (
                        str(sparse_artifact_path) if sparse_artifact_path else None
                    ),
                    "sparse_build_error": sparse_build_error,
                    "warnings": warnings,
                },
                "index_contract": {
                    "eval_reference_coverage": eval_coverage.coverage,
                    "eval_references_total": eval_coverage.references_total,
                    "eval_references_with_any_law": eval_coverage.references_with_any_law,
                    "eval_references_resolved": eval_coverage.references_resolved,
                    "missing_references_sample": list(eval_coverage.missing_references_sample),
                    "coverage_threshold": config.index_contract_min_eval_coverage,
                    "coverage_enforced": bool(config.index_contract_enforce_eval_coverage),
                    "coverage_ok": (
                        (
                            eval_coverage.coverage is None
                            or eval_coverage.coverage
                            >= float(config.index_contract_min_eval_coverage)
                        )
                    ),
                    "payload_field_coverage": payload_coverage,
                },
            },
        )

        if sync_stats.failure_count > 0 and config.strict_validation:
            raise RuntimeError(
                f"Indexing completed with {sync_stats.failure_count} failures. "
                f"See failures.jsonl in {artifacts_dir}"
            )

        if not duplicate_check.get("ok") and config.strict_validation:
            raise RuntimeError(
                f"Duplicate chunk_id found in Qdrant collection {collection_name!r}. "
                f"See duplicates_validation.json in {artifacts_dir}"
            )

        if not filter_validation.get("ok") and config.strict_validation:
            raise RuntimeError(
                "Metadata filter validation failed. "
                f"See filter_validation.json in {artifacts_dir}"
            )

        return summary
    finally:
        if created_client:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["IndexingRunSummary", "run_indexing_pipeline"]

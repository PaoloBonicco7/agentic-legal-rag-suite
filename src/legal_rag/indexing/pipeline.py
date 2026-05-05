"""End-to-end Qdrant indexing pipeline for the clean legal dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

from qdrant_client import QdrantClient

from .dataset import load_chunks, read_manifest, validate_clean_dataset
from .embeddings import SupportsEmbedding, supports_sparse_embedding
from .embeddings import build_embedder
from .hashing import content_hash_for_text, payload_hash, point_id_from_chunk_id
from .io import finalize_run_dir, now_utc, prepare_run_dir, sha256_file, write_json, write_jsonl
from .models import FILTERABLE_FIELDS, INDEXING_SCHEMA_VERSION, REQUIRED_PAYLOAD_FIELDS, IndexingConfig
from .qdrant_store import (
    PreparedPoint,
    build_collection_name,
    collection_point_count,
    connect_qdrant,
    ensure_collection,
    fetch_existing_content_hashes,
    upload_point_batch,
    validate_no_duplicate_chunk_ids,
)
from .retrieval import search_index


@dataclass(frozen=True)
class SyncStats:
    """Point synchronization counters."""

    selected: int
    embedded: int
    skipped: int
    upserted: int
    failures: tuple[dict[str, str], ...]

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def _select_chunks(chunks: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return chunks if limit is None else chunks[:limit]


def _payload_from_chunk(chunk: dict[str, Any], *, dataset_hash: str, embedding_model: str) -> dict[str, Any]:
    embedding_text = str(chunk.get("text_for_embedding") or "")
    content_hash = content_hash_for_text(embedding_text)
    payload = {field: chunk.get(field) for field in sorted(REQUIRED_PAYLOAD_FIELDS) if field != "content_hash"}
    payload["content_hash"] = content_hash
    payload["text_for_embedding"] = embedding_text
    payload["dataset_source_hash"] = dataset_hash
    payload["embedding_model"] = embedding_model
    payload["payload_hash"] = payload_hash(payload)
    return payload


def _prepare_points(chunks: Sequence[dict[str, Any]], *, dataset_hash: str, embedding_model: str) -> list[PreparedPoint]:
    seen: set[str] = set()
    points: list[PreparedPoint] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            raise ValueError("Cannot prepare Qdrant point without chunk_id")
        if chunk_id in seen:
            raise ValueError(f"Duplicate selected chunk_id: {chunk_id}")
        seen.add(chunk_id)
        embedding_text = str(chunk.get("text_for_embedding") or "").strip()
        if not embedding_text:
            raise ValueError(f"{chunk_id}: text_for_embedding is empty")
        payload = _payload_from_chunk(chunk, dataset_hash=dataset_hash, embedding_model=embedding_model)
        points.append(
            PreparedPoint(
                chunk_id=chunk_id,
                point_id=point_id_from_chunk_id(chunk_id),
                embedding_text=embedding_text,
                payload=payload,
                content_hash=str(payload["content_hash"]),
            )
        )
    return points


def _payload_profile(points: Sequence[PreparedPoint]) -> dict[str, Any]:
    total = len(points)
    field_summary: dict[str, dict[str, Any]] = {}
    for field in sorted(REQUIRED_PAYLOAD_FIELDS):
        present = 0
        non_empty = 0
        types: set[str] = set()
        for point in points:
            value = point.payload.get(field)
            if value is not None:
                present += 1
                types.add(type(value).__name__)
            if isinstance(value, list):
                if value:
                    non_empty += 1
            elif isinstance(value, str):
                if value.strip() or field == "structure_path":
                    non_empty += 1
            elif value is not None:
                non_empty += 1
        field_summary[field] = {
            "present": present,
            "missing": total - present,
            "present_coverage": (present / total) if total else 0.0,
            "non_empty_coverage": (non_empty / total) if total else 0.0,
            "types": sorted(types),
        }
    return {
        "total_points_profiled": total,
        "required_fields": sorted(REQUIRED_PAYLOAD_FIELDS),
        "filterable_fields": list(FILTERABLE_FIELDS),
        "fields": field_summary,
    }


def _vectors_from_record(record: Any) -> list[float] | None:
    vector_obj = getattr(record, "vector", None)
    if isinstance(vector_obj, list) and vector_obj:
        return [float(value) for value in vector_obj]
    if isinstance(vector_obj, dict):
        for key in ("", "dense", "default"):
            value = vector_obj.get(key)
            if isinstance(value, list) and value:
                return [float(item) for item in value]
        for value in vector_obj.values():
            if isinstance(value, list) and value:
                return [float(item) for item in value]
    return None


def _validate_filtered_query(client: QdrantClient, *, collection_name: str, sample: PreparedPoint) -> dict[str, Any]:
    records = client.retrieve(collection_name=collection_name, ids=[sample.point_id], with_payload=False, with_vectors=True)
    if not records:
        return {"ok": False, "reason": f"Sample point not found: {sample.point_id}", "matches": []}
    vector = _vectors_from_record(records[0])
    if not vector:
        return {"ok": False, "reason": "Sample point vector not available", "matches": []}
    law_id = str(sample.payload.get("law_id") or "")
    law_status = str(sample.payload.get("law_status") or "")
    from .retrieval import build_qdrant_filter

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        using="dense",
        query_filter=build_qdrant_filter(law_ids=[law_id], law_status=law_status, index_view="current" if "current" in (sample.payload.get("index_views") or []) else None),
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    violations: list[str] = []
    matches: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        row = {
            "chunk_id": payload.get("chunk_id"),
            "law_id": payload.get("law_id"),
            "law_status": payload.get("law_status"),
            "index_views": payload.get("index_views"),
            "score": float(point.score),
        }
        matches.append(row)
        if payload.get("law_id") != law_id:
            violations.append(f"{payload.get('chunk_id')}: law_id mismatch")
        if payload.get("law_status") != law_status:
            violations.append(f"{payload.get('chunk_id')}: law_status mismatch")
    return {"ok": not violations and bool(matches), "matches": matches, "violations": violations}


def _sync_points(
    client: QdrantClient,
    *,
    collection_name: str,
    points: list[PreparedPoint],
    embedder: SupportsEmbedding,
    config: IndexingConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SyncStats:
    existing_hashes: dict[str, str] = {}
    if not config.force_rebuild:
        existing_hashes = fetch_existing_content_hashes(client, collection_name=collection_name, point_ids=[point.point_id for point in points])

    to_process: list[PreparedPoint] = []
    skipped = 0
    for point in points:
        if existing_hashes.get(point.point_id) == point.content_hash:
            skipped += 1
        else:
            to_process.append(point)

    failures: list[dict[str, str]] = []
    embedded = 0
    upserted = 0
    started = perf_counter()
    total_to_process = len(to_process)
    if progress_callback:
        progress_callback(
            {
                "event": "sync_started",
                "selected": len(points),
                "skipped": skipped,
                "to_process": total_to_process,
                "batch_size": config.batch_size,
            }
        )

    def emit_batch_progress(*, batch_number: int, batch_total: int, batch_size: int, batch_started: float) -> None:
        if not progress_callback:
            return
        elapsed = max(perf_counter() - started, 0.001)
        processed = embedded + skipped
        rate = embedded / elapsed if embedded else 0.0
        remaining = max(len(points) - processed, 0)
        progress_callback(
            {
                "event": "batch_finished",
                "batch": batch_number,
                "batch_total": batch_total,
                "batch_size": batch_size,
                "embedded": embedded,
                "upserted": upserted,
                "skipped": skipped,
                "failures": len(failures),
                "processed": processed,
                "selected": len(points),
                "percent": round((processed / len(points)) * 100.0, 2) if points else 100.0,
                "rate_chunks_per_second": round(rate, 3),
                "eta_seconds": round(remaining / rate, 1) if rate else None,
                "batch_seconds": round(perf_counter() - batch_started, 2),
            }
        )

    for start in range(0, len(to_process), config.batch_size):
        batch = to_process[start : start + config.batch_size]
        batch_started = perf_counter()
        batch_number = (start // config.batch_size) + 1
        batch_total = ((total_to_process + config.batch_size - 1) // config.batch_size) if total_to_process else 0
        try:
            vectors = embedder.embed_texts([point.embedding_text for point in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(f"Embedding vector count mismatch: got {len(vectors)}, expected {len(batch)}")
            sparse_vectors = None
            if config.hybrid_enabled:
                if not supports_sparse_embedding(embedder):
                    raise RuntimeError("Hybrid indexing requires an embedder with embed_sparse_texts().")
                sparse_vectors = embedder.embed_sparse_texts([point.embedding_text for point in batch])  # type: ignore[attr-defined]
        except Exception as exc:
            if len(batch) == 1:
                failures.append({"chunk_id": batch[0].chunk_id, "stage": "embedding", "error": str(exc)})
                emit_batch_progress(
                    batch_number=batch_number,
                    batch_total=batch_total,
                    batch_size=len(batch),
                    batch_started=batch_started,
                )
                continue
            for point in batch:
                try:
                    vector = embedder.embed_texts([point.embedding_text])
                    sparse_vector = None
                    if config.hybrid_enabled:
                        if not supports_sparse_embedding(embedder):
                            raise RuntimeError("Hybrid indexing requires an embedder with embed_sparse_texts().")
                        sparse_vector = embedder.embed_sparse_texts([point.embedding_text])  # type: ignore[attr-defined]
                    upload_point_batch(
                        client,
                        collection_name=collection_name,
                        points=[point],
                        vectors=vector,
                        sparse_vectors=sparse_vector,
                        max_retries=config.upload_max_retries,
                    )
                    embedded += 1
                    upserted += 1
                except Exception as inner_exc:
                    failures.append({"chunk_id": point.chunk_id, "stage": "embedding_or_upsert", "error": str(inner_exc)})
            emit_batch_progress(
                batch_number=batch_number,
                batch_total=batch_total,
                batch_size=len(batch),
                batch_started=batch_started,
            )
            continue

        embedded += len(batch)
        for points_part_start in range(0, len(batch), config.upload_batch_size):
            point_batch = batch[points_part_start : points_part_start + config.upload_batch_size]
            vector_batch = vectors[points_part_start : points_part_start + config.upload_batch_size]
            sparse_vector_batch = sparse_vectors[points_part_start : points_part_start + config.upload_batch_size] if sparse_vectors is not None else None
            try:
                upload_point_batch(
                    client,
                    collection_name=collection_name,
                    points=point_batch,
                    vectors=vector_batch,
                    sparse_vectors=sparse_vector_batch,
                    max_retries=config.upload_max_retries,
                )
                upserted += len(point_batch)
            except Exception as exc:
                if len(point_batch) == 1:
                    failures.append({"chunk_id": point_batch[0].chunk_id, "stage": "upsert", "error": str(exc)})
                    continue
                for index, (point, vector) in enumerate(zip(point_batch, vector_batch, strict=True)):
                    try:
                        sparse_vector = [sparse_vector_batch[index]] if sparse_vector_batch is not None else None
                        upload_point_batch(
                            client,
                            collection_name=collection_name,
                            points=[point],
                            vectors=[vector],
                            sparse_vectors=sparse_vector,
                            max_retries=config.upload_max_retries,
                        )
                        upserted += 1
                    except Exception as inner_exc:
                        failures.append({"chunk_id": point.chunk_id, "stage": "upsert", "error": str(inner_exc)})

        emit_batch_progress(
            batch_number=batch_number,
            batch_total=batch_total,
            batch_size=len(batch),
            batch_started=batch_started,
        )

    return SyncStats(selected=len(points), embedded=embedded, skipped=skipped, upserted=upserted, failures=tuple(failures))


def _write_quality_report(path: Path, *, manifest: dict[str, Any]) -> None:
    gates = manifest["quality_gates"]
    lines = [
        "# 03 - Indexing Contract Quality Report",
        "",
        f"- Ready for retrieval: **{manifest['ready_for_retrieval']}**",
        f"- Collection: `{manifest['collection_name']}`",
        f"- Indexed count: {manifest['indexed_count']}",
        f"- Skipped unchanged: {manifest['skipped_count']}",
        f"- Failures: {manifest['failure_count']}",
        "",
        "## Quality Gates",
    ]
    for gate, ok in gates.items():
        lines.append(f"- `{gate}`: **{ok}**")
    lines.extend(
        [
            "",
            "## Embedding",
            f"- backend: `{manifest['embedding']['backend']}`",
            f"- model: `{manifest['embedding']['model']}`",
            f"- hybrid enabled: `{manifest['embedding']['hybrid_enabled']}`",
            f"- vector size: {manifest['embedding']['vector_size']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_indexing_pipeline(
    config: IndexingConfig | dict[str, Any] | None = None,
    *,
    embedder: SupportsEmbedding | None = None,
    client: QdrantClient | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the complete clean-dataset to Qdrant indexing contract pipeline."""
    cfg = config if isinstance(config, IndexingConfig) else IndexingConfig.model_validate(config or {})
    run_id = cfg.effective_run_id
    tmp_dir = prepare_run_dir(cfg.resolved_artifacts_root, run_id)
    final_dir = cfg.resolved_artifacts_root / run_id
    created_client = client is None

    try:
        validation = validate_clean_dataset(cfg.resolved_dataset_dir, strict=cfg.strict)
        write_json(tmp_dir / "dataset_validation.json", validation.to_dict())
        if cfg.strict and not validation.ok:
            raise RuntimeError("Dataset validation failed: " + "; ".join(validation.errors))

        manifest = read_manifest(cfg.resolved_dataset_dir)
        chunks = _select_chunks(load_chunks(cfg.resolved_dataset_dir), cfg.selection_limit)
        source_hash = str(manifest.get("source_hash") or "")
        chunks_hash = str((manifest.get("output_hashes") or {}).get("chunks") or sha256_file(cfg.resolved_dataset_dir / "chunks.jsonl"))
        if not source_hash:
            source_hash = chunks_hash
        points = _prepare_points(chunks, dataset_hash=source_hash, embedding_model=cfg.resolved_embedding_model)
        if not points:
            raise RuntimeError("No chunks selected for indexing")

        if embedder is None:
            embedder = build_embedder(cfg)
        probe = embedder.embed_texts([points[0].embedding_text])
        vector_size = len(probe[0])
        if vector_size <= 0:
            raise RuntimeError("Embedding probe returned an empty vector")
        if cfg.embedding_dim is not None and cfg.embedding_dim != vector_size:
            raise RuntimeError(f"Embedding dim mismatch: configured={cfg.embedding_dim}, detected={vector_size}")
        if cfg.hybrid_enabled and not supports_sparse_embedding(embedder):
            raise RuntimeError("hybrid_enabled=True requires an embedding backend that exposes sparse vectors")

        if client is None:
            client = connect_qdrant(cfg)
        collection_name = build_collection_name(cfg, dataset_hash=source_hash)
        created_collection, removed_count, payload_index_statuses = ensure_collection(
            client,
            cfg,
            collection_name=collection_name,
            vector_size=vector_size,
        )
        sync_stats = _sync_points(
            client,
            collection_name=collection_name,
            points=points,
            embedder=embedder,
            config=cfg,
            progress_callback=progress_callback,
        )

        failures = list(sync_stats.failures)
        write_jsonl(tmp_dir / "failures.jsonl", failures)

        point_count = collection_point_count(client, collection_name=collection_name)
        duplicate_check = validate_no_duplicate_chunk_ids(client, collection_name=collection_name)
        sample = next((point for point in points if "current" in (point.payload.get("index_views") or [])), points[0])
        filter_validation = _validate_filtered_query(client, collection_name=collection_name, sample=sample)
        payload_profile = _payload_profile(points)

        retrieval_rows: list[dict[str, Any]] = []
        for query in cfg.diagnostic_queries:
            try:
                hits = search_index(
                    client,
                    collection_name=collection_name,
                    embedder=embedder,
                    query=query,
                    limit=3,
                    law_status="current",
                    index_view="current",
                    retrieval_mode="hybrid" if cfg.hybrid_enabled else "dense",
                )
                retrieval_rows.append(
                    {
                        "query": query,
                        "hits": [
                            {
                                "chunk_id": hit.chunk_id,
                                "score": hit.score,
                                "law_id": hit.payload.get("law_id"),
                                "article_id": hit.payload.get("article_id"),
                                "text_preview": hit.text[:240],
                            }
                            for hit in hits
                        ],
                    }
                )
            except Exception as exc:
                retrieval_rows.append({"query": query, "error": str(exc), "hits": []})

        indexed_count = sync_stats.upserted + sync_stats.skipped
        count_gate = point_count == len(points) if cfg.force_rebuild else point_count >= len(points)
        gates = {
            "dataset_ready_for_indexing": validation.ok,
            "selected_chunks_non_empty": len(points) > 0,
            "embedding_model_recorded": bool(cfg.embedding_model),
            "vector_size_detected": vector_size > 0,
            "indexed_count_matches_selected": indexed_count == len(points) and sync_stats.failure_count == 0,
            "collection_count_matches_selected": count_gate,
            "duplicate_chunk_ids_rejected": bool(duplicate_check.get("ok")),
            "filter_validation_queryable": bool(filter_validation.get("ok")),
            "payload_indexes_requested": all(not str(status).startswith("error:") for status in payload_index_statuses.values()),
            "required_payload_fields_present": all(
                field["missing"] == 0 for field in payload_profile["fields"].values()
            ),
        }
        index_manifest = {
            "schema_version": INDEXING_SCHEMA_VERSION,
            "created_at": now_utc(),
            "run_id": run_id,
            "config": cfg.public_dict(),
            "source_dataset_dir": str(cfg.resolved_dataset_dir),
            "source_hash": source_hash,
            "source_output_hashes": manifest.get("output_hashes", {}),
            "chunks_hash": chunks_hash,
            "collection_name": collection_name,
            "collection_created": created_collection,
            "qdrant": {
                "mode": "server" if cfg.qdrant_url else "local_path",
                "url": cfg.qdrant_url,
                "path": str(cfg.resolved_index_dir),
                "dense_vector_name": "dense",
                "sparse_vector_name": "sparse" if cfg.hybrid_enabled else None,
                "distance": cfg.qdrant_distance,
                "on_disk_payload": cfg.qdrant_on_disk_payload,
            },
            "embedding": {
                "backend": cfg.embedding_backend,
                "model": getattr(embedder, "model_name", cfg.resolved_embedding_model),
                "configured_model": cfg.embedding_model,
                "resolved_model": cfg.resolved_embedding_model,
                "hybrid_enabled": cfg.hybrid_enabled,
                "vector_size": vector_size,
            },
            "selected_count": len(points),
            "indexed_count": indexed_count,
            "embedded_count": sync_stats.embedded,
            "skipped_count": sync_stats.skipped,
            "upserted_count": sync_stats.upserted,
            "removed_count": removed_count,
            "failure_count": sync_stats.failure_count,
            "collection_points_count": point_count,
            "payload_indexes": list(FILTERABLE_FIELDS),
            "payload_index_statuses": payload_index_statuses,
            "payload_field_summary": payload_profile["fields"],
            "duplicate_validation": duplicate_check,
            "filter_validation": filter_validation,
            "quality_gates": gates,
            "ready_for_retrieval": all(gates.values()),
            "artifacts": {
                "index_manifest": "index_manifest.json",
                "payload_profile": "payload_profile.json",
                "index_quality_report": "index_quality_report.md",
                "sample_retrieval_report": "sample_retrieval_report.json",
                "diagnostic_queries": "diagnostic_queries.json",
                "failures": "failures.jsonl",
            },
        }
        write_json(tmp_dir / "payload_profile.json", payload_profile)
        write_json(tmp_dir / "sample_retrieval_report.json", {"collection_name": collection_name, "queries": retrieval_rows})
        write_json(tmp_dir / "diagnostic_queries.json", {"collection_name": collection_name, "queries": retrieval_rows})
        write_json(tmp_dir / "index_manifest.json", index_manifest)
        _write_quality_report(tmp_dir / "index_quality_report.md", manifest=index_manifest)

        if cfg.strict and failures:
            raise RuntimeError(f"Indexing completed with {len(failures)} failures. See failures.jsonl in {tmp_dir}")
        if cfg.strict and not index_manifest["ready_for_retrieval"]:
            raise RuntimeError(f"Index quality gates failed: {gates}")

        finalize_run_dir(tmp_dir, final_dir)
        return index_manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    finally:
        if created_client and client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

"""End-to-end Qdrant indexing pipeline for the clean legal dataset."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

from qdrant_client import QdrantClient

from .dataset import load_chunks, read_manifest, validate_clean_dataset
from .embeddings import SupportsEmbedding, supports_hybrid_embedding, supports_sparse_embedding
from .embeddings import build_embedder
from .hashing import canonical_dumps, content_hash_for_text, payload_hash, point_id_from_chunk_id, sha256_text
from .io import finalize_run_dir, now_utc, prepare_run_dir, read_json, sha256_file, write_json, write_jsonl
from .models import (
    FILTERABLE_FIELDS,
    IDENTITY_PAYLOAD_FIELDS,
    INDEXING_SCHEMA_VERSION,
    PROFILE_DISTRIBUTION_FIELDS,
    REQUIRED_PAYLOAD_FIELDS,
    SOURCE_CHUNK_REQUIRED_FIELDS,
    IndexingConfig,
)
from .qdrant_store import (
    PreparedPoint,
    build_collection_name,
    collection_point_count,
    connect_qdrant,
    ensure_collection,
    fetch_existing_hashes,
    fetch_reusable_vectors,
    profile_collection_payload,
    restore_collection_indexing_threshold,
    set_point_payload,
    upload_point_batch,
    validate_no_duplicate_chunk_ids,
    validate_vector_reuse_collection,
)
from .retrieval import search_index


@dataclass(frozen=True)
class SyncStats:
    """Point synchronization counters."""

    selected: int
    embedded: int
    skipped: int
    inserted: int
    vector_updated: int
    payload_updated: int
    vector_reused: int
    failures: tuple[dict[str, str], ...]

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def upserted(self) -> int:
        return self.inserted + self.vector_updated

    @property
    def indexed(self) -> int:
        return self.inserted + self.vector_updated + self.payload_updated + self.skipped


def _select_chunks(chunks: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return chunks if limit is None else chunks[:limit]


def _payload_from_chunk(
    chunk: dict[str, Any],
    *,
    dataset_source_hash: str,
    dataset_manifest_hash: str,
    dataset_chunks_hash: str,
    preprocessing_schema_version: str,
    status_rules_version: str,
    embedding_model: str,
) -> dict[str, Any]:
    embedding_text = str(chunk.get("text_for_embedding") or "")
    content_hash = content_hash_for_text(embedding_text)
    payload = {
        field: chunk.get(field)
        for field in sorted(SOURCE_CHUNK_REQUIRED_FIELDS)
        if field != "text_for_embedding"
    }
    payload["content_hash"] = content_hash
    payload["text_for_embedding"] = embedding_text
    payload["dataset_source_hash"] = dataset_source_hash
    payload["dataset_manifest_hash"] = dataset_manifest_hash
    payload["dataset_chunks_hash"] = dataset_chunks_hash
    payload["preprocessing_schema_version"] = preprocessing_schema_version
    payload["status_rules_version"] = status_rules_version
    payload["embedding_model"] = embedding_model
    payload["payload_hash"] = payload_hash(payload)
    return payload


def prepare_points(
    chunks: Sequence[dict[str, Any]],
    *,
    dataset_source_hash: str,
    dataset_manifest_hash: str,
    dataset_chunks_hash: str,
    preprocessing_schema_version: str,
    status_rules_version: str,
    embedding_model: str,
) -> list[PreparedPoint]:
    """Build deterministic point payloads and hashes from clean chunks."""
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
        payload = _payload_from_chunk(
            chunk,
            dataset_source_hash=dataset_source_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            dataset_chunks_hash=dataset_chunks_hash,
            preprocessing_schema_version=preprocessing_schema_version,
            status_rules_version=status_rules_version,
            embedding_model=embedding_model,
        )
        points.append(
            PreparedPoint(
                chunk_id=chunk_id,
                point_id=point_id_from_chunk_id(chunk_id),
                embedding_text=embedding_text,
                payload=payload,
                content_hash=str(payload["content_hash"]),
                payload_hash=str(payload["payload_hash"]),
            )
        )
    return points


def _payload_profile(points: Sequence[PreparedPoint]) -> dict[str, Any]:
    total = len(points)
    field_summary: dict[str, dict[str, Any]] = {}
    distribution_counts: dict[str, dict[str, int]] = {
        field: {} for field in PROFILE_DISTRIBUTION_FIELDS
    }
    distribution_values: dict[str, dict[str, Any]] = {
        field: {} for field in PROFILE_DISTRIBUTION_FIELDS
    }
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
    for point in points:
        for field in PROFILE_DISTRIBUTION_FIELDS:
            value = point.payload.get(field)
            items = value if isinstance(value, list) else [value]
            for item in items:
                if item is None:
                    continue
                key = canonical_dumps(item)
                distribution_counts[field][key] = distribution_counts[field].get(key, 0) + 1
                distribution_values[field][key] = item
    distributions: dict[str, dict[str, Any]] = {}
    for field in PROFILE_DISTRIBUTION_FIELDS:
        ranked = sorted(distribution_counts[field].items(), key=lambda item: (-item[1], item[0]))
        bounded = ranked[:100]
        distributions[field] = {
            "distinct_count": len(ranked),
            "values": [
                {"value": distribution_values[field][key], "count": count}
                for key, count in bounded
            ],
            "truncated": len(ranked) > len(bounded),
        }
    return {
        "total_points_profiled": total,
        "required_fields": sorted(REQUIRED_PAYLOAD_FIELDS),
        "filterable_fields": list(FILTERABLE_FIELDS),
        "fields": field_summary,
        "distributions": distributions,
    }


def _distribution_signature(profile: dict[str, Any], field: str) -> dict[str, int]:
    distribution = (profile.get("distributions") or {}).get(field) or {}
    return {
        canonical_dumps(row.get("value")): int(row.get("count") or 0)
        for row in distribution.get("values") or []
    }


def _pipeline_identity() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    source_hashes = {
        path.name: sha256_file(path)
        for path in sorted(source_root.glob("*.py"), key=lambda item: item.name)
    }
    identity: dict[str, Any] = {
        "source_dir": str(source_root),
        "source_files": source_hashes,
        "source_files_hash": sha256_text(canonical_dumps(source_hashes)),
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        repository_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        identity["repository_root"] = repository_root
        identity["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        identity["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return identity


def _resolve_source_dir(manifest: dict[str, Any], dataset_dir: Path) -> Path:
    candidates: list[Path] = []
    source_dir = manifest.get("source_dir")
    if isinstance(source_dir, str) and source_dir.strip():
        candidates.append(Path(source_dir))
    config_source_dir = (manifest.get("config") or {}).get("source_dir")
    if isinstance(config_source_dir, str) and config_source_dir.strip():
        candidates.append(Path(config_source_dir))
    candidates.append(dataset_dir.parent / "laws_html")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise RuntimeError(
        "Cannot recompute dataset source hash: source HTML directory was not found "
        f"(checked {[str(path) for path in candidates]})"
    )


def _recompute_source_hash(manifest: dict[str, Any], dataset_dir: Path) -> tuple[str, Path]:
    from legal_rag.laws_preprocessing.inventory import build_corpus_registry, compute_source_hash

    source_dir = _resolve_source_dir(manifest, dataset_dir)
    registry, _ = build_corpus_registry(source_dir)
    return compute_source_hash(list(registry.by_law_id.values())), source_dir


def _validate_vector_reuse_manifest(
    config: IndexingConfig,
    *,
    vector_size: int,
) -> dict[str, Any]:
    manifest_path = config.resolved_reuse_vectors_manifest_path
    source_index_dir = config.resolved_reuse_vectors_index_dir
    collection_name = config.reuse_vectors_collection
    if manifest_path is None or source_index_dir is None or collection_name is None:
        return {"enabled": False}
    if not manifest_path.is_file():
        raise RuntimeError(f"Vector reuse manifest does not exist: {manifest_path}")
    source_manifest = read_json(manifest_path)
    embedding = source_manifest.get("embedding") or {}
    qdrant = source_manifest.get("qdrant") or {}
    expected_model = config.resolved_embedding_model
    checks = {
        "ready_for_retrieval": source_manifest.get("ready_for_retrieval") is True,
        "collection_name": source_manifest.get("collection_name") == collection_name,
        "embedding_model": embedding.get("model") == expected_model,
        "resolved_embedding_model": embedding.get("resolved_model") == expected_model,
        "vector_size": embedding.get("vector_size") == vector_size,
        "hybrid_enabled": embedding.get("hybrid_enabled") is config.hybrid_enabled,
        "local_mode": qdrant.get("mode") in {"local", "local_path"} and not qdrant.get("url"),
        "index_path": (
            isinstance(qdrant.get("path"), str)
            and Path(qdrant["path"]).resolve() == source_index_dir
        ),
        "run_id": bool(source_manifest.get("run_id")),
        "chunks_hash": bool(source_manifest.get("chunks_hash")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Vector reuse manifest is incompatible with the target run: "
            + ", ".join(failed)
        )
    return {
        "enabled": True,
        "mode": "local",
        "index_dir": str(source_index_dir),
        "collection_name": collection_name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "run_id": source_manifest["run_id"],
        "chunks_hash": source_manifest["chunks_hash"],
        "embedding_model": expected_model,
        "vector_size": vector_size,
        "hybrid_enabled": config.hybrid_enabled,
        "manifest_checks": checks,
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
    reuse_client: QdrantClient | None = None,
    reuse_collection: str | None = None,
    vector_size: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SyncStats:
    existing_hashes: dict[str, dict[str, str]] = {}
    if not config.force_rebuild:
        existing_hashes = fetch_existing_hashes(
            client,
            collection_name=collection_name,
            point_ids=[point.point_id for point in points],
        )

    to_embed: list[PreparedPoint] = []
    payload_only: list[PreparedPoint] = []
    skipped = 0
    for point in points:
        existing = existing_hashes.get(point.point_id)
        if (
            existing
            and existing["content_hash"] == point.content_hash
            and existing["payload_hash"] == point.payload_hash
        ):
            skipped += 1
        elif existing and existing["content_hash"] == point.content_hash:
            payload_only.append(point)
        else:
            to_embed.append(point)

    failures: list[dict[str, str]] = []
    embedded = 0
    inserted = 0
    vector_updated = 0
    payload_updated = 0
    vector_reused = 0
    started = perf_counter()
    total_to_embed = len(to_embed)
    if progress_callback:
        progress_callback(
            {
                "event": "sync_started",
                "selected": len(points),
                "skipped": skipped,
                "to_embed": total_to_embed,
                "payload_only": len(payload_only),
                "vector_reuse_enabled": reuse_client is not None,
                "batch_size": config.batch_size,
            }
        )

    for point in payload_only:
        try:
            set_point_payload(
                client,
                collection_name=collection_name,
                point=point,
                max_retries=config.upload_max_retries,
            )
            payload_updated += 1
        except Exception as exc:
            failures.append({"chunk_id": point.chunk_id, "stage": "payload_update", "error": str(exc)})

    def record_vector_write(point: PreparedPoint) -> None:
        nonlocal inserted, vector_updated
        if point.point_id in existing_hashes:
            vector_updated += 1
        else:
            inserted += 1

    def emit_batch_progress(*, batch_number: int, batch_total: int, batch_size: int, batch_started: float) -> None:
        if not progress_callback:
            return
        elapsed = max(perf_counter() - started, 0.001)
        processed = inserted + vector_updated + payload_updated + skipped + len(failures)
        completed_vectors = embedded + vector_reused
        rate = completed_vectors / elapsed if completed_vectors else 0.0
        remaining = max(len(points) - processed, 0)
        progress_callback(
            {
                "event": "batch_finished",
                "batch": batch_number,
                "batch_total": batch_total,
                "batch_size": batch_size,
                "embedded": embedded,
                "inserted": inserted,
                "vector_updated": vector_updated,
                "payload_updated": payload_updated,
                "vector_reused": vector_reused,
                "upserted": inserted + vector_updated,
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

    for start in range(0, len(to_embed), config.batch_size):
        batch = to_embed[start : start + config.batch_size]
        batch_started = perf_counter()
        batch_number = (start // config.batch_size) + 1
        batch_total = ((total_to_embed + config.batch_size - 1) // config.batch_size) if total_to_embed else 0
        if reuse_client is not None:
            if not reuse_collection or vector_size is None:
                raise RuntimeError("Vector reuse source is missing collection or vector size")
            reusable = fetch_reusable_vectors(
                reuse_client,
                collection_name=reuse_collection,
                points=batch,
                embedding_model=config.resolved_embedding_model,
                vector_size=vector_size,
                hybrid_enabled=config.hybrid_enabled,
            )
            reusable_points = [point for point in batch if point.point_id in reusable]
            if reusable_points:
                dense_vectors = [reusable[point.point_id].dense for point in reusable_points]
                sparse_vectors: list[tuple[list[int], list[float]]] | None = None
                if config.hybrid_enabled:
                    sparse_vectors = []
                    for point in reusable_points:
                        sparse = reusable[point.point_id].sparse
                        if sparse is None:
                            raise RuntimeError(f"Reusable point is missing sparse vector: {point.chunk_id}")
                        sparse_vectors.append(sparse)
                try:
                    upload_point_batch(
                        client,
                        collection_name=collection_name,
                        points=reusable_points,
                        vectors=dense_vectors,
                        sparse_vectors=sparse_vectors,
                        max_retries=config.upload_max_retries,
                        upload_batch_size=config.upload_batch_size,
                        upload_parallel=config.qdrant_upload_parallel,
                    )
                    for point in reusable_points:
                        record_vector_write(point)
                    vector_reused += len(reusable_points)
                except Exception:
                    for index, point in enumerate(reusable_points):
                        try:
                            sparse_vector = (
                                [sparse_vectors[index]]
                                if sparse_vectors is not None
                                else None
                            )
                            upload_point_batch(
                                client,
                                collection_name=collection_name,
                                points=[point],
                                vectors=[dense_vectors[index]],
                                sparse_vectors=sparse_vector,
                                max_retries=config.upload_max_retries,
                                upload_batch_size=config.upload_batch_size,
                                upload_parallel=config.qdrant_upload_parallel,
                            )
                            record_vector_write(point)
                            vector_reused += 1
                        except Exception as exc:
                            failures.append(
                                {
                                    "chunk_id": point.chunk_id,
                                    "stage": "vector_reuse_upsert",
                                    "error": str(exc),
                                }
                            )
                batch = [point for point in batch if point.point_id not in reusable]
        if not batch:
            emit_batch_progress(
                batch_number=batch_number,
                batch_total=batch_total,
                batch_size=0,
                batch_started=batch_started,
            )
            continue
        try:
            batch_texts = [point.embedding_text for point in batch]
            sparse_vectors = None
            if config.hybrid_enabled:
                if not supports_sparse_embedding(embedder):
                    raise RuntimeError("Hybrid indexing requires an embedder with embed_sparse_texts().")
                if supports_hybrid_embedding(embedder):
                    vectors, sparse_vectors = embedder.embed_dense_and_sparse_texts(batch_texts)  # type: ignore[attr-defined]
                else:
                    vectors = embedder.embed_texts(batch_texts)
                    sparse_vectors = embedder.embed_sparse_texts(batch_texts)  # type: ignore[attr-defined]
            else:
                vectors = embedder.embed_texts(batch_texts)
            if len(vectors) != len(batch):
                raise RuntimeError(f"Embedding vector count mismatch: got {len(vectors)}, expected {len(batch)}")
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
                    sparse_vector = None
                    if config.hybrid_enabled:
                        if not supports_sparse_embedding(embedder):
                            raise RuntimeError("Hybrid indexing requires an embedder with embed_sparse_texts().")
                        if supports_hybrid_embedding(embedder):
                            vector, sparse_vector = embedder.embed_dense_and_sparse_texts([point.embedding_text])  # type: ignore[attr-defined]
                        else:
                            vector = embedder.embed_texts([point.embedding_text])
                            sparse_vector = embedder.embed_sparse_texts([point.embedding_text])  # type: ignore[attr-defined]
                    else:
                        vector = embedder.embed_texts([point.embedding_text])
                    upload_point_batch(
                        client,
                        collection_name=collection_name,
                        points=[point],
                        vectors=vector,
                        sparse_vectors=sparse_vector,
                        max_retries=config.upload_max_retries,
                        upload_batch_size=config.upload_batch_size,
                        upload_parallel=config.qdrant_upload_parallel,
                    )
                    embedded += 1
                    record_vector_write(point)
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
                    upload_batch_size=config.upload_batch_size,
                    upload_parallel=config.qdrant_upload_parallel,
                )
                for point in point_batch:
                    record_vector_write(point)
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
                            upload_batch_size=config.upload_batch_size,
                            upload_parallel=config.qdrant_upload_parallel,
                        )
                        record_vector_write(point)
                    except Exception as inner_exc:
                        failures.append({"chunk_id": point.chunk_id, "stage": "upsert", "error": str(inner_exc)})

        emit_batch_progress(
            batch_number=batch_number,
            batch_total=batch_total,
            batch_size=len(batch),
            batch_started=batch_started,
        )

    return SyncStats(
        selected=len(points),
        embedded=embedded,
        skipped=skipped,
        inserted=inserted,
        vector_updated=vector_updated,
        payload_updated=payload_updated,
        vector_reused=vector_reused,
        failures=tuple(failures),
    )


def _write_quality_report(path: Path, *, manifest: dict[str, Any]) -> None:
    gates = manifest["quality_gates"]
    lines = [
        "# 03 - Indexing Contract Quality Report",
        "",
        f"- Ready for retrieval: **{manifest['ready_for_retrieval']}**",
        f"- Collection: `{manifest['collection_name']}`",
        f"- Indexed count: {manifest['indexed_count']}",
        f"- Inserted: {manifest['inserted_count']}",
        f"- Vector updated: {manifest['vector_updated_count']}",
        f"- Payload updated: {manifest['payload_updated_count']}",
        f"- Vector reused: {manifest['vector_reused_count']}",
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
    reuse = manifest["vector_reuse"]
    if reuse["enabled"]:
        lines.extend(
            [
                "",
                "## Vector Reuse Source",
                f"- collection: `{reuse['collection_name']}`",
                f"- run id: `{reuse['run_id']}`",
                f"- manifest: `{reuse['manifest_path']}`",
                f"- manifest sha256: `{reuse['manifest_sha256']}`",
                f"- source chunks sha256: `{reuse['chunks_hash']}`",
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
    reuse_client: QdrantClient | None = None
    reuse_provenance: dict[str, Any] = {"enabled": False}

    try:
        validation = validate_clean_dataset(cfg.resolved_dataset_dir, strict=cfg.strict)
        write_json(tmp_dir / "dataset_validation.json", validation.to_dict())
        if not validation.ok:
            raise RuntimeError("Dataset validation failed: " + "; ".join(validation.errors))

        manifest = read_manifest(cfg.resolved_dataset_dir)
        chunks = _select_chunks(load_chunks(cfg.resolved_dataset_dir), cfg.selection_limit)
        declared_source_hash = str(manifest.get("source_hash") or "")
        source_hash, source_dir = _recompute_source_hash(manifest, cfg.resolved_dataset_dir)
        if not declared_source_hash or source_hash != declared_source_hash:
            raise RuntimeError(
                "Source corpus hash mismatch: "
                f"manifest={declared_source_hash!r}, actual={source_hash!r}"
            )
        chunks_hash = str(validation.actual_output_hashes["chunks"])
        manifest_hash = str(validation.manifest_hash or "")
        preprocessing_schema_version = str(manifest.get("schema_version") or "")
        status_rules_version = str(manifest.get("status_rules_version") or "")
        pipeline_identity = _pipeline_identity()
        points = prepare_points(
            chunks,
            dataset_source_hash=source_hash,
            dataset_manifest_hash=manifest_hash,
            dataset_chunks_hash=chunks_hash,
            preprocessing_schema_version=preprocessing_schema_version,
            status_rules_version=status_rules_version,
            embedding_model=cfg.resolved_embedding_model,
        )
        if not points:
            raise RuntimeError("No chunks selected for indexing")
        if progress_callback:
            progress_callback(
                {
                    "event": "dataset_ready",
                    "selected": len(points),
                    "chunk_selection_mode": cfg.chunk_selection_mode,
                    "sample_size": cfg.sample_size,
                    "collection_name": cfg.collection_name,
                }
            )

        if embedder is None:
            embedder = build_embedder(cfg)
        if progress_callback:
            progress_callback(
                {
                    "event": "embedder_ready",
                    "backend": cfg.embedding_backend,
                    "model": getattr(embedder, "model_name", cfg.resolved_embedding_model),
                    "hybrid_enabled": cfg.hybrid_enabled,
                }
            )
        probe = embedder.embed_texts([points[0].embedding_text])
        vector_size = len(probe[0])
        if vector_size <= 0:
            raise RuntimeError("Embedding probe returned an empty vector")
        if cfg.embedding_dim is not None and cfg.embedding_dim != vector_size:
            raise RuntimeError(f"Embedding dim mismatch: configured={cfg.embedding_dim}, detected={vector_size}")
        if cfg.hybrid_enabled and not supports_sparse_embedding(embedder):
            raise RuntimeError("hybrid_enabled=True requires an embedding backend that exposes sparse vectors")
        if progress_callback:
            progress_callback({"event": "embedding_probe_finished", "vector_size": vector_size})

        reuse_provenance = _validate_vector_reuse_manifest(cfg, vector_size=vector_size)
        if reuse_provenance["enabled"]:
            reuse_index_dir = cfg.resolved_reuse_vectors_index_dir
            assert reuse_index_dir is not None
            assert cfg.reuse_vectors_collection is not None
            reuse_client = QdrantClient(path=str(reuse_index_dir))
            reuse_provenance["collection_topology"] = validate_vector_reuse_collection(
                reuse_client,
                collection_name=cfg.reuse_vectors_collection,
                vector_size=vector_size,
                hybrid_enabled=cfg.hybrid_enabled,
            )

        if client is None:
            client = connect_qdrant(cfg)
        collection_name = build_collection_name(cfg, dataset_hash=source_hash)
        created_collection, removed_count, payload_index_statuses = ensure_collection(
            client,
            cfg,
            collection_name=collection_name,
            vector_size=vector_size,
        )
        if progress_callback:
            progress_callback(
                {
                    "event": "collection_ready",
                    "collection_name": collection_name,
                    "created": created_collection,
                    "removed_count": removed_count,
                }
            )
        sync_stats = _sync_points(
            client,
            collection_name=collection_name,
            points=points,
            embedder=embedder,
            config=cfg,
            reuse_client=reuse_client,
            reuse_collection=cfg.reuse_vectors_collection,
            vector_size=vector_size,
            progress_callback=progress_callback,
        )
        indexing_threshold_restored = restore_collection_indexing_threshold(client, cfg, collection_name=collection_name)
        if progress_callback and indexing_threshold_restored:
            progress_callback(
                {
                    "event": "optimizer_indexing_threshold_restored",
                    "collection_name": collection_name,
                    "indexing_threshold_kb": cfg.qdrant_restore_indexing_threshold_kb,
                }
            )

        failures = list(sync_stats.failures)
        write_jsonl(tmp_dir / "failures.jsonl", failures)

        point_count = collection_point_count(client, collection_name=collection_name)
        duplicate_check = validate_no_duplicate_chunk_ids(client, collection_name=collection_name)
        sample = next((point for point in points if "current" in (point.payload.get("index_views") or [])), points[0])
        filter_validation = _validate_filtered_query(client, collection_name=collection_name, sample=sample)
        payload_profile = _payload_profile(points)
        collection_payload_profile = profile_collection_payload(
            client,
            collection_name=collection_name,
            required_fields=sorted(REQUIRED_PAYLOAD_FIELDS),
        )
        payload_profile["collection"] = collection_payload_profile

        expected_identity = {
            "dataset_source_hash": source_hash,
            "dataset_manifest_hash": manifest_hash,
            "dataset_chunks_hash": chunks_hash,
            "preprocessing_schema_version": preprocessing_schema_version,
            "status_rules_version": status_rules_version,
        }
        identity_validation: dict[str, bool] = {}
        for field in IDENTITY_PAYLOAD_FIELDS:
            distribution = collection_payload_profile["distributions"][field]
            values = distribution["values"]
            identity_validation[field] = (
                distribution["distinct_count"] == 1
                and len(values) == 1
                and values[0]["value"] == expected_identity[field]
                and values[0]["count"] == point_count
            )
        reconciliation_fields = (
            "law_status",
            "article_status",
            "passage_status",
            "content_availability",
            "index_views",
            "relation_types",
        )
        distribution_reconciliation = {
            field: (
                not payload_profile["distributions"][field]["truncated"]
                and not collection_payload_profile["distributions"][field]["truncated"]
                and _distribution_signature(payload_profile, field)
                == _distribution_signature(collection_payload_profile, field)
            )
            for field in reconciliation_fields
        }
        payload_profile["identity_validation"] = identity_validation
        payload_profile["distribution_reconciliation"] = distribution_reconciliation

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

        indexed_count = sync_stats.indexed
        count_gate = point_count == len(points)
        gates = {
            "dataset_ready_for_indexing": validation.ok,
            "dataset_hashes_verified": bool(validation.actual_output_hashes),
            "source_corpus_hash_verified": source_hash == declared_source_hash,
            "selected_chunks_non_empty": len(points) > 0,
            "embedding_model_recorded": bool(cfg.embedding_model),
            "vector_size_detected": vector_size > 0,
            "vector_reuse_source_compatible": (
                not reuse_provenance["enabled"]
                or all(reuse_provenance["manifest_checks"].values())
            ),
            "indexed_count_matches_selected": indexed_count == len(points) and sync_stats.failure_count == 0,
            "collection_count_matches_selected": count_gate,
            "duplicate_chunk_ids_rejected": bool(duplicate_check.get("ok")),
            "filter_validation_queryable": bool(filter_validation.get("ok")),
            "payload_indexes_requested": all(not str(status).startswith("error:") for status in payload_index_statuses.values()),
            "required_payload_fields_present": all(
                field["missing"] == 0
                for field in collection_payload_profile["required_field_coverage"].values()
            ),
            "collection_identity_single_valued": all(identity_validation.values()),
            "status_distributions_reconciled": all(distribution_reconciliation.values()),
            "diagnostic_queries_return_hits": any(row.get("hits") for row in retrieval_rows),
            "worktree_clean_when_required": (
                not cfg.require_clean_worktree or pipeline_identity.get("git_dirty") is False
            ),
        }
        restore_indexing_threshold_kb = (
            cfg.qdrant_restore_indexing_threshold_kb
            if cfg.qdrant_bulk_indexing_threshold_kb is not None
            else None
        )
        index_manifest = {
            "schema_version": INDEXING_SCHEMA_VERSION,
            "created_at": now_utc(),
            "run_id": run_id,
            "config": cfg.public_dict(),
            "source_dataset_dir": str(cfg.resolved_dataset_dir),
            "source_corpus_dir": str(source_dir),
            "source_hash": source_hash,
            "source_output_hashes": validation.actual_output_hashes,
            "declared_source_output_hashes": manifest.get("output_hashes", {}),
            "chunks_hash": chunks_hash,
            "dataset_manifest_hash": manifest_hash,
            "preprocessing_schema_version": preprocessing_schema_version,
            "status_rules_version": status_rules_version,
            "source_identity": {
                "dataset_source_hash": source_hash,
                "dataset_manifest_hash": manifest_hash,
                "dataset_chunks_hash": chunks_hash,
                "preprocessing_schema_version": preprocessing_schema_version,
                "status_rules_version": status_rules_version,
                "actual_output_hashes": validation.actual_output_hashes,
            },
            "pipeline_identity": pipeline_identity,
            "collection_name": collection_name,
            "collection_identity": {
                "collection_name": collection_name,
                **expected_identity,
            },
            "collection_created": created_collection,
            "qdrant": {
                "mode": "server" if cfg.qdrant_url else "local",
                "url": cfg.qdrant_url,
                "path": str(cfg.resolved_index_dir),
                "dense_vector_name": "dense",
                "sparse_vector_name": "sparse" if cfg.hybrid_enabled else None,
                "distance": cfg.qdrant_distance,
                "on_disk_payload": cfg.qdrant_on_disk_payload,
                "shard_number": cfg.qdrant_shard_number,
                "hnsw_m": cfg.qdrant_hnsw_m,
                "hnsw_ef_construct": cfg.qdrant_hnsw_ef_construct,
                "upload_batch_size": cfg.upload_batch_size,
                "upload_parallel": cfg.qdrant_upload_parallel,
                "bulk_indexing_threshold_kb": cfg.qdrant_bulk_indexing_threshold_kb,
                "restore_indexing_threshold_kb": restore_indexing_threshold_kb,
                "indexing_threshold_restored": indexing_threshold_restored,
            },
            "embedding": {
                "backend": cfg.embedding_backend,
                "model": getattr(embedder, "model_name", cfg.resolved_embedding_model),
                "configured_model": cfg.embedding_model,
                "resolved_model": cfg.resolved_embedding_model,
                "hybrid_enabled": cfg.hybrid_enabled,
                "vector_size": vector_size,
            },
            "vector_reuse": {
                **reuse_provenance,
                "vector_reused_count": sync_stats.vector_reused,
            },
            "selected_count": len(points),
            "indexed_count": indexed_count,
            "embedded_count": sync_stats.embedded,
            "vector_reused_count": sync_stats.vector_reused,
            "skipped_count": sync_stats.skipped,
            "upserted_count": sync_stats.upserted,
            "inserted_count": sync_stats.inserted,
            "vector_updated_count": sync_stats.vector_updated,
            "payload_updated_count": sync_stats.payload_updated,
            "removed_count": removed_count,
            "failure_count": sync_stats.failure_count,
            "collection_points_count": point_count,
            "payload_indexes": list(FILTERABLE_FIELDS),
            "payload_index_statuses": payload_index_statuses,
            "payload_field_summary": payload_profile["fields"],
            "payload_distributions": collection_payload_profile["distributions"],
            "identity_validation": identity_validation,
            "distribution_reconciliation": distribution_reconciliation,
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
        if progress_callback:
            progress_callback(
                {
                    "event": "run_finished",
                    "run_id": run_id,
                    "collection_name": collection_name,
                    "ready_for_retrieval": index_manifest["ready_for_retrieval"],
                    "indexed_count": indexed_count,
                    "vector_reused_count": sync_stats.vector_reused,
                    "collection_points_count": point_count,
                    "failure_count": sync_stats.failure_count,
                }
            )
        return index_manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    finally:
        if reuse_client is not None:
            reuse_client.close()
        if created_client and client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency `qdrant_client`. Activate project venv first "
        "(for example: `source .venv/bin/activate`) and retry."
    ) from exc


DEFAULT_INDEX_FIELDS: dict[str, models.PayloadSchemaType] = {
    "law_id": models.PayloadSchemaType.KEYWORD,
    "article_id": models.PayloadSchemaType.KEYWORD,
    "law_status": models.PayloadSchemaType.KEYWORD,
    "index_views": models.PayloadSchemaType.KEYWORD,
    "relation_types": models.PayloadSchemaType.KEYWORD,
    "related_law_ids": models.PayloadSchemaType.KEYWORD,
    "law_date": models.PayloadSchemaType.DATETIME,
}


@dataclass(frozen=True)
class MigrationStats:
    collection_name: str
    migrated_points: int
    payload_indexes_created: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate Qdrant collections from local embedded storage to a remote/server Qdrant "
            "(for example Docker on http://127.0.0.1:6333)."
        )
    )
    parser.add_argument(
        "--local-path",
        default="data/indexes/qdrant",
        help="Path to embedded/local Qdrant storage (default: data/indexes/qdrant).",
    )
    parser.add_argument(
        "--remote-url",
        default="http://127.0.0.1:6333",
        help="Remote/server Qdrant URL (default: http://127.0.0.1:6333).",
    )
    parser.add_argument(
        "--remote-api-key",
        default=os.getenv("QDRANT_API_KEY", ""),
        help="Remote/server Qdrant API key (optional; default from QDRANT_API_KEY env).",
    )
    parser.add_argument(
        "--collection",
        action="append",
        default=[],
        help="Collection to migrate. Repeat to migrate multiple. If omitted, migrate all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Scroll/upsert batch size (default: 512).",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="If collection already exists on remote, delete and recreate it.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip payload index creation on remote.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing to remote.",
    )
    return parser.parse_args()


def _resolve_collections(
    src: QdrantClient,
    requested_collections: list[str],
) -> list[str]:
    local_collections = [c.name for c in src.get_collections().collections]
    if requested_collections:
        unknown = sorted(set(requested_collections) - set(local_collections))
        if unknown:
            raise RuntimeError(f"Requested collections not found in local storage: {unknown}")
        return sorted(dict.fromkeys(requested_collections))
    return sorted(local_collections)


def _create_collection_from_source(
    *,
    src: QdrantClient,
    dst: QdrantClient,
    collection_name: str,
    dry_run: bool,
    drop_existing: bool,
) -> None:
    info = src.get_collection(collection_name)
    exists = dst.collection_exists(collection_name)

    if exists and not drop_existing:
        raise RuntimeError(
            f"Collection {collection_name!r} already exists on remote. "
            "Use --drop-existing to overwrite it."
        )
    if dry_run:
        return
    if exists and drop_existing:
        dst.delete_collection(collection_name)

    dst.create_collection(
        collection_name=collection_name,
        vectors_config=info.config.params.vectors,
        sparse_vectors_config=info.config.params.sparse_vectors,
        on_disk_payload=info.config.params.on_disk_payload,
    )


def _copy_points(
    *,
    src: QdrantClient,
    dst: QdrantClient,
    collection_name: str,
    batch_size: int,
    dry_run: bool,
) -> int:
    total = 0
    offset: Any = None
    while True:
        points, offset = src.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        total += len(points)
        if not dry_run:
            batch = [
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ]
            dst.upsert(collection_name=collection_name, points=batch, wait=True)

        if offset is None:
            break
    return total


def _ensure_payload_indexes(
    *,
    dst: QdrantClient,
    collection_name: str,
    dry_run: bool,
) -> int:
    created = 0
    for field_name, field_schema in DEFAULT_INDEX_FIELDS.items():
        if dry_run:
            created += 1
            continue
        try:
            dst.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
            created += 1
        except Exception:
            # Idempotent behavior: skip fields already indexed or unavailable in payload.
            continue
    return created


def _migrate_collection(
    *,
    src: QdrantClient,
    dst: QdrantClient,
    collection_name: str,
    batch_size: int,
    dry_run: bool,
    drop_existing: bool,
    skip_indexes: bool,
) -> MigrationStats:
    _create_collection_from_source(
        src=src,
        dst=dst,
        collection_name=collection_name,
        dry_run=dry_run,
        drop_existing=drop_existing,
    )
    migrated_points = _copy_points(
        src=src,
        dst=dst,
        collection_name=collection_name,
        batch_size=batch_size,
        dry_run=dry_run,
    )
    payload_indexes_created = 0
    if not skip_indexes:
        payload_indexes_created = _ensure_payload_indexes(
            dst=dst,
            collection_name=collection_name,
            dry_run=dry_run,
        )
    return MigrationStats(
        collection_name=collection_name,
        migrated_points=migrated_points,
        payload_indexes_created=payload_indexes_created,
    )


def main() -> int:
    args = _parse_args()
    if int(args.batch_size) <= 0:
        raise RuntimeError("--batch-size must be > 0")

    src = QdrantClient(path=str(args.local_path))
    dst = QdrantClient(
        url=str(args.remote_url),
        api_key=(str(args.remote_api_key).strip() or None),
    )
    collections = _resolve_collections(src, list(args.collection))
    if not collections:
        print("No collections found in local source.")
        return 0

    print(f"Source local path: {args.local_path}")
    print(f"Destination URL: {args.remote_url}")
    print(f"Collections to migrate: {collections}")
    print(f"Dry run: {bool(args.dry_run)}")

    stats: list[MigrationStats] = []
    for name in collections:
        print(f"\nMigrating {name} ...")
        result = _migrate_collection(
            src=src,
            dst=dst,
            collection_name=name,
            batch_size=int(args.batch_size),
            dry_run=bool(args.dry_run),
            drop_existing=bool(args.drop_existing),
            skip_indexes=bool(args.skip_indexes),
        )
        stats.append(result)
        print(
            f"  migrated_points={result.migrated_points} "
            f"payload_indexes_created={result.payload_indexes_created}"
        )

    remote_collections = [c.name for c in dst.get_collections().collections]
    print("\nRemote collections:")
    print(remote_collections)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def canonical_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def content_hash_for_text(text: str) -> str:
    return sha256_text((text or "").strip())


def payload_hash(payload: dict[str, Any], *, exclude_keys: tuple[str, ...] = ("payload_hash",)) -> str:
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if key in exclude_keys:
            continue
        clean[key] = value
    return sha256_text(canonical_dumps(clean))


def point_id_from_chunk_id(chunk_id: str) -> str:
    # UUIDv5 keeps deterministic point ids across re-runs.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


__all__ = [
    "canonical_dumps",
    "sha256_text",
    "content_hash_for_text",
    "payload_hash",
    "point_id_from_chunk_id",
]

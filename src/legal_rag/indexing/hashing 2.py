"""Stable hashing helpers for Qdrant point identity and change detection."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Serialize JSON-like data deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def content_hash_for_text(text: str) -> str:
    """Hash the exact embedding input used to detect unchanged chunks."""
    return sha256_text((text or "").strip())


def payload_hash(payload: dict[str, Any], *, exclude_keys: tuple[str, ...] = ("payload_hash",)) -> str:
    """Hash a payload after excluding self-referential fields."""
    clean = {key: value for key, value in payload.items() if key not in exclude_keys}
    return sha256_text(canonical_dumps(clean))


def point_id_from_chunk_id(chunk_id: str) -> str:
    """Return a deterministic UUID string for one chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))

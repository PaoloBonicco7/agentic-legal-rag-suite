"""Helpers used by `notebooks/03_indexing_contract.ipynb` to keep it lean.

These functions wrap Docker setup, progress logging, run inspection and dataset
statistics so the notebook can stay focused on configuration, execution and
visualization.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient


def ensure_qdrant_running(
    *,
    compose_file: Path,
    project_root: Path,
    url: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Start the Qdrant Docker container (idempotent) and wait for HTTP readiness."""
    if not compose_file.exists():
        raise FileNotFoundError(f"Docker compose file not found: {compose_file}")
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d", "qdrant"],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    )
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = QdrantClient(url=url, timeout=5)
        try:
            client.get_collections()
            return {
                "ok": True,
                "url": url,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(1)
        finally:
            client.close()
    raise RuntimeError(f"Qdrant did not become ready at {url}: {last_error}")


def qdrant_collection_state(url: str, collection_name: str) -> dict[str, Any]:
    """Return existence and exact point count for a collection on a Qdrant server."""
    client = QdrantClient(url=url)
    try:
        exists = bool(client.collection_exists(collection_name))
        count = (
            int(client.count(collection_name=collection_name, exact=True).count)
            if exists
            else None
        )
        return {"collection": collection_name, "exists": exists, "points": count}
    except Exception as exc:
        return {
            "collection": collection_name,
            "exists": False,
            "points": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        client.close()


def make_progress_callback(log_path: Path) -> Callable[[dict[str, Any]], None]:
    """Return a callback that prints concise progress lines and appends JSONL events.

    The log is truncated on first call so a re-run does not append to a stale file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    def callback(event: dict[str, Any]) -> None:
        row = {"logged_at": datetime.now(timezone.utc).isoformat(), **event}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        name = event.get("event", "")
        if name == "sync_started":
            print(
                f"Indexing started: selected={event['selected']}, "
                f"skipped={event['skipped']}, to_process={event['to_process']}, "
                f"batch_size={event['batch_size']}",
                flush=True,
            )
        elif name == "batch_finished":
            eta = event.get("eta_seconds")
            eta_text = f" eta={eta:.0f}s" if isinstance(eta, (int, float)) else ""
            print(
                f"batch {event['batch']}/{event['batch_total']} - "
                f"{event['processed']}/{event['selected']} ({event['percent']}%) - "
                f"rate={event['rate_chunks_per_second']}/s{eta_text}",
                flush=True,
            )
        elif name in {
            "dataset_ready",
            "embedder_ready",
            "embedding_probe_finished",
            "collection_ready",
            "run_finished",
        }:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    return callback


def read_progress_log(path: Path) -> list[dict[str, Any]]:
    """Read progress JSONL events as a list of dicts."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_manifest_path(
    *,
    runs_dir: Path,
    run_id: str,
    collection_name: str,
    model: str | None = None,
) -> Path | None:
    """Locate the manifest for a run, falling back to the most recent matching run."""
    expected = runs_dir / run_id / "index_manifest.json"
    if expected.exists():
        return expected
    candidates = sorted(
        runs_dir.glob("*/index_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("collection_name") != collection_name:
            continue
        embedding = data.get("embedding") or {}
        if model and embedding.get("resolved_model") != model:
            continue
        return candidate
    return None


def dataset_distribution(chunks_path: Path, *, top_relations: int = 10) -> dict[str, Any]:
    """Compute lightweight distributions over `chunks.jsonl` for the notebook charts."""
    chunks_per_law: Counter[str] = Counter()
    law_status: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    chunk_lengths: list[int] = []

    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks_per_law[str(chunk.get("law_id") or "")] += 1
            law_status[str(chunk.get("law_status") or "unknown")] += 1
            for relation in chunk.get("relation_types") or []:
                relation_types[str(relation)] += 1
            chunk_lengths.append(len(str(chunk.get("text") or "")))

    return {
        "chunks_per_law": list(chunks_per_law.values()),
        "law_status": dict(law_status),
        "chunk_lengths": chunk_lengths,
        "relation_types": dict(relation_types.most_common(top_relations)),
        "total_chunks": sum(chunks_per_law.values()),
        "total_laws": len(chunks_per_law),
    }

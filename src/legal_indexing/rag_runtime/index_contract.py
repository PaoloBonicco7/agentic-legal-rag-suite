from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .config import RagRuntimeConfig


@dataclass(frozen=True)
class IndexContract:
    source: str
    collection_name: str
    qdrant_path: Path
    run_id: str | None
    indexing_summary_path: Path | None
    indexing_summary: dict[str, Any] | None
    dense_vector_size: int | None = None
    sparse_enabled: bool = False
    sparse_vector_name: str | None = None
    sparse_artifacts_path: Path | None = None
    eval_reference_coverage: float | None = None
    missing_references_sample: tuple[str, ...] = tuple()
    payload_field_coverage: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "collection_name": self.collection_name,
            "qdrant_path": str(self.qdrant_path),
            "run_id": self.run_id,
            "resolved_run_id": self.run_id,
            "indexing_summary_path": (
                str(self.indexing_summary_path) if self.indexing_summary_path else None
            ),
            "dense_vector_size": self.dense_vector_size,
            "sparse_enabled": self.sparse_enabled,
            "sparse_vector_name": self.sparse_vector_name,
            "sparse_artifacts_path": (
                str(self.sparse_artifacts_path) if self.sparse_artifacts_path else None
            ),
            "eval_reference_coverage": self.eval_reference_coverage,
            "missing_references_sample": list(self.missing_references_sample),
            "payload_field_coverage": (
                dict(self.payload_field_coverage) if self.payload_field_coverage else None
            ),
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_TS_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def _run_ids_with_summary(artifacts_root: Path) -> list[str]:
    if not artifacts_root.exists():
        return []
    out: list[Path] = []
    for child in artifacts_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "indexing_summary.json").exists():
            out.append(child)
    return sorted([p.name for p in out])


def _is_timestamp_run_id(run_id: str) -> bool:
    return bool(_TS_RUN_ID_RE.match((run_id or "").strip()))


def _resolve_default_run_id(artifacts_root: Path, run_ids: list[str]) -> str:
    """
    Pick the default run_id with a robust strategy:
    1) prefer canonical timestamp-like run_ids (YYYYMMDD_HHMMSS), newest by lexical order
    2) otherwise fallback to most recently modified run directory
    """
    ts_ids = [rid for rid in run_ids if _is_timestamp_run_id(rid)]
    if ts_ids:
        return sorted(ts_ids)[-1]

    candidates = [artifacts_root / rid for rid in run_ids]
    newest = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    return newest.name


def _collection_name_from_summary(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        name = summary.get("collection_name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    collection = payload.get("collection")
    if isinstance(collection, dict):
        name = collection.get("collection_name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    raise RuntimeError(
        "indexing_summary.json does not contain a valid collection_name in `summary` "
        "or `collection` blocks."
    )


def _qdrant_path_from_summary(payload: dict[str, Any], config: RagRuntimeConfig) -> Path:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        raw = summary.get("qdrant_path")
        if isinstance(raw, str) and raw.strip():
            return config.resolve_path(Path(raw))
    return config.resolved_qdrant_path


def _hybrid_block(payload: dict[str, Any]) -> dict[str, Any]:
    block = payload.get("hybrid_index")
    if isinstance(block, dict):
        return block
    return {}


def _index_contract_block(payload: dict[str, Any]) -> dict[str, Any]:
    block = payload.get("index_contract")
    if isinstance(block, dict):
        return block
    return {}


def resolve_index_contract(config: RagRuntimeConfig) -> IndexContract:
    config.validate()

    if config.collection_name and config.collection_name.strip():
        return IndexContract(
            source="explicit",
            collection_name=config.collection_name.strip(),
            qdrant_path=config.resolved_qdrant_path,
            run_id=None,
            indexing_summary_path=None,
            indexing_summary=None,
            dense_vector_size=None,
            sparse_enabled=False,
            sparse_vector_name=None,
            sparse_artifacts_path=None,
        )

    artifacts_root = config.resolved_indexing_artifacts_root
    run_ids = _run_ids_with_summary(artifacts_root)
    if not run_ids:
        raise RuntimeError(
            "Cannot resolve collection_name: no indexing runs found under "
            f"{artifacts_root}. Set RagRuntimeConfig.collection_name explicitly or run "
            "`notebooks/04_qdrant_indexing_pipeline.ipynb` first."
        )

    run_id = (
        config.indexing_run_id.strip()
        if config.indexing_run_id
        else _resolve_default_run_id(artifacts_root, run_ids)
    )
    if run_id not in run_ids:
        raise RuntimeError(
            f"indexing_run_id={run_id!r} not found under {artifacts_root}. "
            f"Available runs: {', '.join(run_ids)}"
        )

    summary_path = artifacts_root / run_id / "indexing_summary.json"
    payload = _read_json(summary_path)
    collection_name = _collection_name_from_summary(payload)
    qdrant_path = _qdrant_path_from_summary(payload, config)
    hybrid = _hybrid_block(payload)
    index_contract = _index_contract_block(payload)
    sparse_artifacts_path: Path | None = None
    sparse_path_raw = hybrid.get("sparse_artifact_path")
    if isinstance(sparse_path_raw, str) and sparse_path_raw.strip():
        sparse_artifacts_path = config.resolve_path(Path(sparse_path_raw))

    return IndexContract(
        source="artifact",
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        run_id=run_id,
        indexing_summary_path=summary_path,
        indexing_summary=payload,
        dense_vector_size=(
            int(hybrid["dense_vector_size"])
            if hybrid.get("dense_vector_size") is not None
            else None
        ),
        sparse_enabled=bool(hybrid.get("sparse_enabled")),
        sparse_vector_name=(
            str(hybrid.get("sparse_vector_name")).strip()
            if hybrid.get("sparse_vector_name") is not None
            else None
        ),
        sparse_artifacts_path=sparse_artifacts_path,
        eval_reference_coverage=(
            float(index_contract["eval_reference_coverage"])
            if index_contract.get("eval_reference_coverage") is not None
            else None
        ),
        missing_references_sample=tuple(
            [
                str(x).strip()
                for x in (index_contract.get("missing_references_sample") or [])
                if str(x).strip()
            ]
        ),
        payload_field_coverage=(
            {
                str(k): float(v)
                for k, v in (index_contract.get("payload_field_coverage") or {}).items()
            }
            if isinstance(index_contract.get("payload_field_coverage"), dict)
            else None
        ),
    )

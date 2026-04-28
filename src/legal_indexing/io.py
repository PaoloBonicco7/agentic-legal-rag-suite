from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


_REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "passage_id",
    "article_id",
    "law_id",
    "chunk_seq",
    "text",
    "text_for_embedding",
    "law_status",
    "article_is_abrogated",
    "passage_label",
    "related_law_ids",
    "relation_types",
    "inbound_law_ids",
    "outbound_law_ids",
    "index_views",
)

_REQUIRED_LIST_FIELDS: tuple[str, ...] = (
    "related_law_ids",
    "relation_types",
    "inbound_law_ids",
    "outbound_law_ids",
    "index_views",
)

_ARTICLE_SUFFIX_ORDER: dict[str, int] = {
    "": 0,
    "bis": 1,
    "ter": 2,
    "quater": 3,
    "quinquies": 4,
    "sexies": 5,
    "septies": 6,
    "octies": 7,
    "novies": 8,
    "decies": 9,
    "undecies": 10,
    "duodecies": 11,
    "terdecies": 12,
    "quaterdecies": 13,
    "quinquiesdecies": 14,
}


@dataclass(frozen=True)
class DatasetValidationResult:
    dataset_dir: Path
    manifest_path: Path
    required_files: dict[str, bool]
    counts: dict[str, int]
    missing_chunk_fields: dict[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PassageRecord:
    passage_id: str
    article_id: str
    law_id: str
    passage_label: str
    text: str
    source_chunk_ids: tuple[str, ...]

    law_date: str | None
    law_number: int | None
    law_title: str | None

    law_status: str
    article_is_abrogated: bool
    index_views: tuple[str, ...]

    related_law_ids: tuple[str, ...]
    relation_types: tuple[str, ...]
    inbound_law_ids: tuple[str, ...]
    outbound_law_ids: tuple[str, ...]

    status_confidence: float | None
    status_evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DatasetBundle:
    manifest: dict[str, Any]
    articles: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    passages: list[PassageRecord]
    article_order_by_id: dict[str, int]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_hash_from_manifest(manifest: dict[str, Any]) -> str:
    hashes = manifest.get("hashes")
    if isinstance(hashes, dict):
        val = hashes.get("chunks")
        if isinstance(val, str) and val.strip():
            return val.strip()
    run_id = str(manifest.get("run_id") or "")
    if run_id:
        return run_id
    created_at = str(manifest.get("created_at") or "")
    if created_at:
        return created_at
    return "unknown_dataset"


def validate_dataset(dataset_dir: Path, *, strict: bool = True) -> DatasetValidationResult:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    required_files = {
        "manifest": manifest_path.exists(),
        "laws": (dataset_dir / "laws.jsonl").exists(),
        "articles": (dataset_dir / "articles.jsonl").exists(),
        "notes": (dataset_dir / "notes.jsonl").exists(),
        "edges": (dataset_dir / "edges.jsonl").exists(),
        "events": (dataset_dir / "events.jsonl").exists(),
        "chunks": (dataset_dir / "chunks.jsonl").exists(),
    }

    errors: list[str] = []
    warnings: list[str] = []

    for name, ok in required_files.items():
        if not ok:
            errors.append(f"Missing required dataset file: {name}")

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ready = bool(manifest.get("ready_to_embedding"))
        if not ready:
            msg = "Manifest flag ready_to_embedding is false"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
    else:
        errors.append(f"Manifest not found: {manifest_path}")

    counts: dict[str, int] = {}
    for name in ("laws", "articles", "notes", "edges", "events", "chunks"):
        p = dataset_dir / f"{name}.jsonl"
        if not p.exists():
            counts[name] = 0
            continue
        counts[name] = _count_jsonl(p)

    missing_chunk_fields: dict[str, int] = {field: 0 for field in _REQUIRED_CHUNK_FIELDS}
    chunks_path = dataset_dir / "chunks.jsonl"
    if chunks_path.exists():
        for rec in iter_jsonl(chunks_path):
            for field in _REQUIRED_CHUNK_FIELDS:
                if field in _REQUIRED_LIST_FIELDS:
                    value = rec.get(field)
                    if value is None or not isinstance(value, list):
                        missing_chunk_fields[field] += 1
                    continue

                value = rec.get(field)
                if value is None:
                    missing_chunk_fields[field] += 1
                    continue
                if isinstance(value, str) and not value.strip():
                    missing_chunk_fields[field] += 1
    for field, n_missing in missing_chunk_fields.items():
        if n_missing > 0:
            errors.append(f"chunks.jsonl has {n_missing} rows with missing field {field!r}")

    return DatasetValidationResult(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        required_files=required_files,
        counts=counts,
        missing_chunk_fields=missing_chunk_fields,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def load_articles(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "articles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing articles file: {path}")
    return list(iter_jsonl(path))


def load_chunks(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}")
    return list(iter_jsonl(path))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _safe_sorted_unique(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return tuple()
    clean: set[str] = set()
    for item in values:
        s = str(item).strip()
        if s:
            clean.add(s)
    return tuple(sorted(clean))


def _merge_word_sequences(texts: list[str], *, max_overlap_window: int = 160) -> str:
    if not texts:
        return ""
    merged = (texts[0] or "").split()
    for t in texts[1:]:
        cur = (t or "").split()
        if not cur:
            continue
        max_k = min(len(merged), len(cur), max_overlap_window)
        overlap = 0
        for k in range(max_k, 0, -1):
            if merged[-k:] == cur[:k]:
                overlap = k
                break
        merged.extend(cur[overlap:])
    return " ".join(merged).strip()


def reconstruct_passages(chunks: Iterable[dict[str, Any]]) -> list[PassageRecord]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in chunks:
        pid = str(rec.get("passage_id") or "").strip()
        if not pid:
            continue
        grouped[pid].append(rec)

    passages: list[PassageRecord] = []
    for passage_id, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: int(r.get("chunk_seq") or 0))
        text = _merge_word_sequences([str(r.get("text") or "") for r in rows_sorted])
        first = rows_sorted[0]
        status_evidence = first.get("status_evidence")
        evidence_tuple: tuple[dict[str, Any], ...]
        if isinstance(status_evidence, list):
            evidence_tuple = tuple(x for x in status_evidence if isinstance(x, dict))
        else:
            evidence_tuple = tuple()

        passages.append(
            PassageRecord(
                passage_id=passage_id,
                article_id=str(first.get("article_id") or ""),
                law_id=str(first.get("law_id") or ""),
                passage_label=str(first.get("passage_label") or ""),
                text=text,
                source_chunk_ids=tuple(str(r.get("chunk_id") or "") for r in rows_sorted),
                law_date=(str(first.get("law_date")) if first.get("law_date") is not None else None),
                law_number=_as_int(first.get("law_number")),
                law_title=(str(first.get("law_title")) if first.get("law_title") is not None else None),
                law_status=str(first.get("law_status") or "unknown"),
                article_is_abrogated=_as_bool(first.get("article_is_abrogated")),
                index_views=_safe_sorted_unique(first.get("index_views")),
                related_law_ids=_safe_sorted_unique(first.get("related_law_ids")),
                relation_types=_safe_sorted_unique(first.get("relation_types")),
                inbound_law_ids=_safe_sorted_unique(first.get("inbound_law_ids")),
                outbound_law_ids=_safe_sorted_unique(first.get("outbound_law_ids")),
                status_confidence=(
                    float(first.get("status_confidence"))
                    if first.get("status_confidence") is not None
                    else None
                ),
                status_evidence=evidence_tuple,
            )
        )

    passages.sort(key=lambda p: p.passage_id)
    return passages


def _article_label_from_article_id(article_id: str) -> str:
    if "#art:" in article_id:
        return article_id.split("#art:", 1)[1]
    return article_id


def article_sort_key(article_label: str) -> tuple[int, int, int, str, str]:
    lbl = (article_label or "").strip().lower()
    if lbl == "unico":
        return (0, 0, 0, "", "")
    m = re.fullmatch(r"(?P<num>\d+)(?P<suf>[a-z]*)", lbl)
    if not m:
        return (1, 0, 0, lbl, lbl)
    num = int(m.group("num"))
    suf = m.group("suf") or ""
    suf_rank = _ARTICLE_SUFFIX_ORDER.get(suf, 1000)
    return (0, num, suf_rank, suf, lbl)


def build_article_order_map(
    articles: Iterable[dict[str, Any]], passages: Iterable[PassageRecord]
) -> dict[str, int]:
    by_law: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for rec in articles:
        article_id = str(rec.get("article_id") or "").strip()
        law_id = str(rec.get("law_id") or "").strip()
        if not article_id or not law_id:
            continue
        label = str(rec.get("article_label_norm") or _article_label_from_article_id(article_id)).strip()
        by_law[law_id].append((article_id, label))

    for p in passages:
        if not p.article_id or not p.law_id:
            continue
        existing = by_law.setdefault(p.law_id, [])
        known = {aid for aid, _ in existing}
        if p.article_id in known:
            continue
        existing.append((p.article_id, _article_label_from_article_id(p.article_id)))

    out: dict[str, int] = {}
    for law_id, items in by_law.items():
        items_sorted = sorted(
            items,
            key=lambda t: (
                article_sort_key(t[1]),
                t[0],
            ),
        )
        for order, (article_id, _) in enumerate(items_sorted):
            out[article_id] = order

    return out


def load_dataset_bundle(dataset_dir: Path) -> DatasetBundle:
    dataset_dir = dataset_dir.resolve()
    manifest = read_manifest(dataset_dir)
    articles = load_articles(dataset_dir)
    chunks = load_chunks(dataset_dir)
    passages = reconstruct_passages(chunks)
    article_order_by_id = build_article_order_map(articles, passages)
    return DatasetBundle(
        manifest=manifest,
        articles=articles,
        chunks=chunks,
        passages=passages,
        article_order_by_id=article_order_by_id,
    )


__all__ = [
    "DatasetValidationResult",
    "PassageRecord",
    "DatasetBundle",
    "iter_jsonl",
    "read_manifest",
    "dataset_hash_from_manifest",
    "validate_dataset",
    "load_articles",
    "load_chunks",
    "reconstruct_passages",
    "article_sort_key",
    "build_article_order_map",
    "load_dataset_bundle",
]

"""Dataset export orchestration for phase 1 laws preprocessing."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ingest import ingest_law
from .inventory import build_corpus_registry, compute_source_hash, sha256_file
from .models import (
    ALLOWED_LAW_STATUSES,
    ALLOWED_RELATION_TYPES,
    LIST_CHUNK_FIELDS,
    REQUIRED_CHUNK_FIELDS,
    SCHEMA_VERSION,
    LawsPreprocessingConfig,
    chunk_record,
)


def validate_output_dir(source_dir: Path, output_dir: Path) -> None:
    """Reject output paths that could overwrite source data or the project root."""
    source = source_dir.resolve()
    output = output_dir.resolve()
    cwd = Path.cwd().resolve()
    if output == source:
        raise ValueError(f"Output directory must not be the source corpus directory: {output}")
    if output.is_relative_to(source):
        raise ValueError(f"Output directory must not be inside the source corpus directory: {output}")
    if source.is_relative_to(output):
        raise ValueError(f"Output directory must not contain the source corpus directory: {output}")
    if output == cwd:
        raise ValueError(f"Output directory must not be the project/current directory: {output}")
    if output == Path(output.anchor):
        raise ValueError(f"Output directory must not be a filesystem root: {output}")


def _deduplicate(records: list[dict[str, Any]], id_field: str) -> tuple[list[dict[str, Any]], int]:
    """Drop duplicate records deterministically after sorting by stable ID."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    duplicates = 0
    for record in sorted(records, key=lambda item: str(item.get(id_field) or "")):
        record_id = str(record.get(id_field) or "")
        if not record_id:
            out.append(record)
            continue
        if record_id in seen:
            duplicates += 1
            continue
        seen.add(record_id)
        out.append(record)
    return out, duplicates


def _enrich_chunks_with_graph_fields(chunks: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach law-level inbound/outbound graph metadata to every chunk."""
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)
    law_relation_types: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        src = str(edge.get("src_law_id") or "")
        dst = str(edge.get("dst_law_id") or "")
        if not src or not dst:
            continue
        outbound[src].add(dst)
        inbound[dst].add(src)
        law_relation_types[src].add(str(edge.get("relation_type") or ""))
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        record = dict(chunk)
        law_id = str(record.get("law_id") or "")
        record["inbound_law_ids"] = sorted(inbound.get(law_id, set()))
        record["outbound_law_ids"] = sorted(outbound.get(law_id, set()))
        record["relation_types"] = sorted(set(record.get("relation_types") or []) | law_relation_types.get(law_id, set()))
        out.append(record)
    return sorted(out, key=lambda item: str(item.get("chunk_id") or ""))


def _validate_unique(records: list[dict[str, Any]], id_field: str) -> bool:
    """Return whether all records have non-empty unique IDs."""
    ids = [str(record.get(id_field) or "") for record in records]
    return all(ids) and len(ids) == len(set(ids))


def _build_quality(
    *,
    inventory: dict[str, Any],
    laws: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    unresolved_refs: int,
    output_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute counts, diagnostics and quality gates for the generated dataset."""
    chunk_missing_fields = sum(1 for chunk in chunks if not REQUIRED_CHUNK_FIELDS.issubset(chunk.keys()))
    list_metadata_errors = sum(
        1 for chunk in chunks for field in LIST_CHUNK_FIELDS if not isinstance(chunk.get(field), list)
    )
    relation_type_errors = sum(
        1 for edge in edges if str(edge.get("relation_type") or "") not in ALLOWED_RELATION_TYPES
    )
    status_errors = sum(1 for law in laws if str(law.get("law_status") or "") not in ALLOWED_LAW_STATUSES)
    gates = {
        "valid_source_html_found": inventory.get("valid_html_files", 0) > 0,
        "stable_ids_non_empty_duplicate_free": all(
            [
                _validate_unique(laws, "law_id"),
                _validate_unique(articles, "article_id"),
                _validate_unique(passages, "passage_id"),
                _validate_unique(notes, "note_id") if notes else True,
                _validate_unique(edges, "edge_id") if edges else True,
                _validate_unique(chunks, "chunk_id"),
            ]
        ),
        "required_chunk_fields_present": chunk_missing_fields == 0,
        "list_metadata_fields_are_lists": list_metadata_errors == 0,
        "clean_graph_edges_have_no_self_loops": all(edge.get("src_law_id") != edge.get("dst_law_id") for edge in edges),
        "relation_types_allowed": relation_type_errors == 0,
        "law_statuses_allowed": status_errors == 0,
        "manifest_output_files_exist_and_hash": bool(output_hashes) and all(output_hashes.values()),
        "chunks_jsonl_non_empty": len(chunks) > 0,
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "counts": {
            "laws": len(laws),
            "articles": len(articles),
            "passages": len(passages),
            "notes": len(notes),
            "edges": len(edges),
            "chunks": len(chunks),
        },
        "ignored_files": inventory.get("ignored_files", []),
        "unresolved_refs": unresolved_refs,
        "status_distribution": dict(Counter(str(law.get("law_status") or "") for law in laws)),
        "relation_type_distribution": dict(Counter(str(edge.get("relation_type") or "") for edge in edges)),
        "chunk_missing_fields": chunk_missing_fields,
        "list_metadata_errors": list_metadata_errors,
        "relation_type_errors": relation_type_errors,
        "status_errors": status_errors,
        "quality_gates": gates,
        "ready_for_indexing": all(gates.values()),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable, sorted JSON for manifest-like artifacts."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(records: list[dict[str, Any]], path: Path, id_field: str) -> None:
    """Write JSONL records sorted by their stable ID."""
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: str(item.get(id_field) or "")):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_quality_report(path: Path, quality: dict[str, Any]) -> None:
    """Write a human-readable summary of the quality gates."""
    lines = [
        "# 01 - Laws Preprocessing Quality Report",
        "",
        f"- Ready for indexing: **{quality['ready_for_indexing']}**",
        f"- Generated at UTC: `{quality['created_at']}`",
        "",
        "## Quality Gates",
    ]
    for gate, ok in quality["quality_gates"].items():
        lines.append(f"- `{gate}`: **{ok}**")
    lines.extend(
        [
            "",
            "## Counts",
            f"- laws: {quality['counts']['laws']}",
            f"- articles: {quality['counts']['articles']}",
            f"- passages: {quality['counts']['passages']}",
            f"- notes: {quality['counts']['notes']}",
            f"- edges: {quality['counts']['edges']}",
            f"- chunks: {quality['counts']['chunks']}",
            "",
            "## Diagnostics",
            f"- ignored files: {len(quality['ignored_files'])}",
            f"- unresolved references: {quality['unresolved_refs']}",
            f"- chunk missing field errors: {quality['chunk_missing_fields']}",
            f"- list metadata errors: {quality['list_metadata_errors']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_dataset_profile(
    *,
    inventory: dict[str, Any],
    laws: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact profile with counts and representative records."""
    return {
        "counts": quality["counts"],
        "ignored_files": inventory.get("ignored_files", []),
        "status_distribution": quality["status_distribution"],
        "relation_type_distribution": quality["relation_type_distribution"],
        "sample_laws": laws[:3],
        "sample_articles": articles[:3],
        "sample_passages": passages[:3],
        "sample_notes": notes[:3],
        "sample_edges": edges[:3],
        "sample_chunks": chunks[:3],
        "ready_for_indexing": quality["ready_for_indexing"],
    }


def _prepare_output_dir(output_dir: Path) -> Path:
    """Create a fresh sibling temporary directory for atomic output writes."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    return tmp_dir


def _replace_output_dir(tmp_dir: Path, output_dir: Path) -> None:
    """Replace the final output directory only after validation succeeds."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)


def run_laws_preprocessing(config: LawsPreprocessingConfig | None = None) -> dict[str, Any]:
    """Run the complete deterministic HTML-to-clean-dataset pipeline."""
    cfg = config or LawsPreprocessingConfig()
    if not isinstance(cfg, LawsPreprocessingConfig):
        cfg = LawsPreprocessingConfig.model_validate(cfg)
    source_dir = Path(cfg.source_dir)
    output_dir = Path(cfg.output_dir)
    validate_output_dir(source_dir, output_dir)

    registry, inventory = build_corpus_registry(source_dir)
    selected_law_files = [registry.by_law_id[law_id] for law_id in sorted(registry.by_law_id)]
    source_hash = compute_source_hash(selected_law_files)

    laws: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    warnings: list[str] = []
    unresolved_refs = 0

    for law_file in selected_law_files:
        ingested = ingest_law(
            law_file,
            registry,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            strict=cfg.strict,
        )
        laws.append(ingested.law)
        articles.extend(ingested.articles)
        passages.extend(ingested.passages)
        notes.extend(ingested.notes)
        edges.extend(ingested.edges)
        chunks.extend(ingested.chunks)
        warnings.extend(ingested.warnings)
        unresolved_refs += ingested.unresolved_refs

    laws, dropped_laws = _deduplicate(laws, "law_id")
    articles, dropped_articles = _deduplicate(articles, "article_id")
    passages, dropped_passages = _deduplicate(passages, "passage_id")
    notes, dropped_notes = _deduplicate(notes, "note_id")
    edges, dropped_edges = _deduplicate([edge for edge in edges if edge.get("src_law_id") != edge.get("dst_law_id")], "edge_id")
    chunks, dropped_chunks = _deduplicate(chunks, "chunk_id")
    chunks = [chunk_record(chunk) for chunk in _enrich_chunks_with_graph_fields(chunks, edges)]

    output_files = {
        "laws": "laws.jsonl",
        "articles": "articles.jsonl",
        "passages": "passages.jsonl",
        "notes": "notes.jsonl",
        "edges": "edges.jsonl",
        "chunks": "chunks.jsonl",
        "quality_report": "quality_report.md",
        "dataset_profile": "dataset_profile.json",
    }
    tmp_dir = _prepare_output_dir(output_dir)
    try:
        _write_jsonl(laws, tmp_dir / output_files["laws"], "law_id")
        _write_jsonl(articles, tmp_dir / output_files["articles"], "article_id")
        _write_jsonl(passages, tmp_dir / output_files["passages"], "passage_id")
        _write_jsonl(notes, tmp_dir / output_files["notes"], "note_id")
        _write_jsonl(edges, tmp_dir / output_files["edges"], "edge_id")
        _write_jsonl(chunks, tmp_dir / output_files["chunks"], "chunk_id")

        output_hashes = {
            name: sha256_file(tmp_dir / filename)
            for name, filename in output_files.items()
            if filename.endswith(".jsonl")
        }
        quality = _build_quality(
            inventory=inventory,
            laws=laws,
            articles=articles,
            passages=passages,
            notes=notes,
            edges=edges,
            chunks=chunks,
            unresolved_refs=unresolved_refs,
            output_hashes=output_hashes,
        )
        _write_quality_report(tmp_dir / output_files["quality_report"], quality)
        output_hashes["quality_report"] = sha256_file(tmp_dir / output_files["quality_report"])
        profile = _build_dataset_profile(
            inventory=inventory,
            laws=laws,
            articles=articles,
            passages=passages,
            notes=notes,
            edges=edges,
            chunks=chunks,
            quality=quality,
        )
        _write_json(tmp_dir / output_files["dataset_profile"], profile)
        output_hashes["dataset_profile"] = sha256_file(tmp_dir / output_files["dataset_profile"])
        quality = _build_quality(
            inventory=inventory,
            laws=laws,
            articles=articles,
            passages=passages,
            notes=notes,
            edges=edges,
            chunks=chunks,
            unresolved_refs=unresolved_refs,
            output_hashes=output_hashes,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "source_hash": source_hash,
            "config": cfg.model_dump(mode="json"),
            "inventory": inventory,
            "counts": quality["counts"],
            "dropped_duplicates": {
                "laws": dropped_laws,
                "articles": dropped_articles,
                "passages": dropped_passages,
                "notes": dropped_notes,
                "edges": dropped_edges,
                "chunks": dropped_chunks,
            },
            "warnings_sample": warnings[:100],
            "unresolved_refs": unresolved_refs,
            "quality_gates": quality["quality_gates"],
            "ready_for_indexing": quality["ready_for_indexing"],
            "outputs": output_files,
            "output_hashes": output_hashes,
            "manifest_hash_note": "manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
        }
        _write_json(tmp_dir / "manifest.json", manifest)

        output_hashes_with_manifest = dict(manifest["output_hashes"])
        final_quality = _build_quality(
            inventory=inventory,
            laws=laws,
            articles=articles,
            passages=passages,
            notes=notes,
            edges=edges,
            chunks=chunks,
            unresolved_refs=unresolved_refs,
            output_hashes=output_hashes_with_manifest,
        )
        if not final_quality["ready_for_indexing"]:
            raise ValueError(f"Laws preprocessing quality gates failed: {final_quality['quality_gates']}")
        _replace_output_dir(tmp_dir, output_dir)
        return manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

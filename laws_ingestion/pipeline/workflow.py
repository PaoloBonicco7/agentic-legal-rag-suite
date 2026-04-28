from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
import matplotlib.pyplot as plt
import pandas as pd

from laws_ingestion.ingest import ingest_law
from laws_ingestion.registry import build_corpus_registry

from .events import extract_events
from .relations import normalize_edges
from .reporting import build_quality_metrics, sha256_file, write_json, write_quality_markdown
from .scan import scan_inventory
from .status import classify_many_status
from .views import enrich_chunks_with_views


SCHEMA_VERSION = "laws-graph-pipeline-v1"


@dataclass(frozen=True)
class PipelineConfig:
    html_dir: str = "data/laws_html"
    output_dir: str = "data/laws_dataset_clean"
    run_root_dir: str = "notebooks/data/laws_graph_pipeline"
    seed: int = 42
    sample_size: int | None = None
    backend: str = "auto"
    strict: bool = False
    max_words: int = 600
    overlap_words: int = 80


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def _write_parquet_like(records: list[dict], path: Path) -> dict[str, Any]:
    df = _to_dataframe(records)
    mode = "parquet"
    err = None
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - env dependent
        # Conservative fallback for offline environments without pyarrow/fastparquet.
        mode = "jsonl_fallback"
        err = f"{type(exc).__name__}: {exc}"
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"path": str(path), "rows": len(records), "mode": mode, "error": err}


def _write_jsonl(records: list[dict], path: Path, id_field: str | None = None) -> None:
    rows = list(records)
    if id_field is not None:
        rows.sort(key=lambda x: str(x.get(id_field) or ""))
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _drop_duplicate_ids(records: list[dict], id_field: str) -> tuple[list[dict], int]:
    seen: set[str] = set()
    out: list[dict] = []
    dropped = 0
    for rec in records:
        rid = str(rec.get(id_field) or "")
        if not rid:
            out.append(rec)
            continue
        if rid in seen:
            dropped += 1
            continue
        seen.add(rid)
        out.append(rec)
    return out, dropped


def _law_kind(title: str) -> str:
    t = (title or "").lower()
    if "decreto legislativo" in t:
        return "decreto legislativo"
    if "decreto del presidente" in t or "d.p.r" in t:
        return "dpr"
    if "legge regionale" in t:
        return "legge regionale"
    return "unknown"


def _baseline_unresolved(default: int = 0) -> int:
    baseline_manifest = Path("data/laws_dataset/manifest.json")
    if not baseline_manifest.exists():
        return default
    try:
        data = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    except Exception:
        return default
    return int(data.get("unresolved_refs") or default)


def _step_report(run_dir: Path, step_name: str, payload: dict[str, Any]) -> None:
    write_json(run_dir / f"{step_name}_report.json", payload)


def _save_series_table_and_plot(run_dir: Path, stem: str, series: pd.Series, title: str) -> dict[str, str]:
    csv_path = run_dir / f"{stem}.csv"
    png_path = run_dir / f"{stem}.png"
    s = series.sort_values(ascending=False)
    s.to_csv(csv_path, header=["count"])
    fig = plt.figure(figsize=(10, 4))
    s.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)
    return {"table_csv": str(csv_path), "figure_png": str(png_path)}


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    html_dir = _resolve(config.html_dir)
    output_dir = _resolve(config.output_dir)
    run_root_dir = _resolve(config.run_root_dir)

    run_id = _now_run_id()
    run_dir = run_root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = asdict(config)
    run_cfg["run_id"] = run_id
    run_cfg["html_dir"] = str(html_dir)
    run_cfg["output_dir"] = str(output_dir)
    run_cfg["run_root_dir"] = str(run_root_dir)
    write_json(run_dir / "run_config.json", run_cfg)

    registry = build_corpus_registry(html_dir)
    law_ids = sorted(registry.by_law_id.keys())
    if config.sample_size is not None and config.sample_size > 0:
        rng = random.Random(int(config.seed))
        sampled = law_ids[:]
        rng.shuffle(sampled)
        law_ids = sorted(sampled[: int(config.sample_size)])

    run_manifest_init = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "discovered_files": len(registry.by_law_id),
        "selected_law_ids": len(law_ids),
        "html_dir": str(html_dir),
    }
    write_json(run_dir / "run_manifest_init.json", run_manifest_init)

    # Step 01
    inventory_rows, signal_counts = scan_inventory(html_dir)
    step01_meta = _write_parquet_like(inventory_rows, run_dir / "step01_inventory.parquet")
    write_json(run_dir / "step01_signal_counts.json", signal_counts)
    step01_vis = _save_series_table_and_plot(
        run_dir,
        "step01_signal_counts",
        pd.Series({k: v for k, v in signal_counts.items() if k != "n_files"}),
        "Frequenza segnali nel corpus",
    )
    _step_report(run_dir, "step01", {"inventory": step01_meta, "signal_counts": signal_counts, "visuals": step01_vis})

    # Step 02
    laws: list[dict] = []
    articles: list[dict] = []
    passages: list[dict] = []
    notes: list[dict] = []
    edges_ingest: list[dict] = []
    chunks: list[dict] = []
    warnings: list[str] = []
    unresolved_refs_new = 0

    for law_id in law_ids:
        law_file = registry.by_law_id[law_id]
        ing = ingest_law(
            law_file,
            registry,
            backend=config.backend,
            strict=bool(config.strict),
            max_words=int(config.max_words),
            overlap_words=int(config.overlap_words),
        )
        laws.append(dict(ing.law))
        articles.extend([dict(x) for x in ing.articles])
        passages.extend([dict(x) for x in ing.passages])
        notes.extend([dict(x) for x in ing.notes])
        edges_ingest.extend([dict(x) for x in ing.edges])
        chunks.extend([dict(x) for x in ing.chunks])
        warnings.extend(list(ing.warnings or []))
        unresolved_refs_new += int(ing.unresolved_refs or 0)

    articles, dropped_articles = _drop_duplicate_ids(articles, "article_id")
    passages, dropped_passages = _drop_duplicate_ids(passages, "passage_id")
    notes, dropped_notes = _drop_duplicate_ids(notes, "note_id")
    chunks, dropped_chunks = _drop_duplicate_ids(chunks, "chunk_id")

    step02_laws_meta = _write_parquet_like(laws, run_dir / "step02_laws.parquet")
    step02_articles_meta = _write_parquet_like(articles, run_dir / "step02_articles.parquet")
    step02_notes_meta = _write_parquet_like(notes, run_dir / "step02_notes.parquet")
    _step_report(
        run_dir,
        "step02",
        {
            "laws": step02_laws_meta,
            "articles": step02_articles_meta,
            "notes": step02_notes_meta,
            "dropped_duplicates": {
                "articles": dropped_articles,
                "passages": dropped_passages,
                "notes": dropped_notes,
                "chunks": dropped_chunks,
            },
            "warnings_sample": warnings[:50],
        },
    )

    # Step 03
    article_counts_by_law: dict[str, int] = {}
    for art in articles:
        lid = str(art.get("law_id") or "")
        article_counts_by_law[lid] = article_counts_by_law.get(lid, 0) + 1

    statuses = classify_many_status(laws, article_counts_by_law)
    status_by_law = {str(s["law_id"]): s for s in statuses}

    for law in laws:
        lid = str(law.get("law_id") or "")
        s = status_by_law.get(lid) or {}
        law["status"] = s.get("status") or law.get("status") or "unknown"
        law["status_confidence"] = float(s.get("status_confidence") or 0.0)
        law["status_evidence"] = s.get("status_evidence") or []
        law["law_kind"] = _law_kind(str(law.get("law_title") or ""))

    step03_records = [
        {
            "law_id": s["law_id"],
            "status": s["status"],
            "status_confidence": s["status_confidence"],
            "status_evidence": s["status_evidence"],
        }
        for s in statuses
    ]
    step03_meta = _write_parquet_like(step03_records, run_dir / "step03_law_status.parquet")
    step03_vis = _save_series_table_and_plot(
        run_dir,
        "step03_status_distribution",
        pd.Series([str(x.get("status") or "") for x in step03_records]).value_counts(),
        "Distribuzione status leggi",
    )
    _step_report(run_dir, "step03", {"law_status": step03_meta, "visuals": step03_vis})

    # Step 04
    edges_raw, edges_clean, relation_stats = normalize_edges(edges_ingest)
    step04_raw_meta = _write_parquet_like(edges_raw, run_dir / "step04_edges_raw.parquet")
    step04_clean_meta = _write_parquet_like(edges_clean, run_dir / "step04_edges_clean.parquet")
    step04_vis = _save_series_table_and_plot(
        run_dir,
        "step04_relation_types",
        pd.Series([str(e.get("relation_type") or "") for e in edges_clean]).value_counts(),
        "Distribuzione tipi relazione (clean)",
    )
    _step_report(
        run_dir,
        "step04",
        {
            "edges_raw": step04_raw_meta,
            "edges_clean": step04_clean_meta,
            "relation_stats": relation_stats,
            "visuals": step04_vis,
        },
    )

    # Step 05
    events = extract_events(edges_clean)
    step05_meta = _write_parquet_like(events, run_dir / "step05_events.parquet")
    _step_report(run_dir, "step05", {"events": step05_meta})

    # Step 06
    chunks_with_views = enrich_chunks_with_views(chunks, status_by_law, edges_clean)
    chunks_with_views, dropped_chunks_views = _drop_duplicate_ids(chunks_with_views, "chunk_id")
    step06_meta = _write_parquet_like(chunks_with_views, run_dir / "step06_chunks_with_views.parquet")
    _step_report(run_dir, "step06", {"chunks": step06_meta, "dropped_duplicate_chunks_after_views": dropped_chunks_views})

    # Step 07 exports
    paths = {
        "laws": output_dir / "laws.jsonl",
        "articles": output_dir / "articles.jsonl",
        "notes": output_dir / "notes.jsonl",
        "edges": output_dir / "edges.jsonl",
        "events": output_dir / "events.jsonl",
        "chunks": output_dir / "chunks.jsonl",
        "manifest": output_dir / "manifest.json",
    }

    _write_jsonl(laws, paths["laws"], id_field="law_id")
    _write_jsonl(articles, paths["articles"], id_field="article_id")
    _write_jsonl(notes, paths["notes"], id_field="note_id")
    _write_jsonl(edges_clean, paths["edges"], id_field="norm_edge_id")
    _write_jsonl(events, paths["events"], id_field="event_id")
    _write_jsonl(chunks_with_views, paths["chunks"], id_field="chunk_id")

    unresolved_refs_baseline = _baseline_unresolved(default=unresolved_refs_new)

    # Step 08 quality report
    metrics = build_quality_metrics(
        laws=laws,
        articles=articles,
        passages=passages,
        notes=notes,
        edges_raw=edges_raw,
        edges_clean=edges_clean,
        events=events,
        chunks=chunks_with_views,
        unresolved_refs_new=unresolved_refs_new,
        unresolved_refs_baseline=unresolved_refs_baseline,
    )
    write_quality_markdown(run_dir / "step08_quality_report.md", metrics)
    write_json(run_dir / "step08_quality_metrics.json", metrics)

    output_hashes = {name: sha256_file(path) for name, path in paths.items() if name != "manifest"}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_dir": str(html_dir),
        "output_dir": str(output_dir),
        "run_artifacts_dir": str(run_dir),
        "config": run_cfg,
        "counts": {
            "laws": len(laws),
            "articles": len(articles),
            "notes": len(notes),
            "edges": len(edges_clean),
            "events": len(events),
            "chunks": len(chunks_with_views),
        },
        "hashes": output_hashes,
        "unresolved_refs": {
            "baseline": unresolved_refs_baseline,
            "new": unresolved_refs_new,
            "delta": unresolved_refs_new - unresolved_refs_baseline,
        },
        "qa_gates": metrics.get("gates"),
        "ready_to_embedding": metrics.get("ready_to_embedding"),
        "parquet_writes": {
            "step01_inventory": step01_meta,
            "step02_laws": step02_laws_meta,
            "step02_articles": step02_articles_meta,
            "step02_notes": step02_notes_meta,
            "step03_law_status": step03_meta,
            "step04_edges_raw": step04_raw_meta,
            "step04_edges_clean": step04_clean_meta,
            "step05_events": step05_meta,
            "step06_chunks": step06_meta,
        },
        "visuals": {
            "step01": step01_vis,
            "step03": step03_vis,
            "step04": step04_vis,
        },
    }
    write_json(paths["manifest"], manifest)

    return manifest


__all__ = ["PipelineConfig", "run_pipeline", "SCHEMA_VERSION"]

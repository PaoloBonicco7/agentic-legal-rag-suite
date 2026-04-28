from __future__ import annotations

import json
from pathlib import Path

from laws_ingestion import PipelineConfig, run_pipeline


def _write_html(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _build_minimal_corpus(html_dir: Path) -> None:
    _write_html(
        html_dir / "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p>(Abrogata dall'art. 2 della <a href="/app/leggieregolamenti/dettaglio?tipo=L&amp;numero_legge=2%2F2001&amp;versione=V">L.R. 2 gennaio 2001, n. 2</a>)</p>
<p><a name="articolo_1__">Art. 1</a> - Oggetto</p>
<p>1. Testo articolo.</p>
</article>""",
    )
    _write_html(
        html_dir / "0002_LR-2-gennaio-2001-n2.html",
        """<article>
<h1>Legge regionale 2 gennaio 2001, n. 2 - Testo vigente</h1>
<p><a name="articolo_2__">Art. 2</a> - Disposizioni</p>
<p>1. Testo articolo 2.</p>
</article>""",
    )


def test_laws_graph_pipeline_smoke_minimal_corpus(tmp_path: Path) -> None:
    html_dir = tmp_path / "html"
    out_dir = tmp_path / "out"
    run_root = tmp_path / "runs"
    html_dir.mkdir(parents=True, exist_ok=True)
    _build_minimal_corpus(html_dir)

    manifest = run_pipeline(
        PipelineConfig(
            html_dir=str(html_dir),
            output_dir=str(out_dir),
            run_root_dir=str(run_root),
            backend="stdlib",
        )
    )

    assert manifest["schema_version"] == "laws-graph-pipeline-v1"
    assert manifest["counts"]["laws"] == 2
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "laws.jsonl").exists()
    assert (out_dir / "chunks.jsonl").exists()
    assert (Path(manifest["run_artifacts_dir"]) / "step08_quality_metrics.json").exists()
    assert "qa_gates" in manifest
    assert isinstance(manifest.get("ready_to_embedding"), bool)

    parquet_writes = manifest.get("parquet_writes") or {}
    assert parquet_writes, "parquet_writes metadata must be present in manifest"
    for write_meta in parquet_writes.values():
        assert write_meta.get("mode") in {"parquet", "jsonl_fallback"}


def test_laws_graph_pipeline_sample_size_limits_exported_laws(tmp_path: Path) -> None:
    html_dir = tmp_path / "html"
    out_dir = tmp_path / "out"
    run_root = tmp_path / "runs"
    html_dir.mkdir(parents=True, exist_ok=True)
    _build_minimal_corpus(html_dir)
    _write_html(
        html_dir / "0003_LR-3-gennaio-2002-n3.html",
        """<article>
<h1>Legge regionale 3 gennaio 2002, n. 3 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Testo.</p>
</article>""",
    )

    manifest = run_pipeline(
        PipelineConfig(
            html_dir=str(html_dir),
            output_dir=str(out_dir),
            run_root_dir=str(run_root),
            sample_size=1,
            seed=123,
            backend="stdlib",
        )
    )

    assert manifest["counts"]["laws"] == 1

    lines = [line for line in (out_dir / "laws.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    law = json.loads(lines[0])
    assert law.get("law_id")

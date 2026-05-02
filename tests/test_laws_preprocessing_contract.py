from __future__ import annotations

import json
from pathlib import Path

from legal_rag.laws_preprocessing import REQUIRED_CHUNK_FIELDS, LawsPreprocessingConfig, run_laws_preprocessing


def _write_law(root: Path, name: str, html: str) -> None:
    (root / name).write_text(html, encoding="utf-8")


def test_run_laws_preprocessing_exports_contract_files(tmp_path: Path) -> None:
    source = tmp_path / "laws_html"
    output = tmp_path / "laws_dataset_clean"
    source.mkdir()
    (source / ".DS_Store").write_text("ignored", encoding="utf-8")
    _write_law(
        source,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Richiama la Legge regionale 2 gennaio 2001, n. 2.</p>
</article>""",
    )
    _write_law(
        source,
        "0002_LR-2-gennaio-2001-n2.html",
        """<article>
<h1>Legge regionale 2 gennaio 2001, n. 2 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Testo.</p>
</article>""",
    )

    manifest = run_laws_preprocessing(
        LawsPreprocessingConfig(
            source_dir=str(source),
            output_dir=str(output),
            chunk_size=100,
            chunk_overlap=10,
        )
    )

    expected_files = {
        "manifest.json",
        "laws.jsonl",
        "articles.jsonl",
        "passages.jsonl",
        "notes.jsonl",
        "edges.jsonl",
        "chunks.jsonl",
        "quality_report.md",
        "dataset_profile.json",
    }
    assert expected_files == {path.name for path in output.iterdir()}
    assert manifest["ready_for_indexing"] is True
    assert manifest["inventory"]["ignored_files"] == [".DS_Store"]
    assert set(manifest["output_hashes"]) == {
        "laws",
        "articles",
        "passages",
        "notes",
        "edges",
        "chunks",
        "quality_report",
        "dataset_profile",
    }

    saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["ready_for_indexing"] is True
    chunks = [
        json.loads(line)
        for line in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks
    for chunk in chunks:
        assert REQUIRED_CHUNK_FIELDS.issubset(chunk)
        assert isinstance(chunk["index_views"], list)
        assert isinstance(chunk["related_law_ids"], list)
        assert isinstance(chunk["inbound_law_ids"], list)
        assert isinstance(chunk["outbound_law_ids"], list)
        assert isinstance(chunk["relation_types"], list)
    assert any("current" in chunk["index_views"] for chunk in chunks)

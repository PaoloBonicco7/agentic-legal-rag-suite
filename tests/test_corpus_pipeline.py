import json
from pathlib import Path

import pytest

from legal_rag.artifacts import file_hash, write_json, write_jsonl
from legal_rag.contracts import Block, Chunk, Document, SourceRecord, Unit
from legal_rag.corpus.chunking import build_chunks
from legal_rag.corpus.parser import span_text
from legal_rag.corpus.pipeline import build_corpus

from test_corpus_parser import FIXTURES, law, parse


class Counter:
    name = "test-whitespace"
    revision = "1"

    def count(self, text):
        return len(text.split())


def snapshot(tmp_path, names):
    path = tmp_path / "snapshot"
    path.mkdir()
    rows = []
    for name in names:
        target = path / name
        target.write_bytes((FIXTURES / name).read_bytes())
        rows.append(SourceRecord(source_id=name.split("_")[0], url="https://example.org/law", path=name, sha256=file_hash(target)))
    write_jsonl(path / "catalog.jsonl", rows)
    write_json(path / "manifest.json", {"snapshot_id": "test-snapshot", "complete": True,
               "catalog_sha256": file_hash(path / "catalog.jsonl"), "source_count": len(rows),
               "acquisition_finished_at": "2026-10-10T10:00:00Z"})
    return path


def test_real_corpus_subset_exports_valid_exact_artifacts(tmp_path):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html", "2065_LR-19-agosto-1992-n42.html", "1967_LR-5-settembre-1991-n44.html", "2711_LR-15-dicembre-2003-n21.html", "2144_LR-13-maggio-1993-n33.html"])
    output = tmp_path / "corpus"
    manifest = build_corpus(source, output, Counter(), 80, 120)
    assert manifest["ready_for_indexing"], manifest["quality"]
    assert manifest["counts"]["documents"] == 5
    assert all(file_hash(output / name) == value for name, value in manifest["files"].items())
    blocks = {b.block_id: b for b in [Block.model_validate_json(line) for line in (output / "blocks.jsonl").read_text().splitlines()]}
    chunks = [Chunk.model_validate_json(line) for line in (output / "chunks.jsonl").read_text().splitlines()]
    assert all(span_text(c.spans, blocks) == c.text for c in chunks)
    assert all(c.identity.startswith('Legge regionale ') and c.embedding_text.startswith(c.identity + '\n') for c in chunks)
    assert not any(c.document_id == "vda:lr:1950-10-10:1" for c in chunks)
    assert all(c.status == "ceased_confirmed" for c in chunks if c.document_id == "vda:lr:1991-09-05:44")
    assert all("Omissis" not in c.text for c in chunks)
    with pytest.raises(FileExistsError):
        build_corpus(source, output, Counter())


def test_source_hash_change_is_rejected_before_export(tmp_path):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html"])
    with (source / "273_LR-10-ottobre-1950-n1.html").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


def replace_catalog(source, rows):
    write_jsonl(source / "catalog.jsonl", rows)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["catalog_sha256"] = file_hash(source / "catalog.jsonl")
    write_json(source / "manifest.json", manifest)


def test_manipulated_catalog_is_rejected_by_source_manifest_hash(tmp_path):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html"])
    with (source / "catalog.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="catalog hash mismatch"):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


@pytest.mark.parametrize("complete", [False, None])
def test_missing_or_false_acquisition_completeness_is_rejected(tmp_path, complete):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html"])
    manifest = json.loads((source / "manifest.json").read_text())
    if complete is None:
        manifest.pop("complete")
    else:
        manifest["complete"] = complete
    write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete acquisition snapshot"):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


def test_reduced_catalog_is_rejected_even_with_updated_catalog_hash(tmp_path):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html", "2065_LR-19-agosto-1992-n42.html"])
    rows = [SourceRecord.model_validate_json(line) for line in (source / "catalog.jsonl").read_text().splitlines()]
    replace_catalog(source, rows[:1])
    with pytest.raises(ValueError, match="catalog count differs"):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


@pytest.mark.parametrize("field,error", [("source_id", "unique source IDs"), ("path", "paths must be unique")])
def test_duplicate_catalog_identifiers_or_paths_fail_preflight(tmp_path, field, error):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html", "2065_LR-19-agosto-1992-n42.html"])
    rows = [SourceRecord.model_validate_json(line) for line in (source / "catalog.jsonl").read_text().splitlines()]
    rows[1] = rows[1].model_copy(update={field: getattr(rows[0], field)})
    replace_catalog(source, rows)
    with pytest.raises(ValueError, match=error):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


@pytest.mark.parametrize("escape", ["relative", "absolute", "symlink"])
def test_source_path_escape_is_rejected_before_export(tmp_path, escape):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html"])
    outside = tmp_path / "outside.html"
    outside.write_text("<article>Outside the snapshot</article>")
    if escape == "symlink":
        (source / "linked.html").symlink_to(outside)
    path = "../outside.html" if escape == "relative" else str(outside) if escape == "absolute" else "linked.html"
    row = SourceRecord(source_id="outside", url="https://example.org/law", path=path, sha256=file_hash(outside))
    replace_catalog(source, [row])
    with pytest.raises(ValueError, match="Unsafe source path"):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


def test_all_raw_hashes_are_verified_before_any_source_is_parsed(tmp_path, monkeypatch):
    from legal_rag.corpus import pipeline

    source = snapshot(tmp_path, ["2065_LR-19-agosto-1992-n42.html", "273_LR-10-ottobre-1950-n1.html"])
    (source / "273_LR-10-ottobre-1950-n1.html").write_text("changed")

    def unexpected_parse(*args):
        raise AssertionError("Parsing started before raw preflight finished")

    monkeypatch.setattr(pipeline, "parse_source", unexpected_parse)
    with pytest.raises(ValueError, match="Source hash mismatch: 273"):
        build_corpus(source, tmp_path / "corpus", Counter())
    assert not (tmp_path / "corpus").exists()


def test_chunks_include_exact_intro_and_budget_complete_embedding_input():
    document, blocks, units = parse(law('<p>Art. 1</p><p>(Requisiti)</p><p>1. Sono ammesse le seguenti spese:</p>' + "".join(f"<p>{letter}) Acquisto di strumenti tecnologici per le attività regionali.</p>" for letter in "abcde")))
    chunks = build_chunks(document, units, Counter(), 38, 50)
    assert len(chunks) >= 2
    by_block = {b.block_id: b for b in blocks}
    assert all(Counter().count(c.embedding_text) <= 50 for c in chunks)
    assert all(span_text(c.spans, by_block) == c.text for c in chunks)
    assert all("Disciplina dei contributi" in c.embedding_text for c in chunks)
    assert all("Sono ammesse le seguenti spese:" in c.text for c in chunks)
    assert all(any("#art:1" == uid[-6:] for uid in c.unit_ids) for c in chunks)


def test_long_paragraph_splits_without_losing_source_words():
    document, blocks, units = parse(law("<p>Art. 1</p><p>1. " + " ".join(f"parola{i}" for i in range(150)) + "</p>"))
    chunks = build_chunks(document, units, Counter(), 35, 45)
    paragraph = next(u for u in units if u.kind == "paragraph")
    assert " ".join(c.text for c in chunks).split() == paragraph.text.split()
    assert all(Counter().count(c.embedding_text) <= 45 for c in chunks)
    assert all(span_text(c.spans, {b.block_id: b for b in blocks}) == c.text for c in chunks)


def test_status_boundary_prevents_eligible_and_ceased_siblings_mixing():
    document, _, units = parse(law("<p>Art. 1</p><p>1. Primo testo normativo.</p><p>2. Secondo testo normativo.</p>"))
    paragraphs = [u for u in units if u.kind == "paragraph"]
    paragraphs[0].status = "ceased_confirmed"
    chunks = build_chunks(document, units, Counter())
    assert len(chunks) == 2
    assert chunks[0].status == "ceased_confirmed"
    assert chunks[1].status == "no_cessation_evidence"


def test_all_metadata_snapshot_is_valid_but_contains_no_fake_chunks(tmp_path):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html"])
    manifest = build_corpus(source, tmp_path / "corpus", Counter())
    assert manifest["counts"]["chunks"] == 0
    assert not manifest["ready_for_indexing"]
    assert not manifest["quality"]["chunks_present"]

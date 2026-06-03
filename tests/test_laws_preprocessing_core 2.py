from __future__ import annotations

from pathlib import Path

from legal_rag.laws_preprocessing import (
    build_corpus_registry,
    classify_law_status,
    ingest_law,
    law_id_from_date_number,
    normalize_article_label,
    parse_italian_date,
    parse_law_filename,
)


def _write_law(root: Path, name: str, html: str) -> None:
    (root / name).write_text(html, encoding="utf-8")


def test_parse_law_filename_and_stable_ids() -> None:
    law_file = parse_law_filename("0001_LR-25-gennaio-2000-n5.html")
    assert law_file is not None
    assert law_file.law_date.isoformat() == "2000-01-25"
    assert law_file.law_number == 5
    assert law_file.law_id == "vda:lr:2000-01-25:5"
    assert law_id_from_date_number(parse_italian_date(25, "gennaio", 2000), 5) == law_file.law_id
    assert normalize_article_label("Art. 4 bis") == "4bis"


def test_article_anchor_index_comma_letter_and_note_linking(tmp_path: Path) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<h1>INDICE</h1>
<p><a href="#articolo_1__">Art. 1</a> - Titolo indice</p>
<p><a name="articolo_1__">Art. 1</a> - Titolo uno</p>
<p>Premessa breve.</p>
<p>1. Primo comma con nota <a href="#nota_01">(01)</a>.</p>
<p>a) Lettera a.</p>
<p>2. Secondo comma.</p>
<p><a name="nota_01">(<span>01</span>)</a> Nota definizione.</p>
</article>""",
    )
    registry, _ = build_corpus_registry(tmp_path)
    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert len(ingested.articles) == 1
    assert ingested.articles[0]["article_heading"] == "Titolo uno"
    labels = {passage["passage_label"] for passage in ingested.passages}
    assert {"intro", "c1", "c1.lit_a", "c2"}.issubset(labels)
    assert len(ingested.notes) == 1
    assert ingested.notes[0]["linked_article_ids"]
    assert ingested.notes[0]["linked_passage_ids"]


def test_href_and_text_references_create_allowed_edges(tmp_path: Path) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Sostituisce l'art. 2 della <a href="/app/leggieregolamenti/dettaglio?tipo=L&amp;numero_legge=2%2F01&amp;versione=V">L.R. 2/2001</a>.</p>
<p>2. Richiama la Legge regionale 3 gennaio 2002, n. 3.</p>
</article>""",
    )
    _write_law(
        tmp_path,
        "0002_LR-2-gennaio-2001-n2.html",
        "<article><h1>Legge regionale 2 gennaio 2001, n. 2 - Testo vigente</h1></article>",
    )
    _write_law(
        tmp_path,
        "0003_LR-3-gennaio-2002-n3.html",
        "<article><h1>Legge regionale 3 gennaio 2002, n. 3 - Testo vigente</h1></article>",
    )
    registry, _ = build_corpus_registry(tmp_path)
    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=200,
        chunk_overlap=20,
    )

    dsts = {edge["dst_law_id"] for edge in ingested.edges}
    assert {"vda:lr:2001-01-02:2", "vda:lr:2002-01-03:3"}.issubset(dsts)
    relation_types = {edge["relation_type"] for edge in ingested.edges}
    assert "REPLACES" in relation_types
    assert "REFERENCES" in relation_types


def test_self_references_are_not_exported_as_edges(tmp_path: Path) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Richiama la Legge regionale 1 gennaio 2000, n. 1.</p>
</article>""",
    )
    registry, _ = build_corpus_registry(tmp_path)
    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=200,
        chunk_overlap=20,
    )
    assert ingested.edges == []


def test_law_statuses() -> None:
    assert classify_law_status("Testo vigente.", 1)[0] == "current"
    assert classify_law_status("(Legge abrogata dall'art. 1 della L.R. 1 gennaio 2000, n. 1)", 1)[0] == "past"
    assert classify_law_status("(Abrogata dall'art. 1, ad eccezione dell'articolo 4)", 1)[0] == "unknown"
    assert classify_law_status("INDICE", 0)[0] == "index_or_empty"

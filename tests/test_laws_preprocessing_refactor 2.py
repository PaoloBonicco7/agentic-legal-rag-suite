from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from legal_rag.laws_preprocessing import (
    LawsPreprocessingConfig,
    normalize_ws,
    parse_blocks_from_html,
    run_laws_preprocessing,
)


def _write_law(root: Path, name: str, html: str) -> None:
    (root / name).write_text(html, encoding="utf-8")


class _LegacyBlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, object]] = []
        self._in_article = False
        self._restrict_to_article = False
        self._current_kind: str | None = None
        self._text_parts: list[str] = []
        self._anchors: list[dict[str, str]] = []
        self._links: list[dict[str, str]] = []
        self._a_ctx: dict[str, str] | None = None
        self._a_text_parts: list[str] = []

    def parse(self, html: str) -> list[dict[str, object]]:
        self._restrict_to_article = "<article" in html.lower()
        self.feed(html)
        self.close()
        return self.blocks

    def _allowed(self) -> bool:
        return not self._restrict_to_article or self._in_article

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): (value or "") for key, value in attrs}
        if tag == "article":
            self._in_article = True
            return
        if not self._allowed():
            return
        if tag == "br" and self._current_kind is not None:
            self._text_parts.append("\n")
            if self._a_ctx is not None:
                self._a_text_parts.append("\n")
            return
        if tag in {"p", "h1", "h2", "td", "th"} and self._current_kind is None:
            self._current_kind = "table_row" if tag in {"td", "th"} else tag
            self._text_parts = []
            self._anchors = []
            self._links = []
            return
        if tag == "a" and self._current_kind is not None:
            self._a_ctx = {"name": attrs_dict.get("name", ""), "href": attrs_dict.get("href", "")}
            self._a_text_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "article":
            self._flush_block()
            self._in_article = False
            return
        if not self._allowed():
            return
        if tag == "a" and self._a_ctx is not None and self._current_kind is not None:
            anchor_text = normalize_ws("".join(self._a_text_parts))
            name = self._a_ctx.get("name", "").strip()
            href = self._a_ctx.get("href", "").strip()
            if name:
                self._anchors.append({"name": name, "text": anchor_text})
            if href:
                self._links.append({"href": href, "text": anchor_text})
            self._a_ctx = None
            self._a_text_parts = []
            return
        if self._current_kind and tag in {"p", "h1", "h2", "td", "th"}:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if not self._allowed() or self._current_kind is None:
            return
        self._text_parts.append(data)
        if self._a_ctx is not None:
            self._a_text_parts.append(data)

    def _flush_block(self) -> None:
        if self._current_kind is None:
            return
        text = normalize_ws("".join(self._text_parts))
        if text:
            self.blocks.append(
                {
                    "kind": self._current_kind,
                    "text": text,
                    "anchors": tuple(self._anchors),
                    "links": tuple(self._links),
                }
            )
        self._current_kind = None
        self._text_parts = []
        self._anchors = []
        self._links = []


def _block_snapshot(html: str) -> list[dict[str, object]]:
    return [
        {
            "kind": block.kind,
            "text": block.text,
            "anchors": tuple({"name": anchor.name, "text": anchor.text} for anchor in block.anchors),
            "links": tuple({"href": link.href, "text": link.text} for link in block.links),
        }
        for block in parse_blocks_from_html(html)
    ]


@pytest.mark.parametrize(
    "html",
    [
        "<article><h1>Legge regionale 1 gennaio 2000, n. 1</h1><p>Testo semplice.</p></article>",
        """<article>
<h1>INDICE</h1>
<p><a href="#articolo_1__">Art. 1</a> - Indice</p>
<p><a name="articolo_1__">Art. 1</a> - Oggetto</p>
<p>1. Primo comma<br>seconda riga.</p>
</article>""",
        """<article>
<p><a name="articolo_1__">Art. 1</a></p> <p><i>(Rubrica)</i></p>
<p>1. Testo con <a href="/app/leggieregolamenti/dettaglio?tipo=L&amp;numero_legge=2%2F01&amp;versione=V">L.R. 2/2001</a>.</p>
<p><a name="nota_1">(<span>1</span>)</a> Nota.</p>
</article>""",
    ],
)
def test_lxml_block_parser_matches_legacy_stdlib_behavior(html: str) -> None:
    assert _block_snapshot(html) == _LegacyBlockParser().parse(html)


def test_run_laws_preprocessing_rejects_dangerous_output_paths(tmp_path: Path) -> None:
    source = tmp_path / "laws_html"
    source.mkdir()
    _write_law(source, "0001_LR-1-gennaio-2000-n1.html", "<article><h1>Legge regionale 1 gennaio 2000, n. 1</h1></article>")

    for output in [source, source / "generated", tmp_path, Path.cwd()]:
        with pytest.raises(ValueError):
            run_laws_preprocessing(LawsPreprocessingConfig(source_dir=str(source), output_dir=str(output)))


def test_run_laws_preprocessing_golden_jsonl_outputs(tmp_path: Path) -> None:
    source = tmp_path / "laws_html"
    output = tmp_path / "laws_dataset_clean"
    source.mkdir()
    (source / ".DS_Store").write_text("ignored", encoding="utf-8")
    _write_law(
        source,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p>Oggetto della legge.</p>
<p><a name="articolo_1__">Art. 1</a> - Oggetto</p>
<p>1. Richiama la Legge regionale 2 gennaio 2001, n. 2.</p>
<p>2. Testo con nota <a href="#nota_1">(1)</a>.</p>
<p><a name="nota_1">(<span>1</span>)</a> Nota modificata dalla <a href="/app/leggieregolamenti/dettaglio?tipo=L&amp;numero_legge=2%2F01&amp;versione=V">L.R. 2/2001</a>.</p>
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
        LawsPreprocessingConfig(source_dir=str(source), output_dir=str(output), chunk_size=100, chunk_overlap=10)
    )

    assert manifest["ready_for_indexing"] is True
    expected_counts = {"laws": 2, "articles": 2, "passages": 3, "notes": 1, "edges": 2, "chunks": 3}
    assert manifest["counts"] == expected_counts

    expected = {
        "laws.jsonl": [
            {"law_date": "2000-01-01", "law_id": "vda:lr:2000-01-01:1", "law_number": 1, "law_status": "current", "law_title": "Legge regionale 1 gennaio 2000, n. 1 - Testo vigente", "law_type": "LR", "links_out": [], "preamble_text": "Legge regionale 1 gennaio 2000, n. 1 - Testo vigente\nOggetto della legge.", "source_file": "0001_LR-1-gennaio-2000-n1.html", "status_confidence": 0.72, "status_evidence": []},
            {"law_date": "2001-01-02", "law_id": "vda:lr:2001-01-02:2", "law_number": 2, "law_status": "current", "law_title": "Legge regionale 2 gennaio 2001, n. 2 - Testo vigente", "law_type": "LR", "links_out": [], "preamble_text": "Legge regionale 2 gennaio 2001, n. 2 - Testo vigente", "source_file": "0002_LR-2-gennaio-2001-n2.html", "status_confidence": 0.72, "status_evidence": []},
        ],
        "articles.jsonl": [
            {"abrogated_by": None, "amended_by_law_ids": ["vda:lr:2001-01-02:2"], "anchor_name": "articolo_1__", "article_heading": "Oggetto", "article_id": "vda:lr:2000-01-01:1#art:1", "article_label_norm": "1", "article_label_raw": "Art. 1", "article_status": "current", "article_text": "1. Richiama la Legge regionale 2 gennaio 2001, n. 2.\n2. Testo con nota (1).", "is_abrogated": False, "law_id": "vda:lr:2000-01-01:1", "links_out": [{"href": "#nota_1", "text": "(1)"}], "note_anchor_names": ["nota_1"], "structure_path": ""},
            {"abrogated_by": None, "amended_by_law_ids": [], "anchor_name": "articolo_1__", "article_heading": None, "article_id": "vda:lr:2001-01-02:2#art:1", "article_label_norm": "1", "article_label_raw": "Art. 1", "article_status": "current", "article_text": "1. Testo.", "is_abrogated": False, "law_id": "vda:lr:2001-01-02:2", "links_out": [], "note_anchor_names": [], "structure_path": ""},
        ],
        "passages.jsonl": [
            {"article_id": "vda:lr:2000-01-01:1#art:1", "law_id": "vda:lr:2000-01-01:1", "links_out": [], "note_anchor_names": [], "passage_id": "vda:lr:2000-01-01:1#art:1#p:c1", "passage_kind": "comma", "passage_label": "c1", "passage_text": "1. Richiama la Legge regionale 2 gennaio 2001, n. 2.", "related_law_ids": ["vda:lr:2001-01-02:2"], "relation_types": ["REFERENCES"], "structure_path": ""},
            {"article_id": "vda:lr:2000-01-01:1#art:1", "law_id": "vda:lr:2000-01-01:1", "links_out": [{"href": "#nota_1", "text": "(1)"}], "note_anchor_names": ["nota_1"], "passage_id": "vda:lr:2000-01-01:1#art:1#p:c2", "passage_kind": "comma", "passage_label": "c2", "passage_text": "2. Testo con nota (1).", "related_law_ids": [], "relation_types": [], "structure_path": ""},
            {"article_id": "vda:lr:2001-01-02:2#art:1", "law_id": "vda:lr:2001-01-02:2", "links_out": [], "note_anchor_names": [], "passage_id": "vda:lr:2001-01-02:2#art:1#p:c1", "passage_kind": "comma", "passage_label": "c1", "passage_text": "1. Testo.", "related_law_ids": [], "relation_types": [], "structure_path": ""},
        ],
        "notes.jsonl": [
            {"law_id": "vda:lr:2000-01-01:1", "linked_article_ids": ["vda:lr:2000-01-01:1#art:1"], "linked_passage_ids": ["vda:lr:2000-01-01:1#art:1#p:c2"], "links_out": [{"href": "/app/leggieregolamenti/dettaglio?tipo=L&numero_legge=2%2F01&versione=V", "text": "L.R. 2/2001"}], "note_anchor_name": "nota_1", "note_id": "vda:lr:2000-01-01:1#note:nota_1", "note_kind": "modified", "note_number": "1", "note_text": "Nota modificata dalla L.R. 2/2001."},
        ],
        "edges.jsonl": [
            {"confidence": 0.45, "context": "passage", "dst_article_label_norm": None, "dst_law_id": "vda:lr:2001-01-02:2", "edge_id": "33b7c3447ae7ac4810950d58b348468eb5a3f3e7c2744cd192e7d1fe58cf981b", "evidence": "1. Richiama la Legge regionale 2 gennaio 2001, n. 2.", "evidence_text": "1. Richiama la Legge regionale 2 gennaio 2001, n. 2.", "extraction_method": "text_regex", "is_self_loop": False, "note_anchor_name": None, "relation_type": "REFERENCES", "source_file": "0001_LR-1-gennaio-2000-n1.html", "src_article_id": "vda:lr:2000-01-01:1#art:1", "src_law_id": "vda:lr:2000-01-01:1", "src_passage_id": "vda:lr:2000-01-01:1#art:1#p:c1"},
            {"confidence": 0.8, "context": "note", "dst_article_label_norm": None, "dst_law_id": "vda:lr:2001-01-02:2", "edge_id": "461ba7c4f264c0f34ec4f15c195baa3f53febc64909212cc2b149c5634fba279", "evidence": "(1) Nota modificata dalla L.R. 2/2001.", "evidence_text": "(1) Nota modificata dalla L.R. 2/2001.", "extraction_method": "href", "is_self_loop": False, "note_anchor_name": "nota_1", "relation_type": "MODIFIED_BY", "source_file": "0001_LR-1-gennaio-2000-n1.html", "src_article_id": None, "src_law_id": "vda:lr:2000-01-01:1", "src_passage_id": None},
        ],
        "chunks.jsonl": [
            {"article_id": "vda:lr:2000-01-01:1#art:1", "article_label_norm": "1", "article_status": "current", "chunk_id": "vda:lr:2000-01-01:1#art:1#p:c1#chunk:0", "chunk_seq": 0, "inbound_law_ids": [], "index_views": ["historical", "current"], "law_date": "2000-01-01", "law_id": "vda:lr:2000-01-01:1", "law_number": 1, "law_status": "current", "law_title": "Legge regionale 1 gennaio 2000, n. 1 - Testo vigente", "outbound_law_ids": ["vda:lr:2001-01-02:2"], "passage_id": "vda:lr:2000-01-01:1#art:1#p:c1", "passage_label": "c1", "related_law_ids": ["vda:lr:2001-01-02:2"], "relation_types": ["MODIFIED_BY", "REFERENCES"], "source_file": "0001_LR-1-gennaio-2000-n1.html", "structure_path": "", "text": "1. Richiama la Legge regionale 2 gennaio 2001, n. 2.", "text_for_embedding": "[LR 2000-01-01 n.1] Legge regionale 1 gennaio 2000, n. 1 - Testo vigente | Art. 1 | c1 |\n\n1. Richiama la Legge regionale 2 gennaio 2001, n. 2."},
            {"article_id": "vda:lr:2000-01-01:1#art:1", "article_label_norm": "1", "article_status": "current", "chunk_id": "vda:lr:2000-01-01:1#art:1#p:c2#chunk:0", "chunk_seq": 0, "inbound_law_ids": [], "index_views": ["historical", "current"], "law_date": "2000-01-01", "law_id": "vda:lr:2000-01-01:1", "law_number": 1, "law_status": "current", "law_title": "Legge regionale 1 gennaio 2000, n. 1 - Testo vigente", "outbound_law_ids": ["vda:lr:2001-01-02:2"], "passage_id": "vda:lr:2000-01-01:1#art:1#p:c2", "passage_label": "c2", "related_law_ids": [], "relation_types": ["MODIFIED_BY", "REFERENCES"], "source_file": "0001_LR-1-gennaio-2000-n1.html", "structure_path": "", "text": "2. Testo con nota (1).", "text_for_embedding": "[LR 2000-01-01 n.1] Legge regionale 1 gennaio 2000, n. 1 - Testo vigente | Art. 1 | c2 |\n\n2. Testo con nota (1)."},
            {"article_id": "vda:lr:2001-01-02:2#art:1", "article_label_norm": "1", "article_status": "current", "chunk_id": "vda:lr:2001-01-02:2#art:1#p:c1#chunk:0", "chunk_seq": 0, "inbound_law_ids": ["vda:lr:2000-01-01:1"], "index_views": ["historical", "current"], "law_date": "2001-01-02", "law_id": "vda:lr:2001-01-02:2", "law_number": 2, "law_status": "current", "law_title": "Legge regionale 2 gennaio 2001, n. 2 - Testo vigente", "outbound_law_ids": [], "passage_id": "vda:lr:2001-01-02:2#art:1#p:c1", "passage_label": "c1", "related_law_ids": [], "relation_types": [], "source_file": "0002_LR-2-gennaio-2001-n2.html", "structure_path": "", "text": "1. Testo.", "text_for_embedding": "[LR 2001-01-02 n.2] Legge regionale 2 gennaio 2001, n. 2 - Testo vigente | Art. 1 | c1 |\n\n1. Testo."},
        ],
    }

    for filename, expected_records in expected.items():
        actual_records = [json.loads(line) for line in (output / filename).read_text(encoding="utf-8").splitlines()]
        assert actual_records == expected_records

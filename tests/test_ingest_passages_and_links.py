import tempfile
import unittest
from pathlib import Path

from laws_ingestion.core.ingest import ingest_law
from laws_ingestion.core.registry import build_corpus_registry


class TestIngestPassagesAndLinks(unittest.TestCase):
    def _ingest_html(self, html: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            f = p / "0001_LR-1-gennaio-2000-n1.html"
            f.write_text(html, encoding="utf-8")
            registry = build_corpus_registry(p)
            law_file = next(iter(registry.by_law_id.values()))
            return ingest_law(law_file, registry, backend="stdlib")

    def test_anchor_heading_is_extracted(self) -> None:
        html = """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a> - Titolo uno</p>
<p>Testo articolo 1.</p>
</article>"""
        ing = self._ingest_html(html)
        self.assertEqual(len(ing.articles), 1)
        self.assertEqual(ing.articles[0]["article_label_norm"], "1")
        self.assertEqual(ing.articles[0]["article_heading"], "Titolo uno")

    def test_passage_segmentation_comma_and_lettera(self) -> None:
        html = """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>Premessa breve.</p>
<p>1. Primo comma.</p>
<p>a) Lettera a.</p>
<p>b) Lettera b.</p>
<p>2. Secondo comma.</p>
</article>"""
        ing = self._ingest_html(html)
        labels = [p["passage_label"] for p in ing.passages]
        self.assertIn("intro", labels)
        self.assertIn("c1", labels)
        self.assertIn("c1.lit_a", labels)
        self.assertIn("c1.lit_b", labels)
        self.assertIn("c2", labels)

    def test_note_linking_to_passage(self) -> None:
        html = """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Testo con nota <a href="#nota_01">(01)</a>.</p>
<p><a name="nota_01">(<span>01</span>)</a> Nota definizione.</p>
</article>"""
        ing = self._ingest_html(html)
        self.assertEqual(len(ing.notes), 1)
        n = ing.notes[0]
        self.assertEqual(n["note_anchor_name"], "nota_01")
        self.assertTrue(len(n["linked_article_ids"]) >= 1)
        self.assertTrue(len(n["linked_passage_ids"]) >= 1)

    def test_href_reference_resolution_and_edge_metadata(self) -> None:
        # Create two files so that year/number resolution is possible via registry.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            a = p / "0001_LR-1-gennaio-2000-n1.html"
            b = p / "0002_LR-23-ottobre-1995-n45.html"
            a.write_text(
                """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p>(Abrogata dall'art. 65 della <a href="/app/leggieregolamenti/dettaglio?tipo=L&amp;numero_legge=45%2F95&amp;versione=V">L.R. 23 ottobre 1995, n. 45</a>).</p>
</article>""",
                encoding="utf-8",
            )
            b.write_text(
                """<article><h1>Legge regionale 23 ottobre 1995, n. 45 - Testo vigente</h1></article>""",
                encoding="utf-8",
            )
            registry = build_corpus_registry(p)
            law_file = registry.by_law_id["vda:lr:2000-01-01:1"]
            ing = ingest_law(law_file, registry, backend="stdlib")

            # Edge should resolve to the second law in the corpus.
            dsts = {e["dst_law_id"] for e in ing.edges}
            self.assertIn("vda:lr:1995-10-23:45", dsts)

    def test_indice_and_table_wrapper_do_not_duplicate_articles(self) -> None:
        html = """<article>
<h1>Legge regionale 10 aprile 1997, n. 12 - Testo vigente</h1>
<div class="table-wrapper">
<table><tr><td>
<h1>INDICE</h1>
<p>Art. 1 Finalita</p>
<p>Art. 2 Classificazione</p>
</td></tr></table>
</div>
<p><a name="articolo_1__">Art. 1</a></p>
<p>Testo art. 1</p>
<p><a name="articolo_2__">Art. 2</a></p>
<p>Testo art. 2</p>
</article>"""
        ing = self._ingest_html(html)
        ids = [a["article_id"] for a in ing.articles]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)

    def test_artt_range_creates_pseudo_article_and_note(self) -> None:
        html = """<article>
<h1>Legge regionale 24 agosto 1992, n. 55 - Testo vigente</h1>
<p>Artt. 1. - 2. <a href="#nota_1">(1)</a></p>
<p>¦</p>
<p><a name="nota_1">(<span>1</span>)</a> Modificano gli artt. 5 e 6 della L.R. 14 gennaio 1988, n. 6.</p>
</article>"""
        ing = self._ingest_html(html)
        self.assertEqual(len(ing.articles), 1)
        self.assertEqual(ing.articles[0]["article_label_norm"], "unico")
        self.assertEqual(len(ing.notes), 1)

    def test_abrogata_ad_eccezione_and_typo_references_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            a = p / "0001_LR-20-gennaio-1998-n3.html"
            b = p / "0002_LR-1-aprile-2004-n3.html"
            c = p / "0003_LR-16-marzo-2006-n6.html"
            a.write_text(
                """<article>
<h1>Legge regionale 20 gennaio 1998, n. 3 - Testo vigente</h1>
<p>(Abrogata dall'art. 30 della L:R. 1° aprile 2004, n. 3, ad eccezione dell'articolo 40).</p>
<p><a name="articolo_40__">Art. 40</a></p>
<p><a name="nota_1">(<span>1</span>)</a> Sostituisce l'art. 1 della L.R.. 16 marzo 2006, n. 6.</p>
</article>""",
                encoding="utf-8",
            )
            b.write_text("""<article><h1>Legge regionale 1 aprile 2004, n. 3 - Testo vigente</h1></article>""", encoding="utf-8")
            c.write_text(
                """<article><h1>Legge regionale 16 marzo 2006, n. 6 - Testo vigente</h1><p>Testo.</p></article>""",
                encoding="utf-8",
            )
            registry = build_corpus_registry(p)
            ing = ingest_law(registry.by_law_id["vda:lr:1998-01-20:3"], registry, backend="stdlib")
            dsts = {e["dst_law_id"] for e in ing.edges}
            self.assertIn("vda:lr:2004-04-01:3", dsts)
            self.assertIn("vda:lr:2006-03-16:6", dsts)

    def test_cessata_efficacia_and_long_note_are_handled(self) -> None:
        long_tail = " x" * 4000
        html = f"""<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p>(Cessata efficacia dall'art. 2 della L.R. 2 gennaio 2001, n. 2)</p>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. testo.</p>
<p><a name="nota_1">(<span>1</span>)</a> Nota molto lunga{long_tail}</p>
</article>"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            a = p / "0001_LR-1-gennaio-2000-n1.html"
            b = p / "0002_LR-2-gennaio-2001-n2.html"
            a.write_text(html, encoding="utf-8")
            b.write_text("""<article><h1>Legge regionale 2 gennaio 2001, n. 2 - Testo vigente</h1></article>""", encoding="utf-8")
            registry = build_corpus_registry(p)
            ing = ingest_law(registry.by_law_id["vda:lr:2000-01-01:1"], registry, backend="stdlib")
            self.assertEqual(len(ing.notes), 1)
            self.assertGreater(len(ing.notes[0]["note_text"]), 7000)
            self.assertTrue(any(e["edge_type"] == "ABROGATED_BY" for e in ing.edges))


if __name__ == "__main__":
    unittest.main()

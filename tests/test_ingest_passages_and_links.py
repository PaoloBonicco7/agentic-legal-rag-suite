import tempfile
import unittest
from pathlib import Path

from laws_ingestion.ingest import ingest_law
from laws_ingestion.registry import build_corpus_registry


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


if __name__ == "__main__":
    unittest.main()


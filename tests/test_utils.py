import tempfile
import unittest
from pathlib import Path

from laws_ingestion.utils import compute_dataset_id, normalize_article_label, parse_italian_date


class TestUtils(unittest.TestCase):
    def test_normalize_article_label(self) -> None:
        self.assertEqual(normalize_article_label("Art. 4 bis"), "4bis")
        self.assertEqual(normalize_article_label("Art.16"), "16")
        self.assertEqual(normalize_article_label("Art. 10bis."), "10bis")
        self.assertEqual(normalize_article_label("Art. 10 quater"), "10quater")
        self.assertEqual(normalize_article_label("10 ter"), "10ter")
        self.assertEqual(normalize_article_label("ARTICOLO 1"), "1")
        self.assertEqual(normalize_article_label("Articolo 10 bis"), "10bis")

    def test_parse_italian_date(self) -> None:
        d = parse_italian_date(25, "gennaio", 2000)
        self.assertEqual(d.isoformat(), "2000-01-25")

    def test_compute_dataset_id_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            a = p / "a.html"
            b = p / "b.html"
            a.write_text("<html>a</html>", encoding="utf-8")
            b.write_text("<html>b</html>", encoding="utf-8")

            id1 = compute_dataset_id([a, b])
            id2 = compute_dataset_id([a, b])
            self.assertEqual(id1, id2)

            b.write_text("<html>b2</html>", encoding="utf-8")
            id3 = compute_dataset_id([a, b])
            self.assertNotEqual(id1, id3)


if __name__ == "__main__":
    unittest.main()

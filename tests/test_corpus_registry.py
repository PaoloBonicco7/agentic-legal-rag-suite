import unittest

from laws_ingestion.core.registry import parse_law_filename


class TestCorpusRegistry(unittest.TestCase):
    def test_parse_law_filename(self) -> None:
        lf = parse_law_filename("0001_LR-25-gennaio-2000-n5.html")
        assert lf is not None
        self.assertEqual(lf.law_date.isoformat(), "2000-01-25")
        self.assertEqual(lf.law_number, 5)
        self.assertEqual(lf.law_id, "vda:lr:2000-01-25:5")


if __name__ == "__main__":
    unittest.main()

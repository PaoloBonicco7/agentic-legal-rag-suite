import unittest

from baselines.benchmark import parse_domanda_field, parse_reference_line


class TestBenchmarkParsing(unittest.TestCase):
    def test_parse_domanda_field_extracts_stem_and_6_options(self) -> None:
        domanda = (
            "Quali sono gli organi dell'azienda USL?\n"
            "A) Opzione A\n"
            "B) Opzione B\n"
            "C) Opzione C\n"
            "D) Opzione D\n"
            "E) Opzione E\n"
            "F) Opzione F\n"
        )
        stem, options = parse_domanda_field(domanda)
        self.assertEqual(stem, "Quali sono gli organi dell'azienda USL?")
        self.assertEqual([o.label for o in options], ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(options[0].text, "Opzione A")
        self.assertEqual(options[-1].text, "Opzione F")

    def test_parse_reference_line_normalizes_article_label(self) -> None:
        ref = "Legge regionale 25 gennaio 2000, n. 5 - Art. 10 bis"
        parsed = parse_reference_line(ref)
        self.assertEqual(parsed.law_date.isoformat(), "2000-01-25")
        self.assertEqual(parsed.law_number, 5)
        self.assertEqual(parsed.article_label_norm, "10bis")


if __name__ == "__main__":
    unittest.main()

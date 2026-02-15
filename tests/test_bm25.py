import unittest

from baselines.bm25 import BM25Index


class TestBM25(unittest.TestCase):
    def test_bm25_ranks_matching_doc_higher(self) -> None:
        docs = [
            ("d1", "il gatto mangia il pesce", {"doc": 1}),
            ("d2", "il cane corre nel parco", {"doc": 2}),
        ]
        idx = BM25Index()
        idx.build(docs)

        res = idx.search("gatto", k=2)
        self.assertGreaterEqual(len(res), 1)
        self.assertEqual(res[0].doc_id, "d1")


if __name__ == "__main__":
    unittest.main()

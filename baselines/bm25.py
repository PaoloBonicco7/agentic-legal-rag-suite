from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import json
import math
import re
import gzip
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(r"[0-9]+|[a-zàèéìòù]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """
    Deterministic, dependency-free tokenization tuned for Italian legal text.
    """
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    out: list[str] = []
    for t in tokens:
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(t)
    return out


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float
    meta: dict


class BM25Index:
    """
    Simple BM25 Okapi index (stdlib-only).
    """

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)

        self._doc_ids: list[str] = []
        self._doc_meta: list[dict] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0

        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def build(self, docs: Iterable[tuple[str, str, dict]]) -> None:
        doc_ids: list[str] = []
        doc_meta: list[dict] = []
        doc_len: list[int] = []

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: dict[str, int] = defaultdict(int)

        total_len = 0
        for i, (doc_id, text, meta) in enumerate(docs):
            tokens = tokenize(text)
            tf = Counter(tokens)
            dl = sum(tf.values())
            total_len += dl

            doc_ids.append(str(doc_id))
            doc_meta.append(dict(meta or {}))
            doc_len.append(int(dl))

            for term, freq in tf.items():
                postings[term].append((i, int(freq)))
                df[term] += 1

        self._doc_ids = doc_ids
        self._doc_meta = doc_meta
        self._doc_len = doc_len
        self._avgdl = (total_len / len(doc_len)) if doc_len else 0.0

        n_docs = len(doc_len)
        idf: dict[str, float] = {}
        for term, dft in df.items():
            idf[term] = math.log(1.0 + (n_docs - dft + 0.5) / (dft + 0.5))
        self._postings = dict(postings)
        self._idf = idf

    def search(self, query: str, *, k: int = 10) -> list[ScoredDoc]:
        if k <= 0 or not self._doc_ids:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores = [0.0] * len(self._doc_ids)
        avgdl = self._avgdl or 1.0
        k1 = self.k1
        b = self.b

        for term in q_terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_idx, tf in posting:
                dl = self._doc_len[doc_idx] or 0
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                if denom <= 0:
                    continue
                scores[doc_idx] += idf * (tf * (k1 + 1.0) / denom)

        idxs = [i for i, s in enumerate(scores) if s > 0.0]
        if not idxs:
            return []

        top = heapq.nlargest(min(k, len(idxs)), idxs, key=lambda i: scores[i])
        return [ScoredDoc(doc_id=self._doc_ids[i], score=float(scores[i]), meta=self._doc_meta[i]) for i in top]

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self._doc_ids,
            "doc_meta": self._doc_meta,
            "doc_len": self._doc_len,
            "avgdl": self._avgdl,
            "idf": self._idf,
            "postings": {t: [[i, tf] for i, tf in ps] for t, ps in self._postings.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Index":
        idx = cls(k1=float(d.get("k1", 1.2)), b=float(d.get("b", 0.75)))
        idx._doc_ids = list(d.get("doc_ids") or [])
        idx._doc_meta = list(d.get("doc_meta") or [])
        idx._doc_len = [int(x) for x in (d.get("doc_len") or [])]
        idx._avgdl = float(d.get("avgdl") or 0.0)
        idx._idf = {str(k): float(v) for k, v in (d.get("idf") or {}).items()}
        postings_raw = d.get("postings") or {}
        postings: dict[str, list[tuple[int, int]]] = {}
        for term, ps in postings_raw.items():
            postings[str(term)] = [(int(i), int(tf)) for i, tf in ps]
        idx._postings = postings
        return idx

    def save_json_gz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load_json_gz(cls, path: Path) -> "BM25Index":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


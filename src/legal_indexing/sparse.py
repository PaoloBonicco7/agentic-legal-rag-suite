from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


_TOKEN_RE = re.compile(r"[0-9]+|[a-zàèéìòù]+", re.IGNORECASE)

# Compact Italian stopword set tailored for legal prose.
_IT_STOPWORDS = {
    "a",
    "ad",
    "al",
    "alla",
    "alle",
    "anche",
    "che",
    "chi",
    "con",
    "da",
    "dal",
    "dalla",
    "dalle",
    "dei",
    "del",
    "della",
    "delle",
    "di",
    "e",
    "ed",
    "gli",
    "i",
    "il",
    "in",
    "la",
    "le",
    "lo",
    "ma",
    "nel",
    "nella",
    "nelle",
    "nei",
    "o",
    "per",
    "piu",
    "quale",
    "quali",
    "quello",
    "questa",
    "questo",
    "si",
    "sono",
    "su",
    "tra",
    "un",
    "una",
}


@dataclass(frozen=True)
class SparseVectorData:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": list(self.indices),
            "values": list(self.values),
        }


class SparseEncoder:
    """Deterministic sparse encoder with BM25-like IDF weighting."""

    def __init__(
        self,
        *,
        min_token_len: int = 2,
        stopwords_lang: str = "it",
    ) -> None:
        if int(min_token_len) <= 0:
            raise ValueError("min_token_len must be > 0")
        self.min_token_len = int(min_token_len)
        self.stopwords_lang = str(stopwords_lang or "it").strip().lower()
        self._stopwords = self._resolve_stopwords(self.stopwords_lang)
        self._vocab: dict[str, int] = {}
        self._idf_by_index: list[float] = []
        self._doc_count: int = 0

    @staticmethod
    def _resolve_stopwords(lang: str) -> set[str]:
        if lang == "it":
            return set(_IT_STOPWORDS)
        if lang in {"", "none"}:
            return set()
        raise ValueError("stopwords_lang must be one of: it, none")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def doc_count(self) -> int:
        return int(self._doc_count)

    @property
    def is_fitted(self) -> bool:
        return bool(self._vocab)

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for match in _TOKEN_RE.finditer(str(text or "").lower()):
            token = match.group(0).strip()
            if not token:
                continue
            if len(token) < self.min_token_len and not token.isdigit():
                continue
            if token in self._stopwords:
                continue
            out.append(token)
        return out

    def fit(self, texts: Sequence[str]) -> None:
        doc_tokens: list[set[str]] = []
        for text in texts:
            tokens = set(self.tokenize(text))
            doc_tokens.append(tokens)

        n_docs = len(doc_tokens)
        if n_docs <= 0:
            self._vocab = {}
            self._idf_by_index = []
            self._doc_count = 0
            return

        df: dict[str, int] = {}
        for token_set in doc_tokens:
            for tok in token_set:
                df[tok] = df.get(tok, 0) + 1

        vocab_terms = sorted(df.keys())
        vocab = {tok: idx for idx, tok in enumerate(vocab_terms)}
        idf_by_index = [0.0] * len(vocab_terms)
        for tok, idx in vocab.items():
            dft = df.get(tok, 0)
            idf = math.log(1.0 + (n_docs - dft + 0.5) / (dft + 0.5))
            idf_by_index[idx] = float(idf)

        self._vocab = vocab
        self._idf_by_index = idf_by_index
        self._doc_count = n_docs

    def transform(self, text: str, *, is_query: bool = False) -> SparseVectorData:
        if not self._vocab:
            return SparseVectorData(indices=tuple(), values=tuple())

        counts = Counter(self.tokenize(text))
        if not counts:
            return SparseVectorData(indices=tuple(), values=tuple())

        pairs: list[tuple[int, float]] = []
        for tok, tf in counts.items():
            idx = self._vocab.get(tok)
            if idx is None:
                continue
            idf = self._idf_by_index[idx]
            if tf <= 0:
                continue
            # Same weighting for docs/queries, stable across channels.
            weight = (1.0 + math.log(float(tf))) * idf
            if weight > 0:
                pairs.append((idx, float(weight)))

        if not pairs:
            return SparseVectorData(indices=tuple(), values=tuple())

        pairs.sort(key=lambda x: x[0])
        norm = math.sqrt(sum(v * v for _, v in pairs))
        if norm > 0:
            values = tuple(round(v / norm, 8) for _, v in pairs)
        else:
            values = tuple(v for _, v in pairs)
        indices = tuple(idx for idx, _ in pairs)
        return SparseVectorData(indices=indices, values=values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "min_token_len": self.min_token_len,
            "stopwords_lang": self.stopwords_lang,
            "doc_count": self._doc_count,
            "vocab": sorted(self._vocab.items(), key=lambda x: x[1]),
            "idf_by_index": list(self._idf_by_index),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseEncoder":
        enc = cls(
            min_token_len=int(payload.get("min_token_len", 2)),
            stopwords_lang=str(payload.get("stopwords_lang", "it")),
        )
        vocab_raw = payload.get("vocab") or []
        vocab: dict[str, int] = {}
        for item in vocab_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            tok = str(item[0])
            idx = int(item[1])
            vocab[tok] = idx

        idf_raw = payload.get("idf_by_index") or []
        idf = [float(x) for x in idf_raw]
        if vocab and len(idf) != len(vocab):
            raise ValueError("SparseEncoder artifact is corrupted: idf size != vocab size")

        enc._vocab = vocab
        enc._idf_by_index = idf
        enc._doc_count = int(payload.get("doc_count", 0))
        return enc

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "SparseEncoder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid sparse encoder artifact format: {path}")
        return cls.from_dict(payload)


def build_sparse_encoder(
    texts: Iterable[str],
    *,
    min_token_len: int = 2,
    stopwords_lang: str = "it",
) -> SparseEncoder:
    encoder = SparseEncoder(min_token_len=min_token_len, stopwords_lang=stopwords_lang)
    encoder.fit(list(texts))
    return encoder


__all__ = [
    "SparseEncoder",
    "SparseVectorData",
    "build_sparse_encoder",
]

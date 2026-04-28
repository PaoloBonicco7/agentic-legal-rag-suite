from __future__ import annotations

import re


_WORD_RE = re.compile(r"\S+")


def chunk_text_words(
    text: str, *, max_words: int = 600, overlap_words: int = 80
) -> list[str]:
    """
    Deterministic word-based chunking with overlap.

    Used to split long passages (comma/lettera/intro) into embedding-sized chunks.
    """
    words = _WORD_RE.findall(text or "")
    if not words:
        return []
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks

"""Pinned common experimental tokenizer."""

from typing import Protocol

BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class TokenCounter(Protocol):
    name: str
    revision: str

    def count(self, text: str) -> int: ...


class BgeTokenCounter:
    name = BGE_MODEL
    revision = BGE_REVISION

    def __init__(self, *, local_files_only: bool = False):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, revision=self.revision, local_files_only=local_files_only)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

"""Query rewriting helpers for retrieval diagnostics."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.io import sha256_text
from legal_rag.oracle_context_evaluation.llm import StructuredChatClient
from legal_rag.simple_rag.models import RetrievedChunkRecord

from .evaluator import dedupe_chunks, retrieve_direct, sanitize_cache_name
from .query_rewriting_prompts import (
    QUERY_REWRITING_PROMPT_VERSION,
    build_hyde_prompt,
    build_multi_query_prompt,
    build_rewrite_prompt,
)

QueryRewriteStrategy = Literal["rewrite", "hyde", "multi_query"]


class _Record(BaseModel):
    """Strict base model for JSON-friendly query rewrite contracts."""

    model_config = ConfigDict(extra="forbid")


class RewriteOutput(_Record):
    """Structured output for a single rewritten query."""

    rewritten_query: str = Field(min_length=1)


class HydeOutput(_Record):
    """Structured output for a hypothetical answer document."""

    hypothetical_answer: str = Field(min_length=1)


class MultiQueryOutput(_Record):
    """Structured output for multiple query variants."""

    queries: list[str] = Field(min_length=1)


class QueryRewriteResult(_Record):
    """Normalized query rewriting result."""

    strategy: QueryRewriteStrategy
    question: str
    queries: list[str] = Field(min_length=1)
    model: str
    prompt_version: str
    cache_hit: bool


class QueryRewriteCache:
    """File-backed JSONL cache for query rewriting outputs."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    record = json.loads(text)
                    self._entries[str(record["key"])] = _clean_queries(record.get("queries") or [])

    @property
    def path(self) -> Path:
        """Return the on-disk path for this cache."""
        return self._path

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def make_key(*, question: str, strategy: str, model: str, prompt_version: str) -> str:
        """Build the versioned cache key required by the roadmap contract."""
        return sha256_text(
            " | ".join(
                [
                    str(question or ""),
                    str(strategy or ""),
                    str(model or ""),
                    str(prompt_version or ""),
                ]
            )
        )

    def get(self, key: str) -> list[str] | None:
        """Return cached query variants for a key, or None."""
        with self._lock:
            cached = self._entries.get(key)
            return None if cached is None else list(cached)

    def set(self, key: str, queries: Sequence[str]) -> None:
        """Append a new entry to memory and to the JSONL file on disk."""
        normalized = _clean_queries(queries)
        if not normalized:
            raise ValueError("query rewrite cache entries require at least one non-empty query")
        with self._lock:
            self._entries[key] = normalized
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "queries": normalized}, ensure_ascii=False, sort_keys=True) + "\n")


def query_rewrite_cache_path(
    cache_root: str | Path,
    *,
    strategy: QueryRewriteStrategy,
    model: str,
    prompt_version: str = QUERY_REWRITING_PROMPT_VERSION,
) -> Path:
    """Return the default query rewriting cache path for one strategy/model/version."""
    return (
        Path(cache_root)
        / "query_rewriting"
        / f"{strategy}__{sanitize_cache_name(model)}__{sanitize_cache_name(prompt_version)}.jsonl"
    )


def rewrite_query(
    *,
    llm_client: StructuredChatClient,
    question: str,
    model: str,
    timeout_seconds: int,
    cache: QueryRewriteCache,
    prompt_version: str = QUERY_REWRITING_PROMPT_VERSION,
) -> QueryRewriteResult:
    """Rewrite a question into one retrieval query, using a versioned cache."""
    key = QueryRewriteCache.make_key(question=question, strategy="rewrite", model=model, prompt_version=prompt_version)
    cached = cache.get(key)
    if cached is not None:
        return _result(strategy="rewrite", question=question, queries=cached, model=model, prompt_version=prompt_version, cache_hit=True)
    schema = RewriteOutput.model_json_schema()
    call = llm_client.structured_chat(
        prompt=build_rewrite_prompt(question, payload_schema=schema),
        model=model,
        payload_schema=schema,
        timeout_seconds=timeout_seconds,
    )
    output = RewriteOutput.model_validate(call["structured"])
    queries = _clean_queries([output.rewritten_query])
    cache.set(key, queries)
    return _result(strategy="rewrite", question=question, queries=queries, model=model, prompt_version=prompt_version, cache_hit=False)


def generate_hyde(
    *,
    llm_client: StructuredChatClient,
    question: str,
    model: str,
    timeout_seconds: int,
    cache: QueryRewriteCache,
    prompt_version: str = QUERY_REWRITING_PROMPT_VERSION,
) -> QueryRewriteResult:
    """Generate a HyDE-style hypothetical answer for retrieval, using a versioned cache."""
    key = QueryRewriteCache.make_key(question=question, strategy="hyde", model=model, prompt_version=prompt_version)
    cached = cache.get(key)
    if cached is not None:
        return _result(strategy="hyde", question=question, queries=cached, model=model, prompt_version=prompt_version, cache_hit=True)
    schema = HydeOutput.model_json_schema()
    call = llm_client.structured_chat(
        prompt=build_hyde_prompt(question, payload_schema=schema),
        model=model,
        payload_schema=schema,
        timeout_seconds=timeout_seconds,
    )
    output = HydeOutput.model_validate(call["structured"])
    queries = _clean_queries([output.hypothetical_answer])
    cache.set(key, queries)
    return _result(strategy="hyde", question=question, queries=queries, model=model, prompt_version=prompt_version, cache_hit=False)


def multi_query(
    *,
    llm_client: StructuredChatClient,
    question: str,
    model: str,
    timeout_seconds: int,
    cache: QueryRewriteCache,
    n: int = 3,
    prompt_version: str = QUERY_REWRITING_PROMPT_VERSION,
) -> QueryRewriteResult:
    """Generate exactly n query variants, using a versioned cache."""
    if n <= 0:
        raise ValueError("n must be positive")
    key = QueryRewriteCache.make_key(question=question, strategy="multi_query", model=model, prompt_version=prompt_version)
    cached = cache.get(key)
    if cached is not None:
        queries = _require_n_queries(cached, n=n)
        return _result(strategy="multi_query", question=question, queries=queries, model=model, prompt_version=prompt_version, cache_hit=True)
    schema = MultiQueryOutput.model_json_schema()
    call = llm_client.structured_chat(
        prompt=build_multi_query_prompt(question, n=n, payload_schema=schema),
        model=model,
        payload_schema=schema,
        timeout_seconds=timeout_seconds,
    )
    output = MultiQueryOutput.model_validate(call["structured"])
    queries = _require_n_queries(output.queries, n=n)
    cache.set(key, queries)
    return _result(strategy="multi_query", question=question, queries=queries, model=model, prompt_version=prompt_version, cache_hit=False)


def retrieve_query_variants(
    *,
    client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    query_texts: Sequence[str],
    limit: int,
    retrieval_mode: str,
    static_filters: dict[str, Any],
    rrf_k: int,
    index_manifest: dict[str, Any],
) -> list[RetrievedChunkRecord]:
    """Retrieve for multiple query variants and deduplicate candidates in first-seen order."""
    candidates: list[RetrievedChunkRecord] = []
    for query_text in _clean_queries(query_texts):
        candidates.extend(
            retrieve_direct(
                client=client,
                collection_name=collection_name,
                embedder=embedder,
                query_text=query_text,
                limit=limit,
                retrieval_mode=retrieval_mode,
                static_filters=static_filters,
                rrf_k=rrf_k,
                index_manifest=index_manifest,
            )
        )
    return dedupe_chunks(candidates)[:limit]


def _result(
    *,
    strategy: QueryRewriteStrategy,
    question: str,
    queries: Sequence[str],
    model: str,
    prompt_version: str,
    cache_hit: bool,
) -> QueryRewriteResult:
    return QueryRewriteResult(
        strategy=strategy,
        question=str(question or ""),
        queries=_clean_queries(queries),
        model=str(model or ""),
        prompt_version=str(prompt_version or ""),
        cache_hit=cache_hit,
    )


def _clean_queries(queries: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = str(query or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _require_n_queries(queries: Sequence[str], *, n: int) -> list[str]:
    cleaned = _clean_queries(queries)
    if len(cleaned) != n:
        raise ValueError(f"multi_query expected exactly {n} non-empty distinct queries, got {len(cleaned)}")
    return cleaned


__all__ = [
    "HydeOutput",
    "MultiQueryOutput",
    "QUERY_REWRITING_PROMPT_VERSION",
    "QueryRewriteCache",
    "QueryRewriteResult",
    "QueryRewriteStrategy",
    "RewriteOutput",
    "generate_hyde",
    "multi_query",
    "query_rewrite_cache_path",
    "retrieve_query_variants",
    "rewrite_query",
]

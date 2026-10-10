"""Independent dense and sparse representations with explicit query roles."""

import math
import os
import time
from importlib import metadata

import requests
from qdrant_client import models

from legal_rag.artifacts import digest
from legal_rag.contracts import ProviderConfig
from legal_rag.tokens import BGE_MODEL, BGE_REVISION
from .budget import BudgetLedger
from .errors import ProviderError

BGE_ADAPTER_SCHEMA_VERSION = "bge-adapter-v1"
BM25_ADAPTER_SCHEMA_VERSION = "bm25-adapter-v1"


def library_versions(*names: str) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def validate_dense(rows, count: int) -> list[list[float]]:
    vectors = [list(map(float, row)) for row in rows]
    if len(vectors) != count or not vectors:
        raise ValueError("Embedding count mismatch")
    size = len(vectors[0])
    if not size or any(len(row) != size or not all(math.isfinite(x) for x in row) or not any(row) for row in vectors):
        raise ValueError("Dense embeddings must be finite, nonzero and dimensionally consistent")
    return vectors


def validate_sparse(rows: list[models.SparseVector], count: int) -> list[models.SparseVector]:
    if len(rows) != count:
        raise ValueError("Sparse embedding count mismatch")
    for row in rows:
        if len(row.indices) != len(row.values) or len(set(row.indices)) != len(row.indices) or any(i < 0 for i in row.indices) or not all(math.isfinite(x) for x in row.values):
            raise ValueError("Invalid sparse vector")
    return rows


class BgeEmbedder:
    def __init__(self, *, batch_size: int = 2, local_files_only: bool = False, device: str = "cpu", dtype: str = "float32"):
        if device not in {"cpu", "mps"}:
            raise ValueError("Local embedding device must be cpu or mps")
        if dtype not in {"float32", "float16"} or (dtype == "float16" and device != "mps"):
            raise ValueError("float16 embeddings require an explicitly selected MPS device")
        self.batch_size, self.local_files_only = batch_size, local_files_only
        self.device, self.dtype = device, dtype
        self._model = None

    @property
    def identity(self) -> dict:
        return {"provider": "local", "model": BGE_MODEL, "revision": BGE_REVISION, "dimension": 1024,
                "device": self.device, "dtype": self.dtype, "padding_multiple": 32 if self.device == "mps" else None,
                "max_tokens": 512, "sparse": "bge-native", "idf": False,
                "adapter_schema_version": BGE_ADAPTER_SCHEMA_VERSION,
                "libraries": library_versions("FlagEmbedding", "transformers", "torch")}

    def preflight(self):
        if self.batch_size < 1:
            raise ValueError("Embedding batch_size must be positive")
        return self.identity

    def _load(self):
        if self._model is None:
            from huggingface_hub import snapshot_download
            from FlagEmbedding import BGEM3FlagModel
            path = snapshot_download(BGE_MODEL, revision=BGE_REVISION, local_files_only=self.local_files_only)
            self._model = BGEM3FlagModel(path, devices=self.device, use_fp16=self.dtype == "float16", batch_size=self.batch_size,
                                        query_max_length=512, passage_max_length=512)
        return self._model

    def embed_hybrid(self, texts: list[str], *, query: bool = False):
        try:
            model = self._load()
            if any(len(model.tokenizer.encode(text, add_special_tokens=True)) > 512 for text in texts):
                raise ValueError("Embedding input exceeds the 512-token contract; implicit truncation is forbidden")
            method = model.encode_queries if query else model.encode_corpus
            padding = {"pad_to_multiple_of": 32} if self.device == "mps" else {}
            encoded = method(texts, batch_size=self.batch_size, return_dense=True, return_sparse=True, return_colbert_vecs=False, **padding)
            dense = validate_dense(encoded["dense_vecs"], len(texts))
            sparse = [models.SparseVector(indices=[int(k) for k in sorted(row, key=int)], values=[float(row[k]) for k in sorted(row, key=int)]) for row in encoded["lexical_weights"]]
            return dense, validate_sparse(sparse, len(texts))
        except (RuntimeError, IndexError):
            raise ProviderError("local_inference_error", "Local embedding inference failed; no batch, device or provider fallback was used.") from None
        finally:
            if self.device == "mps":
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    def embed_documents(self, texts: list[str]):
        return self.embed_hybrid(texts)[0]

    def embed_queries(self, texts: list[str]):
        return self.embed_hybrid(texts, query=True)[0]


class BM25Embedder:
    def __init__(self, *, avg_len: float = 384, k: float = 1.2, b: float = 0.75):
        self.avg_len, self.k, self.b = avg_len, k, b
        self._model = None

    @property
    def identity(self) -> dict:
        return {"model": "Qdrant/bm25", "language": "italian", "idf": True, "avg_len": self.avg_len, "k": self.k, "b": self.b,
                "adapter_schema_version": BM25_ADAPTER_SCHEMA_VERSION, "libraries": library_versions("fastembed")}

    def _load(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding
            self._model = SparseTextEmbedding("Qdrant/bm25", language="italian", avg_len=self.avg_len, k=self.k, b=self.b)
        return self._model

    def fit(self, texts: list[str]) -> None:
        model = self._load()
        self.avg_len = max(1, model.token_count(texts) / len(texts))
        self._model = None

    def preflight(self):
        return self.identity

    def _encode(self, texts, *, query):
        model = self._load()
        rows = model.query_embed(texts) if query else model.embed(texts)
        vectors = [models.SparseVector(indices=[int(x) for x in row.indices], values=[float(x) for x in row.values]) for row in rows]
        return validate_sparse(vectors, len(texts))

    def embed_documents(self, texts):
        return self._encode(texts, query=False)

    def embed_queries(self, texts):
        return self._encode(texts, query=True)


class UtopiaEmbedder:
    def __init__(self, model: str, *, base_url="https://utopia.hpc4ai.unito.it/api", api_key=None, timeout=180, session=None):
        self.model, self.timeout = model, timeout
        self.url = base_url.rstrip("/").removesuffix("/api") + "/ollama/api/embed"
        self._key = api_key or os.getenv("UTOPIA_API_KEY", "")
        self._session = session or requests.Session()

    @property
    def identity(self):
        return {"provider": "utopia", "model": self.model, "endpoint": self.url, "sparse": None}

    def preflight(self):
        if not self._key:
            raise ProviderError("credentials_missing", "UTOPIA_API_KEY is missing.")
        return self.identity

    def embed_documents(self, texts):
        self.preflight()
        try:
            response = self._session.post(self.url, headers={"Authorization": f"Bearer {self._key}"},
                                          json={"model": self.model, "input": texts, "truncate": False}, timeout=self.timeout)
            if response.status_code >= 400:
                raise ProviderError(f"http_{response.status_code}", "Utopia embedding request failed.")
            return validate_dense(response.json()["embeddings"], len(texts))
        except ProviderError:
            raise
        except (requests.RequestException, ValueError, KeyError):
            raise ProviderError("embedding_error", "Utopia returned invalid embeddings or a transport error.") from None

    def embed_queries(self, texts):
        return self.embed_documents(texts)


class OpenAIEmbedder:
    def __init__(self, config: ProviderConfig, *, ledger_path="data/budget/openai.json", api_key=None, sdk=None):
        self.config, self._sdk = config, sdk
        self._key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.ledger = BudgetLedger(ledger_path, config)

    @property
    def identity(self):
        return {"provider": "openai", "model": self.config.model, "sparse": None}

    def preflight(self):
        if not self._key and self._sdk is None:
            raise ProviderError("credentials_missing", "OPENAI_API_KEY is missing.")
        self.ledger.preflight()
        return self.identity

    def embed_documents(self, texts):
        if not isinstance(texts, (list, tuple)) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ProviderError("embedding_input_invalid", "Embedding inputs must be a nonempty list of nonblank strings.")
        if len(texts) > 2048:
            raise ProviderError("embedding_batch_limit", "Embedding requests must contain at most 2048 inputs.")
        byte_bounds = [len(text.encode("utf-8")) for text in texts]
        if any(bound + 2048 > min(8192, self.config.max_input_tokens) for bound in byte_bounds):
            raise ProviderError("input_limit", "Embedding input exceeds its configured conservative input token bound.")
        batch_bound = sum(byte_bounds) + 2048
        if batch_bound > 300000:
            raise ProviderError("embedding_batch_limit", "Embedding request exceeds the conservative total token bound.")
        self.preflight()
        if self._sdk is None:
            if not self._key:
                raise ProviderError("credentials_missing", "OPENAI_API_KEY is missing.")
            from openai import OpenAI
            self._sdk = OpenAI(api_key=self._key, base_url="https://api.openai.com/v1", max_retries=0, timeout=self.config.timeout_seconds)
        bound = self.ledger.cost(batch_bound, 0)
        for attempt in range(min(self.config.retries, 2) + 1):
            reservation = self.ledger.reserve(bound, model=self.config.model, request_hash=digest("\n".join(texts)))
            try:
                response = self._sdk.embeddings.create(model=self.config.model, input=texts, encoding_format="float")
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                self.ledger.settle(reservation, 0.0 if status in {400, 401, 403, 404, 422, 429} else None)
                transient = status in {408, 429, 500, 502, 503, 504} or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}
                if transient and attempt < min(self.config.retries, 2):
                    time.sleep(min(2 ** attempt, 2))
                    continue
                raise ProviderError(f"http_{status}" if status else "embedding_error", "OpenAI embedding request failed.") from None
            cost = self.ledger.cost(response.usage.total_tokens, 0)
            self.ledger.settle(reservation, cost)
            if cost > bound + 1e-9:
                raise ProviderError("cost_bound_exceeded", "Embedding cost exceeded its reservation.")
            return validate_dense([row.embedding for row in sorted(response.data, key=lambda row: row.index)], len(texts))

    def embed_queries(self, texts):
        return self.embed_documents(texts)


def encode_vectors(embedder, texts: list[str], *, sparse_embedder=None, query=False):
    if sparse_embedder is None and hasattr(embedder, "embed_hybrid"):
        dense, sparse = embedder.embed_hybrid(texts, query=query)
        return validate_dense(dense, len(texts)), validate_sparse(sparse, len(texts))
    method = "embed_queries" if query else "embed_documents"
    dense = validate_dense(getattr(embedder, method)(texts), len(texts))
    sparse = getattr(sparse_embedder, method)(texts) if sparse_embedder is not None else None
    return dense, validate_sparse(sparse, len(texts)) if sparse is not None else None


def build_embedder(backend="local", model=None, **kwargs):
    if backend == "local":
        if model not in {None, BGE_MODEL}:
            raise ValueError("The local reference implementation supports pinned BGE-M3 only")
        return BgeEmbedder(**kwargs)
    if backend == "utopia":
        if not model:
            raise ValueError("An explicit Utopia embedding model is required")
        return UtopiaEmbedder(model, **kwargs)
    if backend == "openai":
        config = kwargs.pop("config", None)
        if config is None:
            raise ValueError("OpenAI embeddings require a ProviderConfig with explicit prices")
        return OpenAIEmbedder(config, **kwargs)
    raise ValueError(f"Unsupported embedding backend: {backend}")

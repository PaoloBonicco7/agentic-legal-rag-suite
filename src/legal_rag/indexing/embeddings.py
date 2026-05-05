"""Embedding adapters used by the Qdrant indexing step."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import requests

from legal_rag.oracle_context_evaluation.env import load_env_file

from .models import IndexingConfig

SparseVectorData = tuple[list[int], list[float]]


class SupportsEmbedding(Protocol):
    """Minimal dense embedding interface used by indexing and retrieval."""

    @property
    def model_name(self) -> str:
        """Return the backend model identifier."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of non-empty texts."""
        ...


class SupportsSparseEmbedding(SupportsEmbedding, Protocol):
    """Embedding interface for backends that also expose sparse vectors."""

    def embed_sparse_texts(self, texts: list[str]) -> list[SparseVectorData]:
        """Embed texts as sparse vectors."""
        ...


def supports_sparse_embedding(embedder: SupportsEmbedding) -> bool:
    """Return whether the embedder exposes sparse vectors."""
    return callable(getattr(embedder, "embed_sparse_texts", None))


def _clean_texts(texts: list[str]) -> list[str]:
    cleaned = [str(text or "").strip() for text in texts]
    if not all(cleaned):
        raise ValueError("Cannot embed empty text items")
    return cleaned


def _validate_dense_vectors(vectors: Any, expected: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise RuntimeError(f"Embedding backend returned {len(vectors)} vectors, expected {expected}")
    out: list[list[float]] = []
    for idx, vector in enumerate(vectors):
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"Embedding vector at index {idx} is empty or not a list")
        out.append([float(value) for value in values])
    return out


def _validate_sparse_vectors(vectors: list[SparseVectorData], expected: int) -> list[SparseVectorData]:
    if len(vectors) != expected:
        raise RuntimeError(f"Sparse embedding backend returned {len(vectors)} vectors, expected {expected}")
    for idx, (indices, values) in enumerate(vectors):
        if len(indices) != len(values):
            raise RuntimeError(f"Sparse vector at index {idx} has mismatched indices and values")
        if len(set(indices)) != len(indices):
            raise RuntimeError(f"Sparse vector at index {idx} has duplicate indices")
    return vectors


def _sparse_from_mapping(mapping: Any) -> SparseVectorData:
    if not isinstance(mapping, dict):
        return ([], [])
    pairs: list[tuple[int, float]] = []
    for key, value in mapping.items():
        try:
            weight = float(value)
            if weight:
                pairs.append((int(key), weight))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda item: item[0])
    return ([idx for idx, _ in pairs], [weight for _, weight in pairs])


@dataclass
class LocalEmbeddingBackend:
    """Local embedding backend for dense and optional sparse vectors."""

    model: str
    hybrid_enabled: bool = True
    use_fp16: bool = True
    _bge_model: Any = None
    _dense_model: Any = None
    _sparse_model: Any = None

    def __post_init__(self) -> None:
        if self.hybrid_enabled and self.model.lower() == "baai/bge-m3":
            try:
                from FlagEmbedding import BGEM3FlagModel
            except Exception as exc:  # pragma: no cover - environment-specific dependency.
                raise RuntimeError(
                    "Local hybrid indexing with BAAI/bge-m3 requires FlagEmbedding. "
                    "Install project dependencies before running full indexing."
                ) from exc
            self._bge_model = BGEM3FlagModel(self.model, use_fp16=self.use_fp16)
            return

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - environment-specific dependency.
            raise RuntimeError(
                "Local dense indexing requires sentence-transformers. "
                "Install project dependencies or set embedding_backend='utopia'."
            ) from exc
        self._dense_model = SentenceTransformer(self.model)

        if self.hybrid_enabled:
            try:
                from fastembed import SparseTextEmbedding
            except Exception as exc:  # pragma: no cover - environment-specific dependency.
                raise RuntimeError(
                    "Hybrid indexing for non-BGE local models requires qdrant-client[fastembed]. "
                    "Disable hybrid_enabled or install the optional fastembed dependency."
                ) from exc
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    @property
    def model_name(self) -> str:
        return self.model

    def _encode_bge(self, texts: list[str], *, sparse: bool) -> Any:
        return self._bge_model.encode(texts, return_dense=not sparse, return_sparse=sparse)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = _clean_texts(texts)
        if self._bge_model is not None:
            encoded = self._encode_bge(cleaned, sparse=False)
            vectors = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
            return _validate_dense_vectors(vectors, len(cleaned))
        vectors = self._dense_model.encode(cleaned, normalize_embeddings=True)
        return _validate_dense_vectors(vectors, len(cleaned))

    def embed_sparse_texts(self, texts: list[str]) -> list[SparseVectorData]:
        cleaned = _clean_texts(texts)
        if self._bge_model is not None:
            encoded = self._encode_bge(cleaned, sparse=True)
            lexical_weights = encoded.get("lexical_weights") if isinstance(encoded, dict) else None
            return _validate_sparse_vectors([_sparse_from_mapping(row) for row in lexical_weights or []], len(cleaned))
        if self._sparse_model is None:
            raise RuntimeError(f"Embedding model {self.model!r} does not expose sparse vectors")
        out: list[SparseVectorData] = []
        for vector in self._sparse_model.embed(cleaned):
            indices = list(getattr(vector, "indices", []) or [])
            values = list(getattr(vector, "values", []) or [])
            out.append(([int(idx) for idx in indices], [float(value) for value in values]))
        return _validate_sparse_vectors(out, len(cleaned))


@dataclass
class UtopiaOllamaEmbedder:
    """HTTP embedder for Utopia's Ollama-compatible embedding endpoint."""

    api_key: str
    model: str
    embed_url: str
    timeout_seconds: float = 60.0
    _active_url: str | None = None
    _active_variant: str | None = None

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def active_endpoint(self) -> str:
        return self._active_url or self.embed_url

    @property
    def active_variant(self) -> str:
        return self._active_variant or "unresolved"

    def _candidate_urls(self) -> list[str]:
        urls = [self.embed_url.strip()]
        if self.embed_url.rstrip("/").endswith("/embeddings"):
            urls.append(self.embed_url.rstrip("/")[: -len("/embeddings")] + "/embed")
        elif self.embed_url.rstrip("/").endswith("/embed"):
            urls.append(self.embed_url.rstrip("/")[: -len("/embed")] + "/embeddings")
        return [url for idx, url in enumerate(urls) if url and url not in urls[:idx]]

    def _post_json(self, url: str, payload: dict[str, Any]) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return requests.post(url, headers=headers, json=payload, timeout=max(1.0, float(self.timeout_seconds)))

    def _parse_json(self, response: requests.Response, *, url: str, variant: str) -> dict[str, Any]:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Utopia embedding request failed ({variant}): "
                f"url={url!r} http={response.status_code} body={response.text[:300]!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Utopia embedding endpoint did not return JSON ({variant}). "
                f"url={url!r} http={response.status_code} body={response.text[:300]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Utopia embedding endpoint returned non-object JSON ({variant}): {type(payload).__name__}")
        return payload

    def _embeddings_from_payload(self, payload: dict[str, Any], expected: int) -> list[list[float]] | None:
        if expected == 1 and isinstance(payload.get("embedding"), list):
            return [payload["embedding"]]
        rows = payload.get("embeddings")
        if isinstance(rows, list):
            if rows and isinstance(rows[0], dict) and "embedding" in rows[0]:
                return [row.get("embedding") for row in rows]
            return rows
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = _clean_texts(texts)
        errors: list[str] = []

        if self._active_url and self._active_variant:
            payload = {"model": self.model, "input": cleaned} if self._active_variant == "input_list" else {"model": self.model, "prompt": cleaned}
            try:
                data = self._parse_json(self._post_json(self._active_url, payload), url=self._active_url, variant=self._active_variant)
                vectors = self._embeddings_from_payload(data, len(cleaned))
                if vectors is not None:
                    return _validate_dense_vectors(vectors, len(cleaned))
            except Exception as exc:
                errors.append(f"{self._active_url} {self._active_variant}: {exc}")
                self._active_url = None
                self._active_variant = None

        for url in self._candidate_urls():
            for variant, payload in (
                ("input_list", {"model": self.model, "input": cleaned}),
                ("prompt_list", {"model": self.model, "prompt": cleaned}),
            ):
                try:
                    data = self._parse_json(self._post_json(url, payload), url=url, variant=variant)
                    vectors = self._embeddings_from_payload(data, len(cleaned))
                    if vectors is not None:
                        self._active_url = url
                        self._active_variant = variant
                        return _validate_dense_vectors(vectors, len(cleaned))
                    errors.append(f"{url} {variant}: unexpected response keys={sorted(data.keys())}")
                except Exception as exc:
                    errors.append(f"{url} {variant}: {exc}")

        vectors: list[list[float]] = []
        for text in cleaned:
            try:
                url = self._candidate_urls()[0]
                data = self._parse_json(self._post_json(url, {"model": self.model, "prompt": text}), url=url, variant="prompt_single")
                single = self._embeddings_from_payload(data, 1)
                if single is None:
                    raise RuntimeError(f"unexpected response keys={sorted(data.keys())}")
                vectors.extend(single)
                self._active_url = url
                self._active_variant = "prompt_single"
            except Exception as exc:
                errors.append(f"prompt_single: {exc}")
                raise RuntimeError(
                    "Utopia embedding failed. Tried batched input, batched prompt, and single prompt variants. "
                    f"model={self.model!r} url={self.embed_url!r} errors={errors[-5:]}"
                ) from exc
        return _validate_dense_vectors(vectors, len(cleaned))


def _root_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def discover_utopia_models(config: IndexingConfig, *, timeout_seconds: float | None = None) -> dict[str, Any]:
    """Return a small model catalog diagnostic from Utopia's Ollama tags endpoint."""
    load_env_file(config.env_file)
    root = _root_url(config.resolved_utopia_base_url) or _root_url(config.resolved_utopia_embed_url)
    tags_url = f"{root}/ollama/api/tags" if root else ""
    out: dict[str, Any] = {"ok": False, "root_url": root, "base_url": config.resolved_utopia_base_url, "tags_url": tags_url, "models": [], "embedding_like_models": []}
    if not tags_url:
        out["error"] = "Cannot derive Utopia root URL"
        return out
    try:
        response = requests.get(
            tags_url,
            headers={"Authorization": f"Bearer {config.embedding_api_key or os.getenv('UTOPIA_API_KEY', '')}"},
            timeout=max(1.0, timeout_seconds or min(3.0, config.embedding_timeout_seconds)),
        )
    except Exception as exc:
        out["error"] = f"tags request failed: {exc}"
        return out
    out["status_code"] = response.status_code
    if response.status_code >= 400:
        out["error"] = f"tags request returned HTTP {response.status_code}: {response.text[:300]!r}"
        return out
    try:
        payload = response.json()
    except ValueError:
        out["error"] = f"tags response was not JSON: {response.text[:300]!r}"
        return out
    rows = payload.get("models")
    if not isinstance(rows, list):
        out["error"] = "tags response missing models list"
        return out
    models = sorted({str(row.get("name") or "").strip() for row in rows if isinstance(row, dict) and row.get("name")})
    out["models"] = models
    out["embedding_like_models"] = [model for model in models if "embed" in model.lower()]
    out["count"] = len(models)
    out["ok"] = True
    return out


def _build_utopia_embedder(config: IndexingConfig) -> UtopiaOllamaEmbedder:
    load_env_file(config.env_file)
    api_key = config.embedding_api_key or os.getenv("UTOPIA_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("UTOPIA_API_KEY is missing. Set environment variable, .env, or IndexingConfig.embedding_api_key.")
    if config.hybrid_enabled:
        raise RuntimeError("Utopia embedding backend is dense-only in this step; set hybrid_enabled=False or use embedding_backend='local'.")
    return UtopiaOllamaEmbedder(
        api_key=api_key,
        model=config.resolved_embedding_model,
        embed_url=config.resolved_utopia_embed_url,
        timeout_seconds=config.embedding_timeout_seconds,
    )


def build_embedder(config: IndexingConfig) -> SupportsEmbedding:
    """Build the configured embedding adapter."""
    if config.embedding_backend == "local":
        return LocalEmbeddingBackend(model=config.resolved_embedding_model, hybrid_enabled=config.hybrid_enabled)
    if config.embedding_backend == "utopia":
        return _build_utopia_embedder(config)
    raise RuntimeError(f"Unsupported embedding_backend={config.embedding_backend!r}")


def debug_utopia_embedding_connection(config: IndexingConfig, probe_text: str = "Verifica connessione embedding Utopia.") -> dict[str, Any]:
    """Run a single Utopia embedding probe and return diagnostics."""
    started = perf_counter()
    diagnostics: dict[str, Any] = {
        "backend": "utopia",
        "model": config.resolved_embedding_model,
        "base_url": config.resolved_utopia_base_url,
        "embed_url": config.resolved_utopia_embed_url,
        "success": False,
    }
    diagnostics["model_catalog"] = discover_utopia_models(config)
    try:
        probe_config = config.model_copy(update={"embedding_backend": "utopia", "hybrid_enabled": False})
        embedder = _build_utopia_embedder(probe_config)
        vectors = embedder.embed_texts([probe_text])
        diagnostics["resolved_embedder"] = type(embedder).__name__
        diagnostics["vector_size"] = len(vectors[0])
        if isinstance(embedder, UtopiaOllamaEmbedder):
            diagnostics["active_endpoint"] = embedder.active_endpoint
            diagnostics["active_variant"] = embedder.active_variant
        diagnostics["success"] = True
        diagnostics["latency_ms"] = round((perf_counter() - started) * 1000.0, 2)
        return diagnostics
    except Exception as exc:
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error"] = str(exc)
        diagnostics["latency_ms"] = round((perf_counter() - started) * 1000.0, 2)
        raise RuntimeError("Utopia embedding debug check failed: " + json.dumps(diagnostics, ensure_ascii=False)) from exc

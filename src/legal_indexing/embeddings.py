from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import requests

from .settings import IndexingConfig

try:  # pragma: no cover - resolved at runtime/tests
    from langchain_openai import OpenAIEmbeddings as OpenAIEmbeddings
except Exception:  # pragma: no cover - fail-soft, used only when OpenAI path is selected
    OpenAIEmbeddings = None  # type: ignore[assignment]


class SupportsEmbedding(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class LangChainUtopiaEmbedder:
    api_key: str
    model: str
    base_url: str
    batch_size: int = 32
    timeout_seconds: float = 60.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        openai_embeddings_cls = OpenAIEmbeddings
        if openai_embeddings_cls is None:
            raise RuntimeError(
                "langchain-openai is not available for OpenAI-compatible embeddings"
            )

        try:
            self._client = openai_embeddings_cls(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                chunk_size=max(1, int(self.batch_size)),
                request_timeout=max(1.0, float(self.timeout_seconds)),
                max_retries=max(0, int(self.max_retries)),
                tiktoken_enabled=True,
                # Utopia model ids are custom; force known tokenizer mapping.
                tiktoken_model_name="text-embedding-3-small",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize OpenAI-compatible embedding client "
                f"(model={self.model!r}, base_url={self.base_url!r}): {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        return self.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [str(t or "").strip() for t in texts]
        if not all(cleaned):
            raise ValueError("Cannot embed empty text items")

        try:
            vectors = self._client.embed_documents(cleaned)
        except Exception as exc:
            raise RuntimeError(
                "Embedding request via LangChain OpenAIEmbeddings failed "
                f"(model={self.model!r}, base_url={self.base_url!r}): {exc}"
            ) from exc

        if len(vectors) != len(cleaned):
            raise RuntimeError(
                "Embedding backend returned unexpected number of vectors "
                f"(expected={len(cleaned)}, got={len(vectors)})"
            )

        for i, vec in enumerate(vectors):
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(f"Embedding vector at index {i} is empty or not a list")

        return vectors


@dataclass
class UtopiaOllamaEmbedder:
    api_key: str
    model: str
    embed_url: str
    timeout_seconds: float = 60.0
    _last_variant: str | None = None

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def last_variant(self) -> str:
        return self._last_variant or "unresolved"

    def _embeddings_url(self) -> str:
        base = self.embed_url.strip()
        if base.endswith("/embed"):
            return base[: -len("/embed")] + "/embeddings"
        return base

    def _post_json(self, url: str, payload: dict[str, Any]) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=max(1.0, float(self.timeout_seconds)),
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [str(t or "").strip() for t in texts]
        if not all(cleaned):
            raise ValueError("Cannot embed empty text items")

        errors: list[str] = []

        # Variant 1: Ollama /api/embed with batch "input" list.
        try:
            response = self._post_json(
                self.embed_url,
                {"model": self.model, "input": cleaned},
            )
            if response.status_code < 400:
                data = response.json()
                vectors = data.get("embeddings")
                if isinstance(vectors, list) and len(vectors) == len(cleaned):
                    self._last_variant = "embed_input_list"
                    return vectors
                errors.append("embed_input_list invalid response shape")
            else:
                body = (response.text or "").strip().replace("\n", " ")[:500]
                errors.append(f"embed_input_list http={response.status_code} body={body!r}")
        except Exception as exc:
            errors.append(f"embed_input_list exception={exc}")

        # Variant 2: Ollama /api/embed single input (sequential).
        sequential_vectors: list[list[float]] = []
        try:
            for text in cleaned:
                response = self._post_json(
                    self.embed_url,
                    {"model": self.model, "input": text},
                )
                if response.status_code >= 400:
                    body = (response.text or "").strip().replace("\n", " ")[:500]
                    raise RuntimeError(
                        f"http={response.status_code} body={body!r}"
                    )
                data = response.json()
                vectors = data.get("embeddings")
                if not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], list):
                    raise RuntimeError("invalid response shape")
                sequential_vectors.append(vectors[0])
            self._last_variant = "embed_input_single"
            return sequential_vectors
        except Exception as exc:
            errors.append(f"embed_input_single {exc}")

        # Variant 3: legacy Ollama /api/embeddings with prompt.
        legacy_url = self._embeddings_url()
        legacy_vectors: list[list[float]] = []
        try:
            for text in cleaned:
                response = self._post_json(
                    legacy_url,
                    {"model": self.model, "prompt": text},
                )
                if response.status_code >= 400:
                    body = (response.text or "").strip().replace("\n", " ")[:500]
                    raise RuntimeError(
                        f"http={response.status_code} body={body!r}"
                    )
                data = response.json()
                vec = data.get("embedding")
                if not isinstance(vec, list) or not vec:
                    raise RuntimeError("invalid response shape")
                legacy_vectors.append(vec)
            self._last_variant = "embeddings_prompt_single"
            return legacy_vectors
        except Exception as exc:
            errors.append(f"embeddings_prompt_single {exc}")

        raise RuntimeError(
            "Embedding request via Utopia Ollama endpoint failed on all payload variants "
            f"(model={self.model!r}, embed_url={self.embed_url!r}, errors={errors})"
        )


@dataclass
class AutoUtopiaEmbedder:
    openai_embedder: LangChainUtopiaEmbedder
    ollama_embedder: UtopiaOllamaEmbedder
    _active_mode: str | None = None
    _active_detail: str | None = None

    @property
    def model_name(self) -> str:
        return self.openai_embedder.model_name

    @property
    def active_mode(self) -> str:
        return self._active_mode or "unresolved"

    @property
    def active_detail(self) -> str:
        return self._active_detail or "unresolved"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._active_mode == "openai":
            return self.openai_embedder.embed_texts(texts)
        if self._active_mode == "ollama":
            return self.ollama_embedder.embed_texts(texts)

        try:
            vectors = self.openai_embedder.embed_texts(texts)
            self._active_mode = "openai"
            self._active_detail = "openai_embeddings"
            return vectors
        except Exception as openai_exc:
            try:
                vectors = self.ollama_embedder.embed_texts(texts)
                self._active_mode = "ollama"
                self._active_detail = f"ollama_{self.ollama_embedder.last_variant}"
                return vectors
            except Exception as ollama_exc:
                raise RuntimeError(
                    "Utopia embedding failed on both OpenAI-compatible and Ollama-compatible paths. "
                    f"openai_error={openai_exc}; ollama_error={ollama_exc}"
                ) from ollama_exc


def _derive_ollama_embed_url(base_url: str) -> str:
    raw = (base_url or "").strip()
    if not raw:
        return ""

    if raw.endswith("/ollama/api/embed"):
        return raw

    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return ""

    root = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    return f"{root}/ollama/api/embed"


def _utopia_root(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def discover_utopia_models(
    config: IndexingConfig,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    root = _utopia_root(config.utopia_base_url) or _utopia_root(config.utopia_embed_url)
    tags_url = f"{root}/ollama/api/tags" if root else ""
    out: dict[str, Any] = {
        "ok": False,
        "root_url": root,
        "tags_url": tags_url,
        "models": [],
        "embedding_like_models": [],
    }
    if not root:
        out["error"] = "Cannot derive root URL from utopia_base_url/utopia_embed_url"
        return out

    headers = {"Authorization": f"Bearer {config.embedding_api_key}"}
    timeout = timeout_seconds if timeout_seconds is not None else min(
        3.0, float(config.embedding_timeout_seconds)
    )
    try:
        resp = requests.get(tags_url, headers=headers, timeout=max(1.0, timeout))
    except Exception as exc:
        out["error"] = f"tags request failed: {exc}"
        return out

    out["status_code"] = int(resp.status_code)
    if resp.status_code >= 400:
        out["error"] = f"tags request returned HTTP {resp.status_code}: {resp.text[:300]!r}"
        return out

    try:
        data = resp.json()
    except Exception as exc:
        out["error"] = f"tags response is not JSON: {exc}"
        return out

    rows = data.get("models")
    if not isinstance(rows, list):
        out["error"] = "tags response missing models list"
        return out

    models: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            name = row.get("name")
            if isinstance(name, str) and name.strip():
                models.append(name.strip())
    models = sorted(set(models))
    out["models"] = models
    out["embedding_like_models"] = [m for m in models if "embed" in m.lower()]
    out["ok"] = True
    out["count"] = len(models)
    return out


def _candidate_embedding_models(requested_model: str, catalog: dict[str, Any]) -> list[str]:
    requested = (requested_model or "").strip()
    candidates: list[str] = []
    if requested:
        candidates.append(requested)

    if requested.startswith("SLURM."):
        candidates.append(requested[len("SLURM.") :])
    else:
        candidates.append(f"SLURM.{requested}")
    if requested.endswith(":latest"):
        candidates.append(requested[: -len(":latest")])
    if requested and ":" not in requested:
        candidates.append(f"{requested}:latest")

    if catalog.get("ok"):
        models = [str(m) for m in catalog.get("models", [])]
        embed_models = [str(m) for m in catalog.get("embedding_like_models", [])]
        if requested and requested not in models:
            requested_norm = requested.lower().replace("slurm.", "").replace(":latest", "")
            for model in embed_models:
                m_norm = model.lower().replace("slurm.", "").replace(":latest", "")
                if requested_norm and (requested_norm == m_norm or requested_norm in m_norm):
                    candidates.append(model)
                    if not model.startswith("SLURM."):
                        candidates.append(f"SLURM.{model}")
            candidates.extend(embed_models[:5])
            candidates.extend(
                [f"SLURM.{m}" for m in embed_models[:5] if not str(m).startswith("SLURM.")]
            )
        else:
            candidates.extend(embed_models[:5])
            candidates.extend(
                [f"SLURM.{m}" for m in embed_models[:5] if not str(m).startswith("SLURM.")]
            )

    # De-duplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out[:16]


def _require_utopia_config(config: IndexingConfig) -> None:
    if not (config.embedding_provider or "").strip():
        raise RuntimeError("embedding_provider is missing")
    if (config.embedding_provider or "").strip().lower() != "utopia":
        raise RuntimeError(
            f"Unsupported embedding_provider={config.embedding_provider!r}. "
            "Only 'utopia' is supported in runtime."
        )
    if not (config.embedding_api_key or "").strip():
        raise RuntimeError(
            "UTOPIA_API_KEY is missing. Set environment variable or IndexingConfig.embedding_api_key."
        )
    if not (config.embedding_model or "").strip():
        raise RuntimeError("embedding_model is missing")
    mode = (config.utopia_embed_api_mode or "").strip().lower()
    if mode not in {"openai", "ollama", "auto"}:
        raise RuntimeError(
            f"Unsupported utopia_embed_api_mode={config.utopia_embed_api_mode!r}. "
            "Use one of: 'openai', 'ollama', 'auto'."
        )

    if mode in {"openai", "auto"} and not (config.utopia_base_url or "").strip():
        raise RuntimeError(
            "UTOPIA_BASE_URL is missing. Set environment variable or IndexingConfig.utopia_base_url."
        )
    if mode in {"ollama", "auto"}:
        embed_url = (config.utopia_embed_url or "").strip() or _derive_ollama_embed_url(
            config.utopia_base_url
        )
        if not embed_url:
            raise RuntimeError(
                "UTOPIA_EMBED_URL is missing and cannot be derived from UTOPIA_BASE_URL."
            )


def build_embedder(config: IndexingConfig) -> SupportsEmbedding:
    _require_utopia_config(config)
    mode = (config.utopia_embed_api_mode or "auto").strip().lower()

    ollama_embedder = UtopiaOllamaEmbedder(
        api_key=config.embedding_api_key,
        model=config.embedding_model,
        embed_url=(config.utopia_embed_url or "").strip()
        or _derive_ollama_embed_url(config.utopia_base_url),
        timeout_seconds=max(1.0, float(config.embedding_timeout_seconds)),
    )

    if mode == "ollama":
        return ollama_embedder

    openai_embedder: LangChainUtopiaEmbedder | None = None
    openai_init_error: Exception | None = None
    try:
        openai_embedder = LangChainUtopiaEmbedder(
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            base_url=config.utopia_base_url,
            batch_size=max(1, int(config.embedding_batch_size)),
            timeout_seconds=max(1.0, float(config.embedding_timeout_seconds)),
        )
    except Exception as exc:
        openai_init_error = exc

    if mode == "openai":
        if openai_embedder is None:
            raise RuntimeError(
                "OpenAI-compatible embedding mode is selected but initialization failed. "
                f"Set UTOPIA_EMBED_API_MODE=ollama or fix openai/langchain-openai versions. "
                f"error={openai_init_error}"
            ) from openai_init_error
        return openai_embedder

    # mode == auto:
    if openai_embedder is None:
        # In auto mode, fail-soft to Ollama-compatible endpoint if OpenAI-compatible
        # stack is unavailable or incompatible.
        return ollama_embedder

    return AutoUtopiaEmbedder(
        openai_embedder=openai_embedder,
        ollama_embedder=ollama_embedder,
    )


def debug_utopia_embedding_connection(
    config: IndexingConfig,
    probe_text: str = "Verifica connessione embedding Utopia.",
) -> dict[str, Any]:
    started = perf_counter()
    diagnostics: dict[str, Any] = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "configured_mode": config.utopia_embed_api_mode,
        "base_url": config.utopia_base_url,
        "embed_url": (config.utopia_embed_url or "").strip()
        or _derive_ollama_embed_url(config.utopia_base_url),
        "batch_size": int(config.embedding_batch_size),
        "timeout_seconds": float(config.embedding_timeout_seconds),
        "success": False,
    }

    catalog = discover_utopia_models(config)
    diagnostics["model_catalog"] = {
        "ok": bool(catalog.get("ok")),
        "count": int(catalog.get("count") or 0),
        "status_code": catalog.get("status_code"),
        "embedding_like_models": list(catalog.get("embedding_like_models", []))[:20],
        "error": catalog.get("error"),
    }
    candidates = _candidate_embedding_models(config.embedding_model, catalog)
    diagnostics["model_candidates"] = candidates

    attempt_errors: dict[str, str] = {}
    try:
        for model_name in (candidates or [config.embedding_model]):
            cfg = config.with_overrides(embedding_model=model_name)
            try:
                embedder = build_embedder(cfg)
                diagnostics["resolved_embedder"] = type(embedder).__name__
                vectors = embedder.embed_texts([probe_text])
                vector_size = len(vectors[0]) if vectors and vectors[0] else 0
                diagnostics["vector_size"] = int(vector_size)
                if isinstance(embedder, AutoUtopiaEmbedder):
                    diagnostics["active_mode"] = embedder.active_mode
                    diagnostics["active_detail"] = embedder.active_detail
                diagnostics["resolved_model"] = model_name
                diagnostics["success"] = True
                if model_name == config.embedding_model:
                    diagnostics["message"] = "Embedding probe completed successfully"
                else:
                    diagnostics["message"] = (
                        "Embedding probe succeeded with fallback model "
                        f"{model_name!r} (requested={config.embedding_model!r})"
                    )
                diagnostics["latency_ms"] = round((perf_counter() - started) * 1000.0, 2)
                return diagnostics
            except Exception as exc:
                attempt_errors[model_name] = str(exc)
                continue

        diagnostics["attempt_errors"] = attempt_errors
        raise RuntimeError(
            "All embedding probe attempts failed. "
            f"attempts={len(attempt_errors)}"
        )
    except Exception as exc:
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error"] = str(exc)
        diagnostics["latency_ms"] = round((perf_counter() - started) * 1000.0, 2)
        raise RuntimeError(
            "Utopia embedding debug check failed: "
            + json.dumps(diagnostics, ensure_ascii=False)
        ) from exc


__all__ = [
    "SupportsEmbedding",
    "LangChainUtopiaEmbedder",
    "UtopiaOllamaEmbedder",
    "AutoUtopiaEmbedder",
    "discover_utopia_models",
    "build_embedder",
    "debug_utopia_embedding_connection",
]

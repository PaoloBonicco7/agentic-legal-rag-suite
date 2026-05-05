"""Minimal structured chat client compatible with Utopia/Ollama endpoints."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse


class StructuredChatClient(Protocol):
    """Protocol for injectable structured chat clients."""

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Call a model and return the parsed structured object."""
        ...


def _validate_http_url(url: str, *, field_name: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http(s) URL: {url!r}")
    return parsed.geturl().rstrip("/")


def resolve_ollama_chat_url(base_url: str, explicit_url: str | None = None) -> str:
    """Resolve an Ollama-compatible chat endpoint from config values."""
    if explicit_url is not None and explicit_url.strip():
        return _validate_http_url(explicit_url, field_name="explicit_url")
    normalized_base = _validate_http_url(base_url, field_name="base_url")
    if normalized_base.endswith("/api"):
        normalized_base = normalized_base[:-4]
    return f"{normalized_base}/ollama/api/chat"


def resolve_openai_chat_completions_url(base_url: str, explicit_url: str | None = None) -> str:
    """Resolve an OpenAI-compatible chat completions endpoint from config values."""
    if explicit_url is not None and explicit_url.strip():
        return _validate_http_url(explicit_url, field_name="explicit_url")
    normalized_base = _validate_http_url(base_url, field_name="base_url")
    return f"{normalized_base}/chat/completions"


def resolve_ollama_root_url(url: str) -> str:
    """Resolve the Utopia/Ollama root URL from a base or endpoint URL."""
    normalized = _validate_http_url(url, field_name="url")
    parsed = urlparse(normalized)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")


def discover_utopia_models(
    *,
    base_url: str | None = None,
    api_url: str | None = None,
    api_key: str,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    """Fetch the Ollama model catalog from Utopia without exposing credentials."""
    root = resolve_ollama_root_url(api_url or base_url or "")
    tags_url = f"{root}/ollama/api/tags"
    out: dict[str, Any] = {
        "ok": False,
        "root_url": root,
        "tags_url": tags_url,
        "models": [],
        "chat_like_models": [],
        "embedding_like_models": [],
    }
    try:
        import requests
    except Exception as exc:  # pragma: no cover - dependency error path
        out["error"] = f"requests is required for model discovery: {exc}"
        return out

    try:
        response = requests.get(
            tags_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=max(1.0, float(timeout_seconds)),
        )
    except Exception as exc:  # pragma: no cover - network behavior
        out["error"] = f"tags request failed: {type(exc).__name__}: {exc}"
        return out

    out["status_code"] = int(response.status_code)
    if response.status_code >= 400:
        out["error"] = f"tags request returned HTTP {response.status_code}: {response.text[:300]!r}"
        return out
    try:
        data = response.json()
    except Exception as exc:
        out["error"] = f"tags response is not JSON: {type(exc).__name__}: {exc}"
        return out

    rows = data.get("models")
    if not isinstance(rows, list):
        out["error"] = "tags response missing models list"
        return out

    models = sorted(
        {
            str(row.get("name")).strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        }
    )
    out["models"] = models
    out["embedding_like_models"] = [model for model in models if "embed" in model.lower()]
    out["chat_like_models"] = [model for model in models if "embed" not in model.lower()]
    out["count"] = len(models)
    out["ok"] = True
    return out


def discover_utopia_api_models(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    """Fetch Utopia OpenAI-compatible preset models from `/api/models`."""
    normalized_base = _validate_http_url(base_url, field_name="base_url")
    models_url = f"{normalized_base}/models"
    out: dict[str, Any] = {
        "ok": False,
        "models_url": models_url,
        "models": [],
        "base_models": [],
        "all_models": [],
        "chat_like_models": [],
    }
    try:
        import requests
    except Exception as exc:  # pragma: no cover - dependency error path
        out["error"] = f"requests is required for model discovery: {exc}"
        return out
    try:
        response = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=max(1.0, float(timeout_seconds)),
        )
    except Exception as exc:  # pragma: no cover - network behavior
        out["error"] = f"models request failed: {type(exc).__name__}: {exc}"
        return out
    out["status_code"] = int(response.status_code)
    if response.status_code >= 400:
        out["error"] = f"models request returned HTTP {response.status_code}: {response.text[:300]!r}"
        return out
    try:
        data = response.json()
    except Exception as exc:
        out["error"] = f"models response is not JSON: {type(exc).__name__}: {exc}"
        return out
    rows = data.get("data")
    if not isinstance(rows, list):
        out["error"] = "models response missing data list"
        return out
    models = sorted(
        {
            str(row.get("id")).strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
    )
    base_models = sorted(
        {
            str(row.get("info", {}).get("base_model_id")).strip()
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("info"), dict)
            and str(row.get("info", {}).get("base_model_id") or "").strip()
        }
    )
    out["models"] = models
    out["base_models"] = base_models
    out["all_models"] = sorted(set(models) | set(base_models))
    out["chat_like_models"] = [
        str(row.get("id")).strip()
        for row in rows
        if isinstance(row, dict)
        and str(row.get("id") or "").strip()
        and any(str(tag.get("name") or "").upper() == "CHAT" for tag in row.get("tags", []) if isinstance(tag, dict))
    ]
    out["count"] = len(models)
    out["ok"] = True
    return out


def parse_structured_content(response_json: dict[str, Any]) -> dict[str, Any]:
    """Extract a structured JSON object from an Ollama chat response."""
    raw_content = response_json.get("message", {}).get("content")
    if raw_content is None:
        raise KeyError("response_json['message']['content'] is missing")
    if isinstance(raw_content, str):
        data = json.loads(raw_content)
    elif isinstance(raw_content, dict):
        data = raw_content
    else:
        raise TypeError(f"Unexpected message.content type: {type(raw_content).__name__}")
    if not isinstance(data, dict):
        raise TypeError(f"Structured content must be an object, got {type(data).__name__}")
    return data


def parse_openai_structured_content(response_json: dict[str, Any]) -> dict[str, Any]:
    """Extract a structured JSON object from an OpenAI-compatible chat response."""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise KeyError("response_json['choices'][0] is missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise KeyError("response_json['choices'][0]['message'] is missing")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise TypeError(f"Unexpected message.content type: {type(content).__name__}")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise TypeError(f"Structured content must be an object, got {type(data).__name__}")
    return data


class UtopiaStructuredChatClient:
    """Small HTTP client using the structured chat pattern from OLD."""

    def __init__(self, *, api_url: str, api_key: str, retry_attempts: int = 1) -> None:
        self.api_url = resolve_ollama_chat_url(api_url) if "/ollama/api/chat" not in api_url else api_url.rstrip("/")
        self.api_key = api_key
        self.retry_attempts = max(1, int(retry_attempts))

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Call the configured endpoint and parse the structured response."""
        try:
            import requests
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("requests is required for remote structured chat calls") from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": payload_schema,
            "options": {"temperature": 0},
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout_seconds)
                if not response.ok:
                    hint = ""
                    if response.status_code == 400 and "not found" in response.text.lower():
                        hint = (
                            " Check UTOPIA_CHAT_MODEL/UTOPIA_JUDGE_MODEL against "
                            "the Utopia Ollama catalog at /ollama/api/tags."
                        )
                    raise RuntimeError(f"HTTP {response.status_code} on {self.api_url}. Body: {response.text}{hint}")
                response_json = response.json()
                return {
                    "structured": parse_structured_content(response_json),
                    "response_json": response_json,
                }
            except Exception as exc:  # pragma: no cover - network behavior
                last_error = exc
                if attempt + 1 < self.retry_attempts:
                    time.sleep(min(2**attempt, 5))
        assert last_error is not None
        raise last_error


class UtopiaOpenAIChatClient:
    """Small HTTP client for Utopia's OpenAI-compatible preset models."""

    def __init__(self, *, api_url: str, api_key: str, retry_attempts: int = 1) -> None:
        self.api_url = (
            api_url.rstrip("/")
            if api_url.rstrip("/").endswith("/chat/completions")
            else resolve_openai_chat_completions_url(api_url)
        )
        self.api_key = api_key
        self.retry_attempts = max(1, int(retry_attempts))

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Call an OpenAI-compatible endpoint and parse the structured response."""
        try:
            import requests
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("requests is required for remote structured chat calls") from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": payload_schema,
                    "strict": True,
                },
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout_seconds)
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status_code} on {self.api_url}. Body: {response.text}")
                response_json = response.json()
                return {
                    "structured": parse_openai_structured_content(response_json),
                    "response_json": response_json,
                }
            except Exception as exc:  # pragma: no cover - network behavior
                last_error = exc
                if attempt + 1 < self.retry_attempts:
                    time.sleep(min(2**attempt, 5))
        assert last_error is not None
        raise last_error

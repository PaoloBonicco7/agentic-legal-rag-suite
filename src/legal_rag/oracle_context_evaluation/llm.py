"""Minimal structured chat client compatible with Utopia/Ollama endpoints."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse


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
                    raise RuntimeError(f"HTTP {response.status_code} on {self.api_url}. Body: {response.text}")
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

"""Structured Utopia and OpenAI clients without cross-provider fallback."""

import json
import os
import time
from typing import Protocol, TypeVar

import requests
from pydantic import Field, ValidationError

from legal_rag.artifacts import digest
from legal_rag.contracts import ProviderConfig, Record, Usage
from .budget import BudgetLedger
from .errors import ProviderError

T = TypeVar("T", bound=Record)


def scientific_identity(config: ProviderConfig) -> dict:
    fields = ("provider", "model", "timeout_seconds", "max_output_tokens", "max_input_tokens", "context_window_tokens", "reasoning_effort", "retries")
    values = config.to_json_record()
    return {field: values[field] for field in fields}


def input_token_bound(prompt: str, schema: type[Record], config: ProviderConfig) -> tuple[str, int]:
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    bound = len((prompt + schema_text).encode("utf-8")) + 2048
    if bound > config.max_input_tokens:
        raise ProviderError("input_limit", "Full prompt and output schema exceed the configured conservative input token bound.")
    if config.context_window_tokens is not None and bound + config.max_output_tokens > config.context_window_tokens:
        raise ProviderError("context_limit", "Conservative input bound plus maximum output exceeds the configured model context window.")
    return schema_text, bound


class ModelInspection(Record):
    provider: str = "utopia"
    requested_model: str
    returned_model: str = "unknown"
    digest: str | None = None
    capabilities: list[str]
    context_window_tokens: int | None = None
    thinking_values: list[str] = Field(default_factory=list)
    thinking_default: str | bool | None = None


class StructuredClient(Protocol):
    def complete(self, prompt: str, schema: type[T]) -> tuple[T, Usage]: ...


class UtopiaClient:
    def __init__(self, config: ProviderConfig, *, api_key: str | None = None, session=None):
        self.config = config
        self._api_key = api_key or os.getenv("UTOPIA_API_KEY", "")
        self._session = session or requests.Session()
        root = config.base_url.rstrip("/")
        self.url = root if root.endswith("/ollama/api/chat") else root.removesuffix("/api") + "/ollama/api/chat"

    @property
    def identity(self) -> dict:
        return {**scientific_identity(self.config), "protocol": "ollama-structured", "endpoint": self.url}

    @property
    def accounting(self) -> dict:
        return {"provider": "utopia", "cost_basis": "unknown"}

    def preflight(self) -> dict:
        if not self._api_key:
            raise ProviderError("credentials_missing", "UTOPIA_API_KEY is missing.")
        return self.identity

    def inspect_model(self) -> ModelInspection:
        self.preflight()
        base = self.url.removesuffix("/chat")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = self._session.post(base + "/show", headers=headers, json={"model": self.config.model}, timeout=self.config.timeout_seconds)
        except requests.RequestException:
            raise ProviderError("transport_error", "Utopia model metadata could not be read.") from None
        if response.status_code >= 400:
            raise ProviderError(f"http_{response.status_code}", "Utopia rejected model metadata inspection.")
        try:
            data = response.json()
            information = data.get("model_info", {})
            thinking = data.get("thinking") if isinstance(data.get("thinking"), dict) else {}
            capacities = {value for key, value in information.items() if key.endswith(".context_length") and isinstance(value, int) and not isinstance(value, bool) and value > 0}
            if isinstance(data.get("context_length"), int) and data["context_length"] > 0:
                capacities.add(data["context_length"])
            inspected = ModelInspection(requested_model=self.config.model, returned_model=data.get("model") or data.get("name") or "unknown",
                                        digest=data.get("digest"), capabilities=data.get("capabilities", []),
                                        context_window_tokens=next(iter(capacities)) if len(capacities) == 1 else None,
                                        thinking_values=thinking.get("values", []), thinking_default=thinking.get("default"))
        except (ValueError, TypeError, AttributeError):
            raise ProviderError("invalid_metadata", "Utopia returned invalid model metadata.") from None
        if inspected.returned_model != "unknown" and inspected.returned_model != self.config.model:
            raise ProviderError("model_mismatch", "Utopia model metadata identifies a model different from the requested model.")
        if hasattr(self._session, "get"):
            try:
                tags = self._session.get(base + "/tags", headers=headers, timeout=self.config.timeout_seconds)
                if tags.status_code == 200:
                    matching = [row for row in tags.json().get("models", []) if self.config.model in {row.get("name"), row.get("model")}]
                    if len(matching) == 1:
                        inspected.returned_model = matching[0].get("model") or matching[0].get("name") or inspected.returned_model
                        inspected.digest = matching[0].get("digest") or inspected.digest
            except (requests.RequestException, ValueError, TypeError, AttributeError):
                pass
        if inspected.returned_model != "unknown" and inspected.returned_model != self.config.model:
            raise ProviderError("model_mismatch", "Utopia model inventory identifies a model different from the requested model.")
        return inspected

    def complete(self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        self.preflight()
        if self.config.context_window_tokens is None:
            raise ProviderError("context_unverified", "Utopia requires an explicit positive context window before chat dispatch.")
        input_token_bound(prompt, schema, self.config)
        payload = {"model": self.config.model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                   "format": schema.model_json_schema(),
                   "options": {"temperature": 0, "num_predict": self.config.max_output_tokens, "num_ctx": self.config.context_window_tokens}}
        if self.config.reasoning_effort is not None:
            payload["think"] = False if self.config.reasoning_effort == "none" else self.config.reasoning_effort
        started = time.perf_counter()
        for attempt in range(min(self.config.retries, 2) + 1):
            try:
                response = self._session.post(self.url, headers={"Authorization": f"Bearer {self._api_key}"}, json=payload, timeout=self.config.timeout_seconds)
            except (requests.Timeout, requests.ConnectionError):
                if attempt < min(self.config.retries, 2):
                    time.sleep(min(2 ** attempt, 2))
                    continue
                raise ProviderError("transport_error", "Utopia request failed after bounded retries.") from None
            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < min(self.config.retries, 2):
                time.sleep(min(2 ** attempt, 2))
                continue
            if response.status_code >= 400:
                failed = Usage(provider="utopia", requested_model=self.config.model, returned_model="unknown", elapsed_seconds=time.perf_counter() - started)
                raise ProviderError(f"http_{response.status_code}", "Utopia rejected the request; no fallback was used.", usage=[failed])
            usage = None
            raw = None
            try:
                data = response.json()
                content = data.get("message", {}).get("content")
                if isinstance(content, str):
                    raw = {"model_text": content}
                usage = Usage(provider="utopia", requested_model=self.config.model, returned_model=data.get("model") or "unknown",
                              input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"), elapsed_seconds=time.perf_counter() - started,
                              raw_output=content if isinstance(content, str) else None)
                if usage.returned_model != "unknown" and usage.returned_model != self.config.model:
                    raise ProviderError("model_mismatch", "Utopia chat returned a model different from the requested model.", usage=[usage], raw_payload=raw)
                if data.get("done_reason") == "length" or data.get("done") is False:
                    raise ProviderError("truncated_output", "Utopia output reached its token limit.", usage=[usage], raw_payload=raw)
                parsed = schema.model_validate_json(content) if isinstance(content, str) else schema.model_validate(content)
                return parsed, usage
            except ProviderError:
                raise
            except (ValueError, KeyError, TypeError, AttributeError, ValidationError):
                raise ProviderError("invalid_output", "Utopia returned invalid structured output.", usage=[usage] if usage else [], raw_payload=raw) from None


class OpenAIClient:
    def __init__(self, config: ProviderConfig, *, ledger_path="data/budget/openai.json", purpose="experiment", api_key=None, sdk=None):
        self.config = config
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.ledger = BudgetLedger(ledger_path, config, purpose=purpose)
        self._sdk = sdk

    @property
    def identity(self) -> dict:
        return {**scientific_identity(self.config), "base_url": "https://api.openai.com/v1", "protocol": "openai-responses", "store": False}

    @property
    def accounting(self) -> dict:
        fields = ("budget_eur", "budget_reserve_eur", "eur_per_usd", "price_date", "price_source_url", "fx_date", "fx_source_url", "input_usd_per_million", "output_usd_per_million")
        values = self.config.to_json_record()
        return {**{field: values[field] for field in fields}, "ledger_path": str(self.ledger.path), "purpose": self.ledger.purpose,
                "cost_basis": "conservative_estimate"}

    def preflight(self) -> dict:
        if not self._api_key and self._sdk is None:
            raise ProviderError("credentials_missing", "OPENAI_API_KEY is missing.")
        self.ledger.preflight()
        if self._sdk is None:
            from openai import OpenAI
            self._sdk = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1", max_retries=0, timeout=self.config.timeout_seconds)
        return self.identity

    def complete(self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        self.preflight()
        schema_text, input_bound = input_token_bound(prompt, schema, self.config)
        bound = self.ledger.cost(input_bound, self.config.max_output_tokens)
        started = time.perf_counter()
        failed_usage = []
        for attempt in range(min(self.config.retries, 2) + 1):
            attempt_started = time.perf_counter()
            reservation = self.ledger.reserve(bound, model=self.config.model, request_hash=digest(prompt + schema_text))
            try:
                arguments = dict(model=self.config.model, input=prompt, text_format=schema, max_output_tokens=self.config.max_output_tokens,
                                 store=False, truncation="disabled", service_tier="default")
                if self.config.reasoning_effort is not None:
                    arguments["reasoning"] = {"effort": self.config.reasoning_effort}
                response = self._sdk.responses.parse(**arguments)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                rejected = status in {400, 401, 403, 404, 422, 429}
                self.ledger.settle(reservation, 0.0 if rejected else None)
                failed_usage.append(Usage(provider="openai", requested_model=self.config.model, returned_model="unknown",
                    elapsed_seconds=time.perf_counter() - attempt_started, cost_eur=0.0 if rejected else bound,
                    ledger_entry_id=reservation, cost_basis="conservative_reservation", price_date=self.config.price_date))
                transient = status in {408, 429, 500, 502, 503, 504} or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}
                if transient and attempt < min(self.config.retries, 2):
                    time.sleep(min(2 ** attempt, 2))
                    continue
                raise ProviderError(f"http_{status}" if status else "provider_error", "OpenAI request failed; reservation was accounted for.", usage=failed_usage) from None
            response_usage = getattr(response, "usage", None)
            raw = {"model_text": response.output_text} if isinstance(getattr(response, "output_text", None), str) else None
            actual = self.ledger.cost(response_usage.input_tokens, response_usage.output_tokens) if response_usage else None
            self.ledger.settle(reservation, actual)
            usage = Usage(provider="openai", requested_model=self.config.model, returned_model=getattr(response, "model", None) or "unknown",
                          input_tokens=getattr(response_usage, "input_tokens", None), output_tokens=getattr(response_usage, "output_tokens", None),
                          elapsed_seconds=time.perf_counter() - started, cost_eur=actual if actual is not None else bound, ledger_entry_id=reservation,
                          cost_basis="conservative_estimate" if actual is not None else "conservative_reservation", price_date=self.config.price_date,
                          raw_output=raw["model_text"] if raw else None)
            if actual is not None and actual > bound + 1e-9:
                raise ProviderError("cost_bound_exceeded", "Actual usage exceeded the reserved bound; further runs require review.", usage=[*failed_usage, usage], raw_payload=raw)
            if getattr(response, "status", "completed") != "completed":
                raise ProviderError("truncated_output", "OpenAI response did not complete.", usage=[*failed_usage, usage], raw_payload=raw)
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise ProviderError("refusal", "OpenAI did not return the requested structured object.", usage=[*failed_usage, usage], raw_payload=raw)
            try:
                parsed = schema.model_validate(parsed)
                if failed_usage:
                    usage.cost_eur += sum(item.cost_eur or 0 for item in failed_usage)
                    usage.cost_basis = "conservative_estimate_including_retry_reservations"
                return parsed, usage
            except ValidationError:
                raise ProviderError("invalid_output", "OpenAI structured output failed validation.", usage=[*failed_usage, usage], raw_payload=raw) from None


def build_chat_client(config: ProviderConfig, *, ledger_path="data/budget/openai.json", purpose="experiment", api_key=None) -> StructuredClient:
    if config.provider == "utopia":
        return UtopiaClient(config, api_key=api_key)
    return OpenAIClient(config, ledger_path=ledger_path, purpose=purpose, api_key=api_key)

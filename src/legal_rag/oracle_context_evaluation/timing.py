"""Timing wrapper for StructuredChatClient calls.

Records per-call latency and structured-output schema, useful when debugging
LLM-bound smoke runs from notebook 06 or any other diagnostic that needs to
attribute time to individual schema calls.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .llm import StructuredChatClient


class TimingStructuredChatClient:
    """Wrap a StructuredChatClient and record latency for every call."""

    def __init__(self, inner: StructuredChatClient, *, verbose: bool = True) -> None:
        self.inner = inner
        self._verbose = verbose
        self.calls: list[dict[str, Any]] = []

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Forward the call to the wrapped client while timing it."""
        schema_name = str(payload_schema.get("title") or "structured_output")
        started = perf_counter()
        if self._verbose:
            print(
                f"[llm-debug] call_start schema={schema_name} model={model} "
                f"prompt_chars={len(prompt)} timeout={timeout_seconds}s",
                flush=True,
            )
        try:
            result = self.inner.structured_chat(
                prompt=prompt,
                model=model,
                payload_schema=payload_schema,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            elapsed = perf_counter() - started
            self.calls.append(
                {
                    "schema": schema_name,
                    "model": model,
                    "ok": False,
                    "seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if self._verbose:
                print(
                    f"[llm-debug] call_fail schema={schema_name} "
                    f"seconds={elapsed:.1f} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            raise
        elapsed = perf_counter() - started
        self.calls.append({"schema": schema_name, "model": model, "ok": True, "seconds": round(elapsed, 3)})
        if self._verbose:
            print(f"[llm-debug] call_ok schema={schema_name} seconds={elapsed:.1f}", flush=True)
        return result

    def total_seconds(self) -> float:
        """Total seconds spent across all recorded calls."""
        return round(sum(call["seconds"] for call in self.calls), 3)

    def seconds_by_schema(self) -> dict[str, float]:
        """Aggregate latency per schema name."""
        out: dict[str, float] = {}
        for call in self.calls:
            schema = str(call["schema"])
            out[schema] = round(out.get(schema, 0.0) + float(call["seconds"]), 3)
        return out

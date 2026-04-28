from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import AdvancedRewriteConfig
from .llm import (
    SupportsInvoke,
    SupportsRunSync,
    invoke_model,
    parse_structured_output,
    run_structured_with_agent,
)


class QueryRewritePayload(BaseModel):
    """Structured output for optional query rewriting/decomposition."""

    model_config = ConfigDict(extra="forbid")

    rewritten_queries: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @field_validator("rewritten_queries", mode="before")
    @classmethod
    def _coerce_queries(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            cur = str(item or "").strip()
            if not cur:
                continue
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
        return out

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_rationale(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_queries: tuple[str, ...]
    used_llm: bool
    fallback_used: bool
    error: str | None = None

    def all_queries(self, *, max_subqueries: int) -> list[str]:
        max_subqueries = max(1, int(max_subqueries))
        out: list[str] = []
        seen: set[str] = set()
        for q in [self.original_query, *self.rewritten_queries]:
            cur = str(q or "").strip()
            if not cur or cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            if len(out) >= max_subqueries:
                break
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_queries": list(self.rewritten_queries),
            "used_llm": self.used_llm,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


def _rewrite_prompt(question: str) -> str:
    return (
        "Sei un assistente per query rewriting legale.\n"
        "Data una domanda utente, proponi fino a 3 varianti utili al retrieval: "
        "una versione normalizzata e eventuali sotto-query complementari.\n"
        "Le varianti devono essere concise, in italiano, senza inventare contenuti.\n"
        "Restituisci solo JSON valido con campi: rewritten_queries, rationale.\n\n"
        "Domanda:\n"
        f"{question.strip()}"
    )


def rewrite_query(
    question: str,
    *,
    config: AdvancedRewriteConfig,
    llm_model: SupportsInvoke | None = None,
    rewrite_agent: SupportsRunSync | None = None,
) -> QueryRewriteResult:
    question = str(question or "").strip()
    if not question:
        return QueryRewriteResult(
            original_query="",
            rewritten_queries=tuple(),
            used_llm=False,
            fallback_used=True,
            error="empty_question",
        )

    if not config.enabled:
        return QueryRewriteResult(
            original_query=question,
            rewritten_queries=tuple(),
            used_llm=False,
            fallback_used=False,
            error=None,
        )

    if not config.use_llm:
        return QueryRewriteResult(
            original_query=question,
            rewritten_queries=tuple(),
            used_llm=False,
            fallback_used=True,
            error=None,
        )

    if llm_model is None and rewrite_agent is None:
        return QueryRewriteResult(
            original_query=question,
            rewritten_queries=tuple(),
            used_llm=False,
            fallback_used=True,
            error="rewrite_llm_unavailable",
        )

    try:
        prompt = _rewrite_prompt(question)
        if rewrite_agent is not None:
            payload = run_structured_with_agent(prompt, rewrite_agent, QueryRewritePayload)
        else:
            assert llm_model is not None
            raw = invoke_model(llm_model, prompt)
            payload = parse_structured_output(raw, QueryRewritePayload)

        max_rewrites = max(0, int(config.max_rewrites))
        rewrites: list[str] = []
        seen: set[str] = {question}
        for item in payload.rewritten_queries:
            cur = str(item or "").strip()
            if not cur or cur in seen:
                continue
            seen.add(cur)
            rewrites.append(cur)
            if len(rewrites) >= max_rewrites:
                break

        fallback_used = len(rewrites) == 0 and bool(config.fallback_to_original)
        return QueryRewriteResult(
            original_query=question,
            rewritten_queries=tuple(rewrites),
            used_llm=True,
            fallback_used=fallback_used,
            error=None,
        )
    except Exception as exc:
        return QueryRewriteResult(
            original_query=question,
            rewritten_queries=tuple(),
            used_llm=True,
            fallback_used=True,
            error=f"{type(exc).__name__}: {exc}",
        )


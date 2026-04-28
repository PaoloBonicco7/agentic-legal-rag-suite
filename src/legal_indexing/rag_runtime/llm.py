from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import RagRuntimeConfig


_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
TModel = TypeVar("TModel", bound=BaseModel)


class SupportsInvoke(Protocol):
    def invoke(self, prompt: str) -> Any:
        ...


class SupportsRunSync(Protocol):
    def run_sync(self, prompt: str) -> Any:
        ...


def build_chat_model(config: RagRuntimeConfig) -> SupportsInvoke:
    provider = (config.llm_provider or "").strip().lower()
    if provider in {"none", "disabled"}:
        raise RuntimeError(
            "LLM is disabled by configuration. Set llm_provider to 'utopia' "
            "or pass a custom llm object to run_rag_question(..., llm=...)."
        )
    if provider not in {"utopia", "openai_compatible", "pydanticai"}:
        raise ValueError(
            f"Unsupported llm_provider={config.llm_provider!r}. "
            "Supported values: utopia, openai_compatible, pydanticai."
        )
    if not config.llm_api_key:
        raise RuntimeError(
            "UTOPIA_API_KEY is missing for answer generation. Set it in .env "
            "or pass a custom llm object."
        )
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("langchain-openai is required for chat model usage") from exc

    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=float(config.llm_temperature),
    )


def build_pydantic_ai_agent(
    config: RagRuntimeConfig,
    *,
    result_type: type[TModel],
    system_prompt: str,
) -> SupportsRunSync:
    """Build a PydanticAI agent on top of an OpenAI-compatible endpoint."""

    if not config.llm_api_key:
        raise RuntimeError(
            "UTOPIA_API_KEY is missing for structured generation via PydanticAI."
        )

    try:
        from pydantic_ai import Agent
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pydantic-ai is required for structured generation") from exc

    # PydanticAI API changed across versions; keep creation defensive.
    model_obj: Any
    try:
        from pydantic_ai.models.openai import OpenAIModel

        model_obj = OpenAIModel(
            model_name=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        return Agent(
            model=model_obj,
            result_type=result_type,
            system_prompt=system_prompt,
        )
    except Exception:
        # Fallback for versions where an OpenAI-compatible model can be passed by name.
        return Agent(
            model=f"openai:{config.llm_model}",
            result_type=result_type,
            system_prompt=system_prompt,
        )


def invoke_model(llm: SupportsInvoke, prompt: str) -> str:
    response = llm.invoke(prompt)
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    candidate = raw
    match = _JSON_OBJ_RE.search(raw)
    if match:
        candidate = match.group(0)

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_structured_output(raw_text: str, model_cls: type[TModel]) -> TModel:
    data = _extract_json_object(raw_text)
    if isinstance(data, dict):
        allowed = set(model_cls.model_fields.keys())
        data = {k: v for k, v in data.items() if k in allowed}
    else:
        data = {}
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        # For models with defaults (e.g. RagAnswer), keep the pipeline alive.
        # For strict schemas (e.g. MCQ/Judge), preserve the original failure.
        try:
            return model_cls.model_validate({})
        except ValidationError:
            raise exc


def is_empty_answer_text(value: Any) -> bool:
    return not str(value or "").strip()


def is_empty_structured_answer(payload: Any, *, field_name: str = "answer") -> bool:
    """Post-parse runtime guard helper for empty textual answers."""
    if isinstance(payload, BaseModel):
        value = getattr(payload, field_name, "")
        return is_empty_answer_text(value)
    if isinstance(payload, dict):
        return is_empty_answer_text(payload.get(field_name))
    return is_empty_answer_text(payload)


def run_structured_with_agent(prompt: str, agent: SupportsRunSync, model_cls: type[TModel]) -> TModel:
    result = agent.run_sync(prompt)
    data = getattr(result, "data", result)
    if isinstance(data, model_cls):
        return data
    if isinstance(data, BaseModel):
        return model_cls.model_validate(data.model_dump())
    return model_cls.model_validate(data)

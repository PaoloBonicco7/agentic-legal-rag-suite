from __future__ import annotations

import json

from legal_indexing.rag_runtime.config import AdvancedRewriteConfig
from legal_indexing.rag_runtime.query_rewriting import rewrite_query


class FakeRewriteLLM:
    def invoke(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(
            {
                "rewritten_queries": [
                    "Domanda normalizzata",
                    "Domanda normalizzata",
                    "Sotto-query 1",
                    "Sotto-query 2",
                ],
                "rationale": "ok",
            },
            ensure_ascii=False,
        )


class BrokenRewriteLLM:
    def invoke(self, prompt: str) -> str:
        _ = prompt
        return "not-json"


def test_rewrite_disabled_keeps_original_query() -> None:
    cfg = AdvancedRewriteConfig(enabled=False)
    result = rewrite_query("Qual e la norma vigente?", config=cfg)
    assert result.original_query == "Qual e la norma vigente?"
    assert result.rewritten_queries == tuple()
    assert result.all_queries(max_subqueries=4) == ["Qual e la norma vigente?"]


def test_rewrite_llm_deduplicates_and_applies_caps() -> None:
    cfg = AdvancedRewriteConfig(enabled=True, use_llm=True, max_rewrites=2, max_subqueries=3)
    result = rewrite_query(
        "Qual e la procedura prevista?",
        config=cfg,
        llm_model=FakeRewriteLLM(),
    )
    assert result.used_llm is True
    assert len(result.rewritten_queries) == 2
    assert result.rewritten_queries[0] == "Domanda normalizzata"
    all_queries = result.all_queries(max_subqueries=3)
    assert len(all_queries) == 3
    assert all_queries[0] == "Qual e la procedura prevista?"


def test_rewrite_fallback_when_model_output_is_invalid() -> None:
    cfg = AdvancedRewriteConfig(enabled=True, use_llm=True, fallback_to_original=True)
    result = rewrite_query(
        "Qual e la legge abrogata?",
        config=cfg,
        llm_model=BrokenRewriteLLM(),
    )
    assert result.fallback_used is True
    assert result.rewritten_queries == tuple()
    assert result.all_queries(max_subqueries=3) == ["Qual e la legge abrogata?"]
    assert result.used_llm is True

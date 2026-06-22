# 06b - Retrieval Diagnostics Specification

## Purpose

Define the retrieval-only diagnostic workflow used before promoting any retrieval improvement into Advanced Graph RAG.

This step isolates retrieval quality from answer generation. It measures whether the expected legal references appear in the candidate set, which ranking position they occupy, and which retrieval variants deserve an end-to-end RAG run. The goal is to improve recall and ranking with reproducible evidence before spending LLM calls on final answers.

## Inputs

- Clean evaluation datasets from step 02 (`data/evaluation_clean/`), including expected legal references.
- Retrieval-ready Qdrant index from step 03 (`data/indexes/qdrant`).
- Clean legal graph and chunk data from step 01 when graph expansion experiments are enabled.
- Retrieval evaluation configuration:
  - `collection_name`: Qdrant collection under test.
  - `retrieval_mode`: dense or hybrid.
  - `top_k`: candidate budget to retrieve or evaluate.
  - `rrf_k`: RRF rank constant when hybrid retrieval is evaluated.
  - `static_filters`: optional metadata filters.
  - graph expansion parameters when graph experiments are enabled.
  - rerank configuration and cache path when rerank experiments are enabled.
  - `rerank_input_k`, `rerank_output_k`, `rerank_model`, and `rerank_prompt_version` for Experiment G.
  - query rewriting configuration and cache path when Experiment H is enabled.
  - `query_rewriting_strategy`, `query_rewriting_model`, `query_rewriting_prompt_version`, and `multi_query_n` for Experiment H.

## Outputs

Default generated output directory: `data/retrieval_eval_runs/<run_name>__<YYYYMMDDTHHMMSSZ>/`.

- `manifest.json`: input references, configuration, index identity, schema and prompt versions where applicable.
- `scenarios.csv`: aggregate retrieval metrics for every scenario.
- `sweep_rerank.csv`: row-level rerank diagnostics when Experiment G runs.
- `sweep_query_rewriting.csv`: row-level query rewriting diagnostics when Experiment H runs.
- Row-level diagnostic files: per-question candidates, expected references, hit flags, ranking positions, and skipped/error status.
- Historical comparison tables and plots when a previous dense-only run is configured.
- Optional cache files under `data/cache/` for expensive rerank or query rewriting calls.
- A human-readable results report in `docs/results/06b_retrieval_diagnostics.md`.

## Pipeline

1. Load evaluation records and expected references.
   Why: retrieval quality is measured against fixed law and article targets.
2. Retrieve candidate chunks for each question with the configured retrieval mode and budget.
   Why: dense, hybrid, filtered, and graph-expanded runs must be comparable.
3. Optionally expand or rerank the candidate set.
   Why: each advanced retrieval component must be measured before promotion to step 06.
4. Compare retrieved candidates with expected references.
   Why: the diagnostic task is reference coverage, not answer generation.
5. Summarize every scenario with shared retrieval metrics.
   Why: later phases need a compact decision table for selecting the best configuration.
6. Write run artifacts and update the experiment log.
   Why: every promoted configuration must be traceable to a specific diagnostic run.

## Contract

Each scenario summary must include:

- `scenario_name`
- `dataset`
- `stage`
- `article_hit_pct`
- `law_hit_pct`
- `article_mrr`
- `n_questions`
- `n_filter_excluded`
- `config`
- `delta_vs_baseline`
- `experiment_name`
- `status`
- `skip_reason`

Skipped scenarios must keep a row in `scenarios.csv` with `status="skipped"` and a non-empty `skip_reason`.

Row-level diagnostics must preserve the question id, expected references, retrieved chunk ids, retrieved law/article ids, hit flags, retrieval mode, `top_k`, `rrf_k` when applicable, and rank of the first matching article when present.

Rerank row-level diagnostics must additionally preserve `base_scenario`, `rerank_model`, `rerank_input_k`, `rerank_output_k`, `cache_hit`, `reranked_chunk_ids`, `rerank_scores`, and whether rerank recovered or demoted the expected article.

Query rewriting row-level diagnostics must additionally preserve `strategy`, `query_rewriting_model`, `query_rewriting_prompt_version`, `cache_hit`, `rewritten_queries`, `transformed_query_count`, and the candidate ids retrieved after the transformation.

## Quality Gates

- Diagnostic runs never overwrite previous runs.
- Scenario metrics are derived from row-level diagnostics, not hand-edited.
- The baseline dense scenario is always present for each evaluated dataset.
- Skipped experiments are explicit and explain why they could not run.
- Hybrid scenarios run only when the tested index exposes sparse vectors and the embedder can produce sparse embeddings.
- Experiment F evaluates BGE-M3 dense against BGE-M3 hybrid over `top_k in {10, 20, 50, 100}` and `rrf_k in {30, 60, 90}`.
- Experiment G evaluates LLM reranking over the best hybrid RRF setting with `rerank_input_k in {20, 50, 100}` and `rerank_output_k in {3, 5, 10}` on a deterministic 30-question pilot per dataset before any full run.
- Experiment G calls the LLM only through `build_rerank_prompt()` plus `UtopiaStructuredChatClient.structured_chat()` with `RerankOutput.model_json_schema()`.
- Rerank cache keys include question text, ordered candidate chunk ids, model name, and `RERANK_PROMPT_VERSION`; the cache stores raw `{chunk_id, score}` entries, not final ordering.
- Experiment H evaluates `none`, `rewrite`, `hyde`, and `multi_query` on a deterministic 30-question pilot per dataset before any full run.
- Experiment H calls the LLM only through `UtopiaStructuredChatClient.structured_chat()` with strict Pydantic schemas from `retrieval_evaluation.query_rewriting`.
- Query rewriting cache keys include question text, strategy, model name, and `QUERY_REWRITING_PROMPT_VERSION`; cache files live under `data/cache/query_rewriting/<strategy>__<model>__<prompt_version>.jsonl`.
- `multi_query(n=3)` must return exactly three non-empty distinct strings; invalid structured outputs fail the row and are counted in the manifest.
- The notebook comparison separates old-index dense performance, new-index dense performance, and new-index hybrid performance.
- Rerank and query rewriting experiments use caches and versioned prompts.
- Experiment decisions (promoted vs rejected levers, key metrics) are recorded in `docs/results/06b_retrieval_diagnostics.md`.

## Notebook Role

`notebooks/06b_retrieval_diagnostics.ipynb` should:

- load a diagnostic run and display `scenarios.csv`;
- compare dense, filtered, graph, hybrid, rerank, and query rewriting experiments when available;
- compare the current BGE-M3 hybrid-ready index with a configured historical dense-only run;
- inspect representative failures where the correct article is missing from top-k candidates;
- identify the best configuration to promote into `notebooks/06_advanced_graph_rag.ipynb`;
- avoid redefining reusable retrieval logic already implemented under `src/legal_rag/`.

## Acceptance Criteria

- A reader can reproduce the retrieval-only baseline and understand why each later retrieval experiment was selected or rejected.
- `scenarios.csv` contains enough information to compare article hit rate, law hit rate, MRR, status, and configuration across scenarios.
- Experiment decisions are recorded in `docs/results/06b_retrieval_diagnostics.md`.
- The selected retrieval configuration can be promoted to Advanced Graph RAG without relying on notebook-only state.

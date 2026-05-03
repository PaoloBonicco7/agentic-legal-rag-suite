# 06 - Advanced Graph RAG Specification

## Purpose

Define explainable RAG improvements over the simple baseline.

This step proves whether metadata filters, hybrid retrieval, explicit graph expansion, and LLM reranking improve performance while remaining understandable for a thesis PoC. Every improvement is configurable through boolean flags so that ablation runs can isolate the contribution of each component, and every additional candidate or expanded chunk is traceable to explicit evidence (a payload filter, a sparse-or-dense match, an explicit edge in `edges.jsonl`, or an LLM rerank score).

## Inputs

- Clean evaluation datasets from step 02 (`data/evaluation_clean/`).
- Retrieval-ready Qdrant index from step 03 with hybrid (dense + sparse) vectors and payload indexes on `law_id`, `law_status`, `index_views`, `article_id`, `relation_types`.
- Clean legal dataset from step 01, especially `edges.jsonl`, `chunks.jsonl`, and `manifest.json`.
- Simple RAG run output from step 05 for direct comparison.
- Advanced RAG configuration (`AdvancedRagConfig` Pydantic model):
  - paths and `chat_model` / `judge_model` consistent with step 05;
  - `prompt_version: str` (default `advanced-rag-prompts-v1`);
  - `run_name: str` (default `default`): used to disambiguate ablation runs in the output directory.
  - **Feature flags** (each independently switchable for ablation):
    - `metadata_filters_enabled: bool` (default `True`).
    - `hybrid_enabled: bool` (default `True`).
    - `graph_expansion_enabled: bool` (default `True`).
    - `rerank_enabled: bool` (default `True`).
  - **Retrieval parameters**:
    - `static_filters: dict[str, Any]` (default `{"law_status": "current"}`): applied when `metadata_filters_enabled=True`.
    - `top_k: int` (default `10`): top-k from each vector type before fusion.
    - `rrf_k: int` (default `60`): Reciprocal Rank Fusion constant used by Qdrant `Query API`.
  - **Graph expansion parameters**:
    - `graph_expansion_seed_k: int` (default `5`): how many of the top retrieved chunks seed the expansion.
    - `graph_expansion_relation_types: list[str]` (default: all 8 from step 01 — `REFERENCES`, `ABROGATED_BY`, `ABROGATES`, `MODIFIED_BY`, `AMENDS`, `REPLACED_BY`, `REPLACES`, `INSERTED_BY`, `INSERTS`).
    - `max_chunks_per_expanded_law: int` (default `2`): cap on chunks added per law reached via expansion.
    - `graph_expansion_hops: int` (default `1`): kept as `1` for the PoC; multi-hop is out of scope.
  - **Rerank parameters**:
    - `rerank_input_k: int` (default `20`): number of candidates passed to the reranker.
    - `rerank_output_k: int` (default `5`): number of candidates kept after reranking; this is the cap on context chunks.
    - `max_context_chars: int` (default `16000`): same cap as step 05.

## Outputs

Default generated output directory: `data/rag_runs/advanced/<run_name>/`.

- `advanced_rag_manifest.json`: dataset hashes, index manifest reference, simple RAG run reference, advanced settings (all flags + parameters), model identities.
- `mcq_results.jsonl`: row-level MCQ answers with retrieval, expansion, and rerank diagnostics.
- `no_hint_results.jsonl`: row-level open answers with retrieval, expansion, rerank diagnostics, and judge results.
- `advanced_rag_summary.json`: aggregate metrics (shared contract) plus per-feature counters.
- `advanced_diagnostics.json`: per-feature counters (filtered counts, hybrid hits, expansion edges used, rerank score distribution) and failure category counts.
- `quality_report.md`: run status, errors, comparison readiness against the referenced simple RAG run.

## Pipeline

1. Load evaluation records, the index manifest, the simple RAG run manifest, and configuration.
   Why: advanced behavior must be comparable against a known simple RAG baseline, not against an arbitrary one.
2. For each question, build the retrieval request.
   Why: filters and retrieval mode must be recorded per row before any candidate is fetched.
3. Apply metadata filters when `metadata_filters_enabled=True`.
   Why: filters such as current-law view or law status improve precision when justified by the project; when disabled, the same retrieval runs without them.
4. Retrieve candidates from Qdrant.
   - When `hybrid_enabled=True`: use the Qdrant `Query API` with two `prefetch` blocks (dense and sparse), each returning `top_k`, fused by RRF with `rrf_k`.
   - When `hybrid_enabled=False`: use dense-only retrieval with `top_k`.
   Why: hybrid is intended to improve exact legal-reference matching while preserving semantic recall.
5. Expand candidates via explicit graph edges when `graph_expansion_enabled=True`.
   Why: related laws (referenced, modifying, modified, replaced) often hold the article that resolves the question.
   - Take the `graph_expansion_seed_k` top retrieved chunks; collect their `law_id` values.
   - For every edge in `edges.jsonl` whose source `law_id` is in the seed set and whose `relation_type` is in `graph_expansion_relation_types`, add up to `max_chunks_per_expanded_law` chunks from the target law (selected by Qdrant via filter on the target `law_id` plus the same `static_filters` when enabled).
   - Record the edges actually used in `graph_relations_used` for traceability.
6. Rerank candidates when `rerank_enabled=True`.
   Why: the LLM reranker reorders by legal relevance and prunes noisy candidates before the answer step.
   - Pass `rerank_input_k` candidates (retrieval + expansion, deduplicated) to the LLM reranker.
   - The reranker prompt asks the LLM to assign a relevance score in `{0, 1, 2}` per chunk against the question, returning a structured object.
   - Keep the top `rerank_output_k` candidates by score; ties broken by original retrieval rank.
7. Build bounded context.
   Why: the context budget must remain explicit and reproducible, identical in shape to step 05.
   - Concatenate kept candidates in rerank order until either `rerank_output_k` chunks or `max_context_chars` is reached.
8. Generate MCQ and no-hint answers, attach citations, judge no-hint answers.
   Why: the answer contract must remain compatible with step 05 so that step 07 can compare directly.
9. Export row-level traces, diagnostics, and summary metrics.
   Why: improvements must be explainable, not only numerically better.

## Contract

Advanced result rows must include all simple RAG fields plus:

- `metadata_filters`: dict actually applied on this row (empty when the flag is off).
- `retrieval_mode`: `"dense"` or `"hybrid"`.
- `graph_expanded_law_ids`: list of law ids reached via expansion (empty when the flag is off).
- `graph_expanded_chunk_ids`: list of chunk ids added via expansion (empty when the flag is off).
- `graph_relations_used`: list of `{source_law_id, target_law_id, relation_type}` triples actually consumed.
- `reranked_chunk_ids`: list of chunk ids in rerank-order (equal to retrieved+expanded order when rerank is off).
- `rerank_scores`: list of integer scores aligned with `reranked_chunk_ids` (empty when rerank is off).
- `context_included_count`: number of chunks placed in the context.
- `reference_law_hit`: `bool` flag set to `True` when at least one `law_id` from the question's `expected_references` appears in `context_chunk_ids`.
- `failure_category`: one of the allowed values below, populated for failed or low-quality rows.

Allowed failure categories:

- `retrieval_miss`: no chunk retrieved or retrieved chunks miss every expected reference;
- `context_noise`: context built but `reference_law_hit=False` and answer wrong;
- `abstention`: the model returned an empty or "non lo so"-style answer;
- `contradiction`: the answer contradicts the cited chunks;
- `generation_error`: structured-output failure or HTTP error during answer generation;
- `judge_error`: judge returned an invalid score;
- `unknown`.

Citation contract is identical to step 05: `{law_id, article_id, chunk_id, chunk_text}`, valid only when `chunk_id` is in `context_chunk_ids`.

Graph expansion must use only explicit relations from `edges.jsonl`. Inferred or LLM-suggested edges are not allowed.

The advanced manifest must record every flag value, every parameter value, the simple RAG run referenced for comparison, the index manifest hash, and the evaluation manifest hash.

## Quality Gates

- The advanced run records the simple RAG baseline it compares against and uses the same evaluation dataset hash.
- For every row, every flag's effect is observable: `metadata_filters` is empty iff `metadata_filters_enabled=False`; `retrieval_mode` is `dense` iff `hybrid_enabled=False`; expansion fields are empty iff `graph_expansion_enabled=False`; `rerank_scores` is empty iff `rerank_enabled=False`.
- Every entry in `graph_relations_used` corresponds to an actual record in `edges.jsonl`.
- Hybrid retrieval is only enabled when the index manifest declares sparse vectors are present.
- Rerank scores are integers in `{0, 1, 2}`; out-of-range scores cause `judge_error`-style row errors and are excluded from the kept set.
- Summary metric names match the no-RAG and simple RAG contracts so that step 07 can compare directly.
- `failure_category` is populated for every wrong or empty row.

## Notebook Role

`notebooks/06_advanced_graph_rag.ipynb` should:

- run one ablation matrix (e.g., baseline-mode = all flags on; then one off at a time) using distinct `run_name` values writing into separate subdirectories under `data/rag_runs/advanced/`;
- demonstrate one advanced query trace from retrieval through expansion and rerank into the final answer, highlighting the edges used and the rerank scores;
- compare global and by-level metrics across simple RAG (step 05) and the advanced runs;
- show the `failure_category` breakdown to explain where each component helps or fails.

The notebook focuses on *why* each advanced component helps or fails, not on adding production complexity.

## Acceptance Criteria

- Advanced RAG can be run on the same evaluation records as simple RAG using the same index from step 03.
- Each improvement is explainable through row-level diagnostics: a reader can see, for any wrong or right answer, which filters were applied, which mode retrieved the candidates, which edges expanded the set, and which rerank scores survived.
- Graph expansion uses only explicit legal relations from `edges.jsonl`.
- Metrics are directly comparable with no-RAG (step 04) and simple RAG (step 05) using the shared metric contract.
- Each advanced feature can be switched off via its flag without breaking the pipeline; ablation runs are reproducible by changing flags only and renaming `run_name`.

# 06 - Advanced Graph RAG Specification

## Purpose

Define explainable RAG improvements over the simple baseline.

This step proves whether metadata filters, explicit graph expansion, reranking, and optional hybrid retrieval improve performance while remaining understandable for a thesis PoC.

## Inputs

- Clean evaluation datasets from step 02.
- Retrieval-ready index from step 03.
- Clean legal dataset from step 01, especially `edges.jsonl`, `chunks.jsonl`, and `manifest.json`.
- Simple RAG results from step 05 for comparison.
- Advanced RAG configuration: filter policy, graph expansion policy, reranking policy, optional hybrid retrieval flag, top-k values, model identities, and prompt versions.

## Outputs

Default generated output directory: `data/rag_runs/advanced/`.

- `advanced_rag_manifest.json`: dataset hashes, index reference, advanced settings, and model identities.
- `mcq_results.jsonl`: row-level MCQ answers with retrieval diagnostics.
- `no_hint_results.jsonl`: row-level open answers with graph and reranking diagnostics.
- `advanced_rag_summary.json`: aggregate metrics.
- `advanced_diagnostics.json`: retrieval mode counts, graph expansion counts, reranking effects, and failure categories.
- `quality_report.md`: run status, errors, and comparison readiness.

## Pipeline

1. Start from the simple RAG contract.
   Why: advanced behavior must be comparable and not become a separate experiment.
2. Apply explainable metadata filters.
   Why: filters such as current-law view or law status can improve precision when justified by the question.
3. Retrieve candidate chunks.
   Why: advanced RAG still begins with inspectable retrieval evidence.
4. Expand using explicit graph edges.
   Why: related laws can add context when the retrieved law refers to, modifies, or is modified by another law.
5. Optionally combine lexical and vector retrieval.
   Why: hybrid retrieval may improve exact legal reference matching, but it remains optional for the PoC.
6. Rerank candidates.
   Why: reranking should prefer legally relevant, less noisy context.
7. Build bounded context and generate answers.
   Why: advanced retrieval should improve evidence quality without changing the answer contract.
8. Export diagnostics and metrics.
   Why: improvements must be explainable, not only numerically better.

## Contract

Advanced result rows must include all simple RAG fields plus:

- `metadata_filters`
- `retrieval_mode`
- `graph_expanded_law_ids`
- `graph_expanded_chunk_ids`
- `reranked_chunk_ids`
- `context_included_count`
- `reference_law_hit`
- `failure_category`

Allowed failure categories:

- `retrieval_miss`
- `context_noise`
- `abstention`
- `contradiction`
- `generation_error`
- `judge_error`
- `unknown`

Graph expansion must use explicit relations from `edges.jsonl`. It must not invent inferred legal links.

## Quality Gates

- Advanced run records the simple RAG baseline it compares against.
- Every graph expansion is traceable to explicit edge evidence.
- Metadata filters are recorded per row.
- Reranking changes are observable through candidate order diagnostics.
- Optional hybrid retrieval can be disabled without breaking the pipeline.
- Metrics remain compatible with no-RAG and simple RAG.
- Failure categories are populated for failed or low-quality rows.

## Notebook Role

`notebooks/06_advanced_graph_rag.ipynb` should demonstrate one advanced query trace, show graph expansion evidence, compare simple and advanced metrics, and inspect failures by category.

The notebook should focus on why each advanced component helps or fails, not on adding production complexity.

## Acceptance Criteria

- Advanced RAG can be run with the same evaluation records as simple RAG.
- Each improvement is explainable through row-level diagnostics.
- Graph expansion uses only explicit legal relations.
- Metrics and failures can be compared directly with no-RAG and simple RAG.
- Optional advanced features can be switched off for ablation-style explanation.


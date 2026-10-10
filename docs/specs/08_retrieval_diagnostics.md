# 08 — Retrieval diagnostics

## Purpose

Inspect retrieval on the unchanged benchmark without generating or judging answers.
Article matching is a reference proxy, not legal correctness or evidence sufficiency.

## Inputs

`run_retrieval_diagnostics(benchmark, retriever, counter, output_dir,
profiles=('dense', 'sparse', 'hybrid', 'graph'), config=EngineConfig())` uses one
immutable index/corpus and the shared token counter. Only question text enters
search; gold and expected references are used after retrieval for diagnostics.

## Outputs

`manifest.json`, `records.jsonl` and `metrics.json` record identities, fixed QIDs,
profile, original pool, final selection, packed context, flags and stage timings.
Each record contains an `AnswerResult` explicitly labelled `retrieval_only=true`,
with `status='insufficient_evidence'` and no generated answer, score or LLM usage.
Existing output directories are rejected.

## Pipeline

1. Freeze benchmark, profile/config, code, index and tokenizer identities.
2. Search dense/sparse/hybrid using the same candidate budget and filter.
3. For graph, expand the hybrid pool using the same bounded one-hop policy.
4. Pack final candidates using the shared exact-source context builder.
5. Calculate article success/MRR at 5/10/100 and context article success against
   expected references. Keep original pool, final selection and context separate.
6. Export per-case records and fixed-denominator profile summaries and timings.

## Contract

Only deterministic retrieval profiles are accepted. Generation, query rewriting
and LLM reranking are excluded from this command. Failed retrieval is an error
record with unavailable proxies; it never becomes a correctness score. The 100
legacy intents remain explored; diagnostic matches cannot certify current law.
No precision, nDCG or evidence recall is reported from incomplete annotations.

## Quality Gates

- One record per QID/profile; source and output hashes are preserved.
- Identical candidate/context budgets and ceased-content policy across profiles.
- Gold never reaches the retriever; retrieval-only records contain no usage/score.
- Errors and proxy coverage are visible rather than removed from denominators.
- Original pool and final/context stages remain independently inspectable.

## Notebook Role

Optional notebooks call the runner or inspect saved records. No metric logic lives
in notebooks. Offline reports accept `diagnostic_dirs=()` as an additional input.

## Acceptance Criteria

Focused fake-retriever tests cover dense/hybrid/graph staging, failed retrieval,
no-gold leakage and report labels. A real run may execute all 100 questions with
the indexed local embeddings even when generation providers are unavailable.

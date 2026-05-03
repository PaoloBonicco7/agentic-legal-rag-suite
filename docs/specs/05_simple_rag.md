# 05 - Simple RAG Specification

## Purpose

Define the clearest possible RAG baseline using the indexed legal chunks.

This step proves that retrieval, context construction, answer generation, citations, and evaluation work end to end before any advanced improvement is added. It uses a single retrieval mode (dense vector search), a static context budget, and a minimal answer + citation contract. Hybrid retrieval, graph expansion, and reranking are explicitly excluded and are introduced only in step 06.

## Inputs

- Clean evaluation datasets from step 02 (`data/evaluation_clean/`).
- Retrieval-ready Qdrant index from step 03 (`data/indexes/qdrant`, with the index manifest under `data/indexing_runs/`).
- Clean legal dataset from step 01 for provenance and optional inspection.
- Simple RAG configuration (`SimpleRagConfig` Pydantic model):
  - `evaluation_dir: str` (default `data/evaluation_clean`).
  - `index_dir: str` (default `data/indexes/qdrant`).
  - `index_manifest_path: str` (default `data/indexing_runs/<latest>/index_manifest.json`).
  - `output_dir: str` (default `data/rag_runs/simple`).
  - `collection_name: str` (default `legal_chunks`).
  - `chat_model: str`, `judge_model: str | None` (defaults from `AGENTS.md`).
  - `prompt_version: str` (default `simple-rag-prompts-v1`).
  - `top_k: int` (default `5`): number of chunks fetched from Qdrant.
  - `max_context_chunks: int` (default `5`): hard cap on chunks placed in the context window.
  - `max_context_chars: int` (default `16000`): hard cap on total characters of concatenated chunk text.
  - `static_filters: dict[str, Any]` (default `{}`): metadata filters applied to every retrieval call (e.g., `{"law_status": "current"}`).
  - `benchmark_size: int | None`, `start: int`, `smoke: bool`, `retry_attempts: int` (consistent with `OracleEvaluationConfig`).
  - `env_file: str | None` (default `.env`).

## Outputs

Default generated output directory: `data/rag_runs/simple/`.

- `simple_rag_manifest.json`: evaluation dataset hashes, index manifest reference, model identities, prompt version, run configuration, and counts.
- `mcq_results.jsonl`: row-level MCQ answers with retrieved chunks and citations.
- `no_hint_results.jsonl`: row-level open answers, citations, judge results, and retrieved chunks.
- `simple_rag_summary.json`: aggregate metrics using the shared contract.
- `quality_report.md`: retrieval coverage, answer coverage, errors, and timing summary.

## Pipeline

1. Load evaluation records, the index manifest, and configuration.
   Why: the run must be traceable to fixed questions and a fixed index.
2. For each question, retrieve top-k chunks from Qdrant using dense search and `static_filters`.
   Why: dense retrieval with static filters is the minimal retrieval baseline.
3. Build a bounded context.
   Why: answer generation must use a clear and reproducible context budget. The context concatenates retrieved chunks in order until either `max_context_chunks` or `max_context_chars` is reached, whichever comes first.
4. Generate MCQ and no-hint answers using prompts from `simple_rag.prompts` (versioned by `prompt_version`).
   Why: both tasks must be comparable with the no-RAG baseline (step 04).
5. Attach citations.
   Why: answers must point to retrieved legal evidence actually present in the context.
6. Judge no-hint answers using the same `0-2` rubric as step 02b.
   Why: open-answer evaluation must use the same scoring shape as the baseline.
7. Export row-level traces and summaries.
   Why: retrieval misses and answer failures must be diagnosable without re-running the pipeline.

## Contract

Every result row must include:

- `qid`
- `level`
- `question`
- `retrieved_chunk_ids`: ordered list of chunk ids returned by Qdrant (length up to `top_k`).
- `retrieved_law_ids`: deduplicated list of `law_id` values across `retrieved_chunk_ids`.
- `context_chunk_ids`: ordered list of chunk ids actually placed in the context after applying `max_context_chunks` and `max_context_chars`.
- `answer`
- `citations`: list of citation objects (see below).
- `score` (MCQ) or `judge_score` and `judge_explanation` (no-hint).
- `error`: a short string when the row failed any stage; otherwise null.

MCQ rows additionally include `options`, `correct_label`, `predicted_label`. No-hint rows additionally include `predicted_answer`, `correct_answer`.

Each citation object must include:

- `law_id`
- `article_id`
- `chunk_id`
- `chunk_text`: the full text of the cited chunk as stored in the index payload.

A citation is valid only if `chunk_id` is contained in `context_chunk_ids`. Citations referring to retrieved-but-not-included chunks or to chunks outside the index are reported in `error` and counted in the quality report.

Simple RAG must not use graph expansion, reranking, hybrid retrieval, or query-conditional filters. The only filters allowed are the static ones declared in `static_filters` at run start.

Summary metrics use the shared contract from step 04 and step 02b: `processed`, `judged`, `score_sum`, `accuracy`, `coverage`, `strict_accuracy`, `errors`, `by_level`. For no-hint, `mean_score` and `max_score_sum` are also reported, consistent with step 02b.

## Quality Gates

- Index manifest and evaluation manifest hashes are recorded in `simple_rag_manifest.json`.
- `prompt_version` and model identities are recorded.
- Each processed row records `retrieved_count` and `context_count`.
- Empty retrieval is counted separately from generation errors.
- Every citation refers to a `chunk_id` present in `context_chunk_ids`; violations are reported.
- MCQ scoring is deterministic (label match) and follows the same rules as step 02b.
- No-hint judge scores are integers in `{0, 1, 2}` with non-empty explanation, otherwise the row is marked as `judge_error`.
- Summary metric names match the no-RAG baseline (step 04) so that step 07 can compare directly.
- Row-level traces are sufficient to inspect every wrong or empty answer.

## Notebook Role

`notebooks/05_simple_rag.ipynb` should:

- run a small benchmark in smoke mode and one full benchmark run;
- show one MCQ and one no-hint query end to end (question, retrieved chunks, context, answer, citations, score);
- compare global and by-level metrics against the no-RAG baseline (step 04) using shared metric names;
- inspect a handful of common failure cases (retrieval miss, citation outside context, generation error).

The notebook should make the basic RAG flow understandable before any advanced component is introduced in step 06. It must not redefine metrics — only display them.

## Acceptance Criteria

- Simple RAG can run end to end on MCQ and no-hint evaluation records using the index produced in step 03.
- Results are comparable with no-RAG (step 04) using shared metrics.
- Citations are grounded: every cited chunk is in the context window of its row.
- Retrieval, context, and generation failures are visible in row-level outputs and aggregated in the quality report.
- The simple pipeline remains small enough to be explained as the baseline RAG architecture in the thesis, with no graph, hybrid, or rerank logic involved.

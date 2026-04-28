# 05 - Simple RAG Specification

## Purpose

Define the clearest possible RAG baseline using the indexed legal chunks.

This step proves that retrieval, context construction, answer generation, citations, and evaluation work end to end before advanced improvements are added.

## Inputs

- Clean evaluation datasets from step 02.
- Retrieval-ready index from step 03.
- Clean legal dataset metadata from step 01 for provenance and optional graph inspection.
- Simple RAG configuration: top-k, filters, context size, prompt version, model identity, judge identity, and benchmark size.

## Outputs

Default generated output directory: `data/rag_runs/simple/`.

- `simple_rag_manifest.json`: dataset hashes, index manifest reference, model identities, prompt versions, and run configuration.
- `mcq_results.jsonl`: row-level MCQ answers with retrieved chunks.
- `no_hint_results.jsonl`: row-level open answers, citations, judge results, and retrieved chunks.
- `simple_rag_summary.json`: aggregate metrics.
- `quality_report.md`: retrieval coverage, answer coverage, errors, and timing summary.

## Pipeline

1. Load evaluation records and index contract.
   Why: the run must be traceable to fixed questions and a fixed index.
2. Retrieve top-k chunks for each question.
   Why: retrieval is the main difference from the no-RAG baseline.
3. Build a bounded context.
   Why: answer generation must use a clear and reproducible context budget.
4. Generate MCQ and no-hint answers.
   Why: both tasks should be comparable with the no-RAG baseline.
5. Attach citations.
   Why: answers must point to retrieved legal evidence.
6. Judge no-hint answers.
   Why: open-answer evaluation must use the same scoring shape as the baseline.
7. Export row-level traces and summaries.
   Why: retrieval misses and answer failures must be diagnosable.

## Contract

Every result row must include:

- `qid`
- `level`
- `question`
- `retrieved_chunk_ids`
- `retrieved_law_ids`
- `context_chunk_ids`
- `answer`
- `citations`
- `score`
- `error`

Each citation must identify at least `law_id`, `article_id`, `chunk_id`, and source text span or chunk text reference.

Simple RAG must not use graph expansion, reranking, or hybrid retrieval. It is the minimal retrieval baseline.

## Quality Gates

- Index manifest and evaluation manifest are recorded.
- Each processed row records retrieved count and context count.
- Empty retrieval is counted separately from generation errors.
- Citations must refer only to chunks included in context.
- Metrics use the same summary contract as no-RAG.
- Row-level traces are sufficient to inspect wrong answers.

## Notebook Role

`notebooks/05_simple_rag.ipynb` should run a small benchmark, inspect one query end to end, show retrieved chunks and citations, compare against no-RAG metrics, and display common failure cases.

The notebook should make the basic RAG flow understandable before advanced components are introduced.

## Acceptance Criteria

- Simple RAG can run end to end on MCQ and no-hint evaluation records.
- Results are comparable with no-RAG using shared metrics.
- Citations are grounded in retrieved context.
- Retrieval and generation failures are visible in row-level outputs.
- The simple pipeline remains small enough to explain as the baseline RAG architecture.


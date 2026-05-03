# 04 - No-RAG Baseline Specification

## Purpose

Evaluate model behavior without retrieval.

This step provides the scientific baseline for the thesis: it shows what the model can answer from its own behavior before the legal corpus is used as retrieval context.

## Inputs

- Clean evaluation datasets from step 02.
- Baseline configuration (`NoRagConfig` Pydantic model): `chat_model`, `judge_model`, `prompt_version` (default `no-rag-prompts-v1`), `benchmark_size`, `start`, `smoke`, `retry_attempts`, `random_seed` if sampling is used, `env_file` (default `.env`).

No legal retrieval index is used in this step.

LLM calls follow the project stack defined in `AGENTS.md`: Utopia structured chat with `temperature=0` for deterministic answers; the resolved seed (when supported by the model) is recorded in the manifest.

## Outputs

Default generated output directory: `data/baseline_runs/no_rag/`.

- `no_rag_manifest.json`: run configuration, dataset hash, model identities, and counts.
- `mcq_results.jsonl`: row-level MCQ predictions and scores.
- `no_hint_results.jsonl`: row-level open answers, judge results, and scores.
- `no_rag_summary.json`: aggregate metrics.
- `quality_report.md`: run status, errors, and coverage.

## Pipeline

1. Load the clean evaluation datasets.
   Why: all benchmarks must use the same validated question records.
2. Run MCQ answering without retrieval.
   Why: MCQ accuracy gives a direct model-only comparison point.
3. Run no-hint answering without retrieval.
   Why: open-answer behavior is closer to final RAG usage.
4. Judge no-hint answers.
   Why: open answers need a consistent scoring mechanism for comparison.
5. Aggregate metrics.
   Why: later RAG methods need the same metric shape for direct comparison.
6. Export row-level results.
   Why: failure cases must be inspectable, not only summarized.

## Contract

MCQ result rows must include:

- `qid`
- `level`
- `question`
- `options`
- `predicted_label`
- `correct_label`
- `score`
- `error`

MCQ scoring is deterministic and identical to step 02b:

- `score=1` when `predicted_label` equals `correct_label`;
- `score=0` when `predicted_label` is a valid label other than `correct_label`;
- invalid or missing labels are recorded in `error` and do not count toward `judged`.

No-hint result rows must include:

- `qid`
- `level`
- `question`
- `predicted_answer`
- `correct_answer`
- `judge_score`
- `judge_explanation`
- `error`

No-hint judge scoring uses the same `0-2` rubric as step 02b: `2` correct or semantically equivalent, `1` partially correct and not contradictory, `0` wrong, contradictory, empty, or not evaluable. Out-of-range scores or missing explanations are recorded as `judge_error`.

Summary metrics must include:

- `processed`
- `judged`
- `score_sum`
- `max_score_sum` (equals `judged` for MCQ, `2 * judged` for no-hint)
- `accuracy`
- `mean_score`
- `coverage`
- `strict_accuracy`
- `errors`
- `by_level`

## Quality Gates

- Evaluation dataset hash is recorded.
- Model and prompt versions are recorded.
- Processed count matches the selected benchmark size.
- Errors are counted and surfaced.
- No-hint answers are judged only when answer generation succeeds.
- Metrics use the shared names required by later comparison steps.

## Notebook Role

`notebooks/04_no_rag_baseline.ipynb` should run a small benchmark, show example answers, display global and by-level metrics, and inspect a few errors.

The notebook should explain why this baseline matters before introducing retrieval.

## Acceptance Criteria

- No-RAG MCQ and no-hint results are reproducible for a fixed configuration.
- Summary metrics can be compared directly with simple and advanced RAG results.
- Row-level outputs make wrong or failed cases inspectable.
- The baseline can be explained without any retrieval-specific logic.


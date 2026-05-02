# 02b - Oracle Context Evaluation Specification

## Purpose

Evaluate how much the model improves when it receives the exact legal articles cited by the evaluation dataset.

This step is a controlled evaluation, not a retrieval benchmark. It isolates the value of source-of-truth context by comparing model answers with and without oracle legal context across MCQ and no-hint questions.

## Inputs

- Clean MCQ dataset from step 02: `data/evaluation_clean/questions_mcq.jsonl`.
- Clean no-hint dataset from step 02: `data/evaluation_clean/questions_no_hint.jsonl`.
- Evaluation manifest from step 02: `data/evaluation_clean/evaluation_manifest.json`.
- Clean legal dataset from step 01: `data/laws_dataset_clean/laws.jsonl`, `data/laws_dataset_clean/articles.jsonl`, and `data/laws_dataset_clean/manifest.json`.
- Oracle evaluation configuration: answer model identity, judge model identity, prompt versions, benchmark size, random seed if sampling is used, retry policy, and output directory.

The oracle context must come only from the legal references already present in the evaluation dataset. No retrieval index is used in this step.

## Outputs

Default generated output directory: `data/evaluation_runs/oracle_context/`.

- `oracle_context_manifest.json`: run configuration, dataset hashes, legal dataset hash, model identities, prompt versions, and counts.
- `source_truth_contexts.jsonl`: resolved source-of-truth contexts for each question.
- `mcq_no_context_results.jsonl`: row-level MCQ predictions without context.
- `mcq_oracle_context_results.jsonl`: row-level MCQ predictions with oracle article context.
- `no_hint_no_context_results.jsonl`: row-level no-hint answers and judge results without context.
- `no_hint_oracle_context_results.jsonl`: row-level no-hint answers and judge results with oracle article context.
- `oracle_context_summary.json`: global metrics, by-level metrics, and context-vs-no-context deltas.
- `quality_report.md`: context resolution status, errors, coverage, and representative examples.

Generated outputs are benchmark artifacts and should not replace the source datasets.

## Pipeline

1. Load the clean evaluation datasets and manifests.
   Why: all four runs must use the same validated question records and stable source hashes.
2. Resolve expected legal references to clean legal records.
   Why: oracle context must be traceable to explicit source-of-truth references, not retrieved evidence.
3. Build oracle article contexts.
   Why: the context condition should contain the cited legal articles only, with minimal metadata and no fallback retrieval.
4. Run MCQ answering without context.
   Why: this provides the deterministic model-only MCQ baseline.
5. Run MCQ answering with oracle article context.
   Why: this measures whether exact legal context improves label selection.
6. Run no-hint answering without context.
   Why: this measures open-answer behavior on the harder dataset without legal evidence.
7. Run no-hint answering with oracle article context.
   Why: this measures open-answer behavior when the model receives the source-of-truth articles.
8. Judge no-hint answers.
   Why: open answers require a stable semantic scoring mechanism using the official correct answer.
9. Aggregate and compare metrics.
   Why: the experiment should show context deltas globally and by difficulty level.
10. Export row-level traces and quality diagnostics.
    Why: improved, unchanged, and degraded answers must be inspectable.

## Contract

Each `source_truth_contexts.jsonl` row must include:

- `qid`
- `level`
- `expected_references`
- `resolved_references`
- `context_article_ids`
- `context_text`
- `context_hash`
- `error`

Each resolved reference must include:

- `reference_text`
- `law_id`
- `law_title`
- `article_id`
- `article_label_norm`
- `article_text`

Reference parsing must support multiple references separated by `|`. Contexts must preserve the reference order from the evaluation dataset.

Article labels must be normalized consistently with step 01. Labels such as `4 bis`, `10 bis`, and `30 quater` must resolve to article labels such as `4bis`, `10bis`, and `30quater`.

If a reference cannot be resolved to both a law and an article, the affected row must be marked with a context error. The pipeline must not silently fall back to a full law, retrieval result, or approximate match.

Each MCQ result row must include:

- `qid`
- `level`
- `question`
- `options`
- `correct_label`
- `predicted_label`
- `score`
- `context_article_ids`
- `error`

MCQ scoring is deterministic:

- `score=1` when `predicted_label` equals `correct_label`;
- `score=0` when `predicted_label` is a valid but wrong label;
- invalid or missing labels are errors and do not count as judged.

Each no-hint result row must include:

- `qid`
- `level`
- `question`
- `predicted_answer`
- `correct_answer`
- `judge_score`
- `judge_explanation`
- `context_article_ids`
- `error`

No-hint judge scoring uses a `0-2` scale:

- `2`: the answer is correct or semantically equivalent to the official correct answer;
- `1`: the answer is partially correct, incomplete, and not contradictory;
- `0`: the answer is wrong, contradictory, empty, ambiguous, or not evaluable.

The judge must receive the question, the official correct answer, and the model answer. It must not receive MCQ alternatives as a shortcut for no-hint scoring.

Summary metrics must include:

- `processed`
- `judged`
- `score_sum`
- `max_score_sum`
- `accuracy`
- `mean_score`
- `coverage`
- `strict_accuracy`
- `errors`
- `by_level`
- `delta_oracle_minus_no_context`

For MCQ, `max_score_sum` equals the judged count. For no-hint, `max_score_sum` equals `2 * judged`.

## Quality Gates

- Evaluation and legal dataset hashes are recorded.
- All selected records have source-of-truth references.
- All selected references are parsed into law and article components.
- Every parsed law reference resolves to a `law_id`.
- Every parsed article reference resolves to an `article_id`.
- Oracle contexts include only cited article text and minimal metadata.
- MCQ outputs contain exactly one valid label when judged.
- No-hint judge outputs contain a valid `0-2` score and explanation when judged.
- All four runs process the same question set.
- Context-vs-no-context deltas are computed globally and by level.
- Errors are surfaced in row-level outputs and the quality report.

## Notebook Role

`notebooks/02b_oracle_context_evaluation.ipynb` should demonstrate this controlled experiment without hiding the setup.

The notebook should:

- load the clean evaluation datasets and oracle contexts;
- show one MCQ example and one no-hint example with and without oracle context;
- run the four configured evaluations;
- display MCQ accuracy, no-hint mean score, strict accuracy, coverage, and by-level breakdowns;
- show context-vs-no-context deltas;
- inspect examples where oracle context improves, does not change, or degrades the answer.

The notebook should explain that this step measures upper-bound context usefulness, not retrieval quality.

## Test Plan

- Parse and resolve all no-hint dataset references: 100 questions, 106 article references, all resolvable.
- Validate article label normalization for `4 bis`, `10 bis`, and `30 quater`.
- Validate MCQ/no-hint alignment by `qid` and question stem.
- Validate MCQ scoring for correct labels, wrong labels, and invalid labels.
- Validate judge schema acceptance for scores `0`, `1`, and `2`.
- Reject judge outputs outside the `0-2` scale or without an explanation.
- Validate global and by-level aggregations, including coverage, strict accuracy, and oracle-vs-no-context deltas.

## Acceptance Criteria

- The step can run the same selected questions across all four evaluation conditions.
- Oracle contexts are fully traceable to explicit evaluation references and clean legal articles.
- MCQ results are scored deterministically by label.
- No-hint results are judged with the official correct answer using a `0-2` rubric.
- Context-vs-no-context improvements are visible globally and by difficulty level.
- Row-level outputs make answer, context, judge, and error behavior inspectable.
- The experiment remains clearly separate from no-RAG baseline and RAG retrieval benchmarks.

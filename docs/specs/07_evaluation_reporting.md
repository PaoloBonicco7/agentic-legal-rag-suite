# 07 - Evaluation Reporting Specification

## Purpose

Define the shared reporting layer for the thesis PoC.

This step proves the project results in a consistent, explainable way by comparing no-RAG, simple RAG, and advanced Graph RAG with common metrics and failure analysis.

## Inputs

- No-RAG outputs from step 04.
- Simple RAG outputs from step 05.
- Advanced Graph RAG outputs from step 06.
- Evaluation dataset manifest from step 02.
- Optional indexing and preprocessing manifests for traceability.

## Outputs

Default generated output directory: `data/reports/`.

- `comparison_summary.json`: global metrics and deltas across methods.
- `comparison_by_level.json`: metrics by question level.
- `failure_analysis.json`: failure categories and representative examples.
- `thesis_tables.md`: compact tables suitable for documentation or thesis drafting.
- `report_manifest.json`: source run references and hashes.
- `quality_report.md`: reporting completeness and consistency checks.

## Pipeline

1. Load run manifests and summaries.
   Why: the report must compare known, reproducible runs.
2. Validate metric compatibility.
   Why: methods can only be compared when they use the same metric contract.
3. Compute global comparisons.
   Why: the thesis needs a clear headline result.
4. Compute by-level comparisons.
   Why: benchmark difficulty levels should show where retrieval helps or fails.
5. Build failure analysis.
   Why: failure categories explain more than aggregate accuracy.
6. Select representative examples.
   Why: qualitative cases make the result easier to discuss with a research team.
7. Export thesis-ready artifacts.
   Why: final results should be reusable outside notebooks.

## Contract

Every compared method must expose:

- `processed`
- `judged`
- `score_sum`
- `accuracy`
- `coverage`
- `strict_accuracy`
- `errors`
- `by_level`

Comparison outputs must include:

- no-RAG vs simple RAG deltas;
- simple RAG vs advanced Graph RAG deltas;
- no-RAG vs advanced Graph RAG deltas;
- separate MCQ and no-hint summaries;
- references to source run manifests.

Failure analysis should preserve row-level identifiers so examples can be traced back to original questions and run outputs.

## Quality Gates

- All source run manifests exist.
- Compared runs use the same evaluation dataset hash.
- Compared summaries expose the required metric fields.
- Row-level result counts match summary counts.
- Deltas are computed separately for MCQ and no-hint.
- Report manifest records all source runs.
- Missing or incompatible runs are reported instead of silently ignored.

## Notebook Role

`notebooks/07_evaluation_reporting.ipynb` should load completed run artifacts, display comparison tables, show charts, inspect representative failures, and summarize the thesis-relevant conclusions.

The notebook should be the final narrative layer, not the place where metrics are redefined.

## Acceptance Criteria

- No-RAG, simple RAG, and advanced Graph RAG can be compared with one shared metric contract.
- Reports are traceable to exact run artifacts.
- Global, by-level, and failure-category views are available.
- The final notebook can demonstrate the project outcome without rerunning every previous step.


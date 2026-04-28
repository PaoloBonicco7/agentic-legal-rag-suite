# 02 - Evaluation Dataset Specification

## Purpose

Prepare the versioned evaluation question files for reproducible benchmarking across no-RAG, simple RAG, and advanced RAG.

This step proves that the benchmark data is clean, aligned, and explainable before any model or retrieval system is evaluated.

## Inputs

- Source MCQ file: `data/evaluation/questions.csv`.
- Source no-hint file: `data/evaluation/questions_no_hint.csv`.

The source files are part of the application and should not be overwritten by generated normalized outputs.

## Outputs

Default generated output directory: `data/evaluation_clean/`.

- `questions_mcq.jsonl`: normalized multiple-choice records.
- `questions_no_hint.jsonl`: normalized open-answer records.
- `evaluation_manifest.json`: source hashes, counts, level distribution, and validation status.
- `evaluation_profile.json`: notebook-friendly summary of records, levels, references, and examples.
- `quality_report.md`: human-readable validation report.

## Pipeline

1. Load source CSV files.
   Why: keep the versioned source files as the raw benchmark authority.
2. Normalize column names and whitespace.
   Why: make downstream evaluation independent from CSV formatting details.
3. Parse MCQ options and correct labels.
   Why: structured MCQ evaluation requires explicit options and a correct answer label.
4. Normalize no-hint questions.
   Why: open-answer evaluation needs the same question intent without answer choices.
5. Align MCQ and no-hint records.
   Why: comparisons must evaluate the same underlying question in both formats.
6. Normalize legal references and levels.
   Why: level breakdowns and reference coverage diagnostics must be stable.
7. Export and validate.
   Why: every later benchmark should consume the same clean evaluation contract.

## Contract

Each MCQ record must include:

- `qid`
- `source_position`
- `level`
- `question_stem`
- `options`
- `correct_label`
- `correct_answer`
- `expected_references`

Each no-hint record must include:

- `qid`
- `source_position`
- `level`
- `question`
- `correct_answer`
- `expected_references`
- `linked_mcq_qid`

`qid` values must be stable across runs. Levels must preserve the source benchmark levels. Expected references must remain human-readable and should not be silently dropped.

## Quality Gates

- Source files exist and are readable.
- MCQ and no-hint datasets have the same number of valid records.
- MCQ and no-hint records align by position and question intent.
- Every MCQ record has options, a correct label, and a correct answer.
- Every no-hint record has a question, correct answer, level, and linked MCQ record.
- No required fields are empty.
- Level distribution is reported.
- Source hashes are recorded in the manifest.

## Notebook Role

`notebooks/02_evaluation_dataset.ipynb` should show source counts, normalized examples, MCQ/no-hint alignment, level distribution, reference examples, and validation results.

The notebook should explain why the no-hint dataset is needed and how it supports fair comparison with RAG systems.

## Acceptance Criteria

- Clean MCQ and no-hint datasets are generated from the versioned CSV files.
- Records are aligned and traceable to source positions.
- The dataset is ready for all later evaluation steps.
- Quality gates make benchmark data issues visible before model evaluation.


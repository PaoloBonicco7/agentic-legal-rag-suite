# 02 - Evaluation Dataset Results

## Implementation Summary

This step implements a deterministic CSV-to-clean-evaluation-dataset pipeline in `src/legal_rag/evaluation_dataset/`.

The pipeline reads the versioned source files from `data/evaluation/`, filters empty trailing MCQ rows, parses multiple-choice options, validates alignment with the no-hint dataset, and writes generated artifacts to `data/evaluation_clean/`.

The source CSV files are not modified. The generated output directory is ignored by Git and can be rebuilt from the versioned source data.

## Run Configuration

- MCQ source: `data/evaluation/questions.csv`
- No-hint source: `data/evaluation/questions_no_hint.csv`
- Output directory: `data/evaluation_clean`
- Expected valid records: `100`
- Schema version: `evaluation-dataset-v1`
- MCQ source hash: `b7f8038649ff7ac5cdff68aab574348013ed4fcb4c92dbef14418f27aa4763a2`
- No-hint source hash: `73345dcdc5f8e41693ff032718ededeae11108779c3c8fddb19775cbf5090d45`

## Observed Results

- MCQ source rows: `163`
- MCQ valid records: `100`
- MCQ dropped empty rows: `63`
- No-hint source rows: `100`
- No-hint valid records: `100`
- No-hint dropped empty rows: `0`
- Level distribution: `L1=25`, `L2=25`, `L3=25`, `L4=25`

All quality gates passed and `evaluation_manifest.json` exposes `ready_for_evaluation: true`.

## Quality Gates

- Source files readable: passed
- Expected record count: passed
- MCQ/no-hint record counts match: passed
- Required fields are present: passed
- Required fields are non-empty: passed
- Stable qids are aligned: passed
- Level distribution is reported: passed
- Source hashes are recorded: passed
- Output files exist and have hashes: passed

## Known Limitations

- Legal references are preserved as human-readable strings; this step does not parse them into structured law/article identifiers.
- Alignment is intentionally strict: MCQ stem, no-hint question, level, and answer text must match after whitespace normalization.
- `evaluation_manifest.json` excludes a self-hash because a file cannot contain a stable hash of itself.

## Reproducibility

Run from the repository root:

```bash
PYTHONPATH=src python -m legal_rag.evaluation_dataset --mcq-source data/evaluation/questions.csv --no-hint-source data/evaluation/questions_no_hint.csv --output data/evaluation_clean
```

The supporting notebook is `notebooks/02_evaluation_dataset.ipynb`.

# 01 - Laws Preprocessing Results

## Implementation Summary

This step implements a deterministic HTML-to-clean-dataset pipeline in `src/legal_rag/laws_preprocessing/`.

The pipeline reads the versioned source corpus from `data/laws_html/`, ignores non-source files such as `.DS_Store`, extracts legal structure and explicit relations, builds RAG-ready chunks, validates the output contract, and writes generated artifacts to `data/laws_dataset_clean/`.

The source corpus is not modified. The generated output directory is ignored by Git and can be rebuilt from the versioned source data.

For the high-level methodology behind this transformation, see [`docs/notes/01_laws_preprocessing_methodology.md`](../notes/01_laws_preprocessing_methodology.md).

## Run Configuration

- Source directory: `data/laws_html`
- Output directory: `data/laws_dataset_clean`
- Chunk size: `600` words
- Chunk overlap: `80` words
- Strict mode: `false`
- Parser backend: `lxml.html`
- Source hash: `aa46ea3758c1de5596a8902b95e90cdfc6d08fbc2956086495a020680be22459`

## Observed Results

- Valid HTML law files: `3145`
- Ignored files: `1` (`.DS_Store`)
- Laws: `3145`
- Articles: `17774`
- Passages: `76390`
- Notes: `8380`
- Edges: `35159`
- Chunks: `76467`
- Unresolved references: `277`
- Dropped duplicate IDs: `0`

All quality gates passed and `manifest.json` exposes `ready_for_indexing: true`.

## Quality Gates

- Valid source HTML found: passed
- Stable IDs are non-empty and duplicate-free: passed
- Required chunk fields are present: passed
- List metadata fields are real lists: passed
- Clean graph edges have no self-loops: passed
- Relation types are within the allowed contract: passed
- Law statuses are within the allowed contract: passed
- Output files exist and have hashes: passed
- `chunks.jsonl` is non-empty: passed

## Known Limitations

- Relations are extracted only from explicit hyperlink and citation evidence; no speculative legal inference is attempted.
- Reference resolution is limited to laws present in the local corpus and deterministic citation patterns.
- The `lxml.html` parser preserves a few malformed/nested note and link structures that the previous stdlib parser skipped, which slightly increases notes, edges, chunks, and unresolved reference counts.
- `manifest.json` excludes a self-hash because a file cannot contain a stable hash of its complete final contents.

## Reproducibility

Run from the repository root:

```bash
PYTHONPATH=src python -m legal_rag.laws_preprocessing --source data/laws_html --output data/laws_dataset_clean
```

The supporting notebook is `notebooks/01_laws_preprocessing.ipynb`.

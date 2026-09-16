# Validity audit: consolidation and controlled rerun

## Review on 2026-09-16

The audit branch descends from `main` at `5940c5e`. Its changes are integrated into
`main` after the preprocessing regression tests. The local
`backup-filter-repo-rewritten-20260603` branch has unrelated, pre-rewrite history;
its `src/` tree and `pyproject.toml` are identical to `5940c5e`. Its historical
documents and generated outputs are preserved under the archive tag
`archive/backup-filter-repo-rewritten-20260603`, rather than restored into main.

The review found three additional parser defects:

- Ambiguous or narrower passage notes could overwrite a status inherited from a
  resolved whole-law/article cessation. Whole-target cessation now remains binding.
- Duplicate note anchors produced ambiguous events without candidate targets,
  leaving affected text implicitly current. Candidate targets now receive unknown.
- Note links in article headings were discarded before backlink resolution.
  They now attach to the article without creating a synthetic passage.

These are corrections to the existing `legal-status-rules-v1` contract, not a new
status taxonomy. Source-code hashes distinguish the corrected implementation.
The July clean dataset and index predate these fixes and remain historical artifacts.
Their documented counts are not predictions for the next run.

## What the reference review establishes

The versioned `data/evaluation/vigency_reference_review.csv` covers the ten QIDs
excluded by the historical filters. It does not certify the rest of the benchmark.

| QIDs | Finding in the corpus snapshot |
|---|---|
| 0002, 0073, 0075, 0100 | Partial article cessation was incorrectly treated as whole-article cessation; answer-support passages remain active. |
| 0003 | Note contamination caused a false past classification; comma 2 supports the answer, while other editorial evidence is ambiguous. |
| 0013–0016 | LR 44/1991 is genuinely past in the corpus snapshot. |
| 0013 | The answer is supported by article 3, not expected article 2. |
| 0076 | Expected article 5 is past; the answer is supported by article 3 comma 5. |
| 0100 | Applicability to the 2020 emergency cannot be determined from status alone. |

The question number is prefixed with `eval-`. Categories overlap: a historical
reference can also have an incorrect qrel. Original questions and qrels remain fixed.
`current` means no explicit cessation detected in this snapshot, not legal certification.

A read-only re-ingestion of the four source laws on 2026-09-16 reproduced all ten
reviewed article statuses with the corrected parser. Supporting passages remained
current for 0002, 0003, 0073, 0075, 0076 and 0100; those for 0013, 0015 and 0016
remained past. The support for 0014 is annotated at article level and remained past.
This targeted check does not replace the full corpus regeneration or retrieval audit.

All 23 preprocessing tests passed, including five new regression cases. The complete
suite was attempted but interrupted while importing local dependencies from the
iCloud-hosted virtual environment; it has no current pass/fail verdict. Rerunning
the full suite is the first remaining validation gate. No full preprocessing,
indexing, or retrieval job was launched during this review.

## Execution sequence

Run these commands from the repository root, on a clean `main`. Long jobs are run
by the researcher, one stage at a time. No Docker or Qdrant server is required.
Close notebook kernels using the same local Qdrant directory before indexing.

### 1. Unit tests

```bash
.venv/bin/python -m pytest -q
```

### 2. Regenerate preprocessing into a new directory

```bash
PYTHONPATH=src .venv/bin/python -m legal_rag.laws_preprocessing \
  --source data/laws_html \
  --output data/laws_dataset_clean_status_v2_review_20260916 \
  --chunk-size 600 --chunk-overlap 80
```

Inspect `quality_report.md` and `manifest.json`: `ready_for_indexing` must be true,
Git provenance clean, and all gates satisfied. Recheck the ten reviewed references
against the new events and passage states before interpreting retrieval scores.
The benchmark normalization can be rerun independently with:

```bash
PYTHONPATH=src .venv/bin/python -m legal_rag.evaluation_dataset
```

### 3. Build an isolated full index

The old index cannot acquire corrected metadata from a code merge. A rebuild is
required for the definitive audit, but unchanged embedding inputs can reuse their
verified vectors. This command uses the July full status-v2 index as the source:

```bash
PYTHONPATH=src .venv/bin/python -m legal_rag.indexing \
  --clean-dataset-dir data/laws_dataset_clean_status_v2_review_20260916 \
  --index-dir data/indexes/qdrant_status_v2_review_20260916 \
  --collection-name legal_chunks_bge_m3_status_v2_review_20260916 \
  --runs-dir data/indexing_runs \
  --run-id status_v2_review_20260916 \
  --embedding-backend local --embedding-model BAAI/bge-m3 --embedding-dim 1024 \
  --batch-size 64 --upload-batch-size 64 --chunk-selection-mode full \
  --force-rebuild --require-clean-worktree \
  --reuse-vectors-index-dir data/indexes/qdrant_status_v2 \
  --reuse-vectors-collection legal_chunks_bge_m3_status_v2 \
  --reuse-vectors-manifest-path data/indexing_runs/status_v2_full_20260723/index_manifest.json
```

Use a new run ID and target paths for subsequent definitive attempts. If the source
index is unavailable or incompatible, omit all three `--reuse-vectors-*` options;
this recomputes embeddings and takes longer. Verify `ready_for_retrieval`, exact
point counts, failures, payload distributions, and reuse/embedding counts.

### 4. Run the filter audit

In the same terminal, set all four identities explicitly so the notebook cannot
silently use the July defaults:

```bash
export RETRIEVAL_DIAGNOSTICS_LAWS_DIR="$PWD/data/laws_dataset_clean_status_v2_review_20260916"
export RETRIEVAL_DIAGNOSTICS_INDEX_DIR="$PWD/data/indexes/qdrant_status_v2_review_20260916"
export RETRIEVAL_DIAGNOSTICS_COLLECTION=legal_chunks_bge_m3_status_v2_review_20260916
export RETRIEVAL_DIAGNOSTICS_INDEX_MANIFEST="$PWD/data/indexing_runs/status_v2_review_20260916/index_manifest.json"
export RETRIEVAL_DIAGNOSTICS_RUN_NAME=validity_filter_audit_review_20260916
export RETRIEVAL_DIAGNOSTICS_MAX_WORKERS=1
PYTHONPATH=src .venv/bin/jupyter lab notebooks/06b_retrieval_diagnostics.ipynb
```

Execute cells in order. The setup cell performs full file/collection reconciliation
and loads the embedder; `run_filter_audit` is the long sweep. The final cell publishes
the completed artifact set. An interrupted run has no resumable checkpoint and
must not be used as terminal evidence. Keep the worktree clean during the run.

### 5. Interpret before running answer generation

First compare each filter with `none` on this same new index and the same QIDs.
Use `filter_reference_audit.csv`, `filter_exclusions.csv`, and the event evidence
to distinguish annotation errors, historical references, incorrect qrels, and
retrieval misses. `filter_impact.csv` contains paired effects and uncertainty;
`filter_exact_control.csv` is a local-mode sanity check, not an HNSW server benchmark.

The primary comparison remains no-hint Hybrid@10 for the two status views. No filter
is enabled automatically, even if a point estimate improves. Comparing the new
unfiltered run with the historical 73/100 Dense@10 is descriptive: corpus parsing
and index identity have changed, so that delta alone is not a filter effect.

After reviewing these artifacts, configure notebooks 05 and 06 to use the same new
clean dataset, manifest, and collection. Rerun Simple and Advanced with fixed model,
judge, prompts, QIDs, and explicit candidate/context budgets; compare paired answer
scores and confidence intervals. A no-RAG rerun is needed for a fully fresh
three-way comparison, particularly if the model endpoint has changed. The oracle
notebook 02b is an optional upper-bound control. Preserve historical runs and write
new run directories. Notebook 06b currently performs the filter audit only, not
the old F/G/H rewriting and reranking sweep.

Unit-test success establishes software regression coverage. A stable research
result additionally requires successful preprocessing, full indexing, complete
audit artifacts, and the subsequent end-to-end comparison. Those long executions
remain pending; no performance gain is claimed from this consolidation.

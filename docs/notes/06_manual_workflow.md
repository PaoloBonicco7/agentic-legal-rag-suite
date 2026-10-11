# Manual workflow

Run each stage yourself and inspect its output before continuing. No import,
notebook or report starts the complete experiment automatically. Existing
artifact directories are immutable: use a new output path for each build.
Run commands from the isolated checkout root containing `pyproject.toml`, on
`feature/legal-rag-rebuild`; the preserved original checkout contains the legacy
implementation.

## Available starting point

| Asset | Path / state |
| --- | --- |
| Frozen source | `data/snapshots/current_20261010`, 3,176 official HTML documents |
| Parsed corpus | `data/corpora/current-final`, 21,322 contextual chunks |
| Corpus inspection | `data/reports/current-final/index.html` |
| Explanatory examples | [HTML walkthrough](05_preprocessing_walkthrough.html) |
| Small Luna pilot | `data/reports/pilot_20261011/index.html`, 8 intents, three profiles |
| Full baseline and ablations | Not executed; separate future work |

These generated assets are local and ignored by Git. The delivery note records
the index completion state. Preserve the frozen source bytes when sharing the
experiment; the `bundle` command is provided but was not launched after the
execution stop.

## First: verify the software

```bash
uv run --no-sync pytest
```

The 2026-10-11 implementation passed 251 offline tests, including progress
callbacks and their CLI wiring, before the authorized pilot. The subsequent
reporter fix has separate focused tests. Rerun relevant checks after code changes.

## Follow the preprocessing

```bash
uv run --no-sync legal-rag preprocess --snapshot data/snapshots/current_20261010 --output data/corpora/manual01 --local-files-only
uv run --no-sync legal-rag report --corpus data/corpora/manual01 --output data/reports/manual01
```

`--local-files-only` requires the pinned tokenizer to be cached, as it is in
this checkout. Without it, the tokenizer may be downloaded; preprocessing does
not load embedding weights or call a language model.

| Progress stage | Inspect before moving on |
| --- | --- |
| `preflight` | Complete snapshot, catalog identity and source checksums |
| `parsing` | `blocks.jsonl`, regions, units and unstructured content |
| `graph_chunking` | Local reference evidence, unresolved destinations and complete embedding input |
| `quality` | All gates in `quality.json`; passing gates do not certify semantic accuracy |
| `export` | Final `manifest.json`, source hashes and `ready_for_indexing` |

Counts describe their named stage. A parsing count is not the number of indexed
chunks; `ready` stays false until verified export. Never infer completion from
the last progress line after an interruption.

Start with the walkthrough's real cases. Compare body and editorial notes,
inspect source spans, check that a stub has no invented article, and follow a
list introduction into its contextual chunk. Review warnings separately from
legal applicability. [Spec 02](../specs/02_preprocessing.md) defines the exact
contract; [spec 03](../specs/03_metadata_graph.md) defines conservative status.

## Then: index and experiments

Use the index command in README only after the corpus gates pass. The CLI reports
processed chunks and cache reuse; the manifest distinguishes embedding inference
batch from upload batch. Keep device, dtype and model identity explicit.

`diagnose` measures retrieval without a judge. `evaluate --qids ...` is a pilot;
the full baseline is a separate deliberate run. First restore provider access,
verify the live model, refresh sourced OpenAI prices and check the common budget
ledger. The [eight-intent Luna pilot](../results/07_luna_pilot.md) is complete;
its configuration is not the default full-benchmark command. Launch one profile
at a time. Generation checkpoints are persisted before
their progress notification; retrieval-only records are written at final export.

The comparison reporter reads saved artifacts and marks incompatible comparisons
as descriptive. It does not generate answers. Use unchanged questions, gold,
source, model, filters and final context budget to measure one strategy at a time.
The HTML walkthrough explains what each comparison can establish; it contains
no simulated quality scores.

Notebook flags default to false. Choose individual cells and calls explicitly;
all reusable logic remains in Python modules.

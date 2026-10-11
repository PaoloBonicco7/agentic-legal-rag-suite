# Legal RAG rebuild

A small research engine for stateless question answering over frozen Italian laws of Valle d’Aosta. The thesis hypothesis is that better retrieval increases fully correct answers. Software tests establish behavior; measured benchmark runs establish evidence for the hypothesis.

One package contains `corpus`, `indexing`, `retrieval`, `generation`, `evaluation`, `reporting` and `providers`. CLI, UI and notebooks call these modules. No framework, server Qdrant or separate graph database is required.

## Start here

- [Numbered workflow](docs/specs/README.md): inputs, outputs and quality gates.
- [Architecture decisions](docs/notes/00_architecture.md): reasons, alternatives and limits.
- [Implementation status](docs/results/00_delivery.md): actual artifacts and outstanding empirical work.
- [Preprocessing walkthrough](docs/notes/05_preprocessing_walkthrough.html): real source examples, transformations and limits.
- [Manual workflow](docs/notes/06_manual_workflow.md): stages to launch and inspect yourself.
- [Literature](docs/notes/01_literature.md): primary references and what they support.

## Setup

Python 3.11–3.13 and `uv` (this delivery was verified on Python 3.12):

```bash
uv sync --locked --default-index https://pypi.org/simple --extra local --extra dev --extra ui
cp .env.example .env
uv run --no-sync pytest
uv run --no-sync legal-rag --help
```

Fill the runtime keys in `.env`. No credential belongs in a config, manifest, notebook or commit. Use `--env-file /absolute/path/.env` before the command to reuse a private existing environment. `local`, `ui`, `notebooks` and `bm25` are optional dependency groups; BGE needs `local`.

The original research checkout remains intact. This branch deliberately replaces its implementation; history and the verified private source archive preserve the previous work. The 100 original evaluation intents and CSV gold are unchanged. Ignored HTML is preserved separately; a Git branch cannot preserve it.

## Frozen data to index

These commands describe a fresh build. In the delivered checkout the snapshot,
corpus and index already exist; use the inspection commands below. Rebuilds and
new experiments require new output directories, because completed artifacts are
immutable. See the delivery note for the actual paths and completion state.

```bash
uv run --no-sync legal-rag acquire --output data/snapshots/current_20261010
# After interruption only:
uv run --no-sync legal-rag acquire --output data/snapshots/current_20261010 --resume
uv run --no-sync legal-rag preprocess --snapshot data/snapshots/current_20261010 --output data/corpora/current-final
uv run --no-sync legal-rag index --corpus data/corpora/current-final --output data/indexes/current-bge
```

Acquisition refreshes every detail page, compares initial/final catalogues and keeps original bytes. A completed snapshot is immutable. The interval of acquisition is provenance, not a legal applicability certificate. Missing pages never imply repeal. Catalogue stability cannot guarantee an atomic view of every body during a long acquisition.

CPU, float32 and batch 2 are portable defaults. GPU execution is explicit (`--embedding-device mps --embedding-dtype float16`), with its dtype, padding policy and batch settings recorded. A short speed calibration is insufficient to establish stability; the delivery note records the configuration actually validated on the M1. Run only one model-loading/index-writing job. BGE revision is pinned, dense and native sparse share one encoding, and the complete embedding input must fit 512 tokens. Interrupted indexes remain incomplete: build into a new directory and reuse verified vector cache entries.

## Ask and inspect

```bash
uv run --no-sync legal-rag preflight --live --output data/diagnostics/utopia.json
uv run --no-sync legal-rag ask --index data/indexes/current-bge --corpus data/corpora/current-final 'Quali sono i requisiti previsti dalla legge?'
uv run --no-sync legal-rag ui --index data/indexes/current-bge --corpus data/corpora/current-final
uv run --no-sync legal-rag diagnose --index data/indexes/current-bge --output data/diagnostics/retrieval-manual01
uv run --no-sync legal-rag report --corpus data/corpora/current-final --diagnostics data/diagnostics/retrieval-manual01 --output data/reports/retrieval-manual01
```

Profiles: `no_rag`, `dense`, `sparse`, `hybrid`, `multiquery`, `graph`, `rerank`, `oracle`. Every profile uses the same context builder (4096 BGE tokens, 12 units). Current retrieval excludes only confirmed cessations with resolved scope. Unknown cases remain included. `--include-ceased` is an explicit alternative policy and must match across compared runs.

The form has no conversational memory. Citations quote only source text shown in the prompt. Candidate pool, selection, context, omissions and citations remain inspectable. Reports are static and offline: rendering never performs retrieval or model calls. Optional `--comments notes.jsonl` reads durable `ReportComment` records maintained separately through the reporting API; rebuilding HTML leaves annotations intact.

Utopia explicitly requests a 32768-token provider window. Use
`--context-window-tokens` to choose another window before freezing an experiment.
Controlled runs verify it against inspected model capacity. The full prompt,
schema and maximum output must fit a conservative bound; the provider never
receives a shortened context to make it fit. Gateway compliance still needs a
live smoke test after access is restored.

## Controlled experiments

```bash
uv run --no-sync legal-rag evaluate --index data/indexes/current-bge --corpus data/corpora/current-final --profile no_rag --output data/runs/no_rag
uv run --no-sync legal-rag evaluate --index data/indexes/current-bge --corpus data/corpora/current-final --profile dense --output data/runs/dense
uv run --no-sync legal-rag evaluate --index data/indexes/current-bge --corpus data/corpora/current-final --profile hybrid --output data/runs/hybrid
uv run --no-sync legal-rag compare data/runs/no_rag data/runs/dense data/runs/hybrid --output data/comparison.json
uv run --no-sync legal-rag report --corpus data/corpora/current-final --runs data/runs/no_rag data/runs/dense data/runs/hybrid --output data/reports/baseline
```

Utopia is the default generator; OpenAI Luna judges and Sol checks a fixed stratified sample of 20 QID. OpenAI requires a key and prices verified **today** in `configs/openai-pricing.json`, with a recent sourced exchange rate. The shared persistent ledger caps all attempts and retries at 20 EUR (16 experiment + 4 validation). Accounted amounts are conservative estimates/reservations, not invoiced costs. No fallback changes provider or model. `--without-control` runs a diagnostic comparison that cannot justify promotion.

Run one profile at a time. `--resume` keeps frozen answers and retries missing judgments only when code and scientific inputs match. `--qids ...` makes an explicitly marked pilot. `--mcq` adds the secondary view without doubling independent observations. Oracle requires `--oracle-evidence` with approved per-QID chunks; annotations never enter runtime prompts. Approved gold-only revisions can use `rejudge`; changed questions, evidence policy or sources require regeneration. Missing judgments keep a run incomplete. The same 100 intents have already informed development, so findings are exploratory.

For a small Luna-only pilot, set `--provider openai --model gpt-6-luna
--judge-model gpt-6-luna --without-control --qids ... --budget-eur 0.5
--budget-reserve-eur 0.1 --retries 0`. The budget is cumulative across all clients
using the same ledger, including earlier runs. Keep a new output directory per
profile; a pilot cannot authorize promotion.

Acquisition, preprocessing, indexing and evaluation expose stage/count progress
on stderr. stdout remains the final JSON result; dependencies may also print
diagnostics on stderr. Progress is observational, not proof of completion: use
the final manifest and quality gates. The offline suite passed 251 tests before
the [eight-intent Luna pilot](docs/results/07_luna_pilot.md) on 2026-10-11.
That pilot compares No-RAG, Dense and Hybrid; the full 100-intent benchmark
and Advanced ablations remain unexecuted.

## Portable experiment package

```bash
uv run --no-sync legal-rag bundle --snapshot data/snapshots/current_20261010 --corpus data/corpora/current-final --output data/bundles/current_20261010.tar.gz
uv run --no-sync legal-rag restore data/bundles/current_20261010.tar.gz --output data/restored/current
```

When packaging revised runs, add `--review-files /path/to/reviews.jsonl`; the raw
review file must match the run hash. Add `--runs` for saved evaluation, diagnostic
or annotation artifacts.

Bundles verify source, benchmark, corpus and run provenance before export and every checksum before restoration. They include frozen bytes, benchmark, source, lockfile and documentation; they exclude keys and model weights. Indexes are rebuilt from the restored corpus. Re-downloading a site does not recreate the experimental snapshot. Generated data, indexes, caches, runs and reports are ignored by Git. Keep the archive beside the source checkout when sharing a reproducible experiment.

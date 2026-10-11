# Manual workflow

Run from the isolated checkout containing `pyproject.toml`, on
`feature/legal-rag-rebuild`. The original checkout and verified private backup
preserve the previous research; original benchmark CSVs and gold are unchanged.
Imports and notebook defaults do not launch a full experiment. Completed
artifact directories are immutable: choose a fresh path for every new build/run.

## Inspect the delivered artifacts first

| Asset | Path / verified state |
| --- | --- |
| Frozen official source | `data/snapshots/current_20261010`, 3,176 documents acquired on 2026-10-10 |
| Corpus | `data/corpora/validated_20261011`, 21,322 chunks, 119,398 relations, 4,010 unresolved destinations |
| Local BGE-M3 index | `data/indexes/validated-bge_20261011`, 21,322 cached document vectors |
| Main report | `data/reports/validated_20261011/index.html`, seven no-hint profiles on four QIDs |
| Separate MCQ report | `data/reports/validated_20261011_mcq/index.html`, one QID and three profiles |
| Retrieval-only diagnostics | `data/diagnostics/validated_20261011/retrieval`, 16 records / four QIDs |
| Executed notebook previews | `data/notebooks/validated_20261011`, three notebooks and HTML previews |

These local generated assets are ignored by Git. The current code passed
313 offline tests. All three notebooks were executed with build/paid switches
false and made zero API calls. Their HTML previews are available, but browser
pixel review was not performed. Portable packaging/restoration verified 7,795
files, and restored source reproduced the 21 main comparisons without API
calls. Report links, checksums and offline resources passed static QA. See the
[delivery note](../results/00_delivery.md) for the package scope and limits.

To restore the frozen package into a fresh directory:

```bash
uv run --no-sync legal-rag restore data/bundles/validated_20261011.tar.gz --output data/restored/manual02
```

The archive includes source, benchmark, snapshot, corpus, ten evaluation runs
and retrieval diagnostics. It excludes credentials, the provider ledger,
indexes and model weights; index reconstruction requires the pinned local
model and explicit device/dtype. Saved responses do not require regeneration.

Open the saved report before rebuilding anything. It exposes source snippets,
rankings, selected context, answers, judges, errors and separate audit comments.
The [validation note](../results/08_pipeline_validation.md) explains the four-case
scores: No-RAG 2/4; Dense/Sparse 3/4; Hybrid 4/4; Multi-query 2/4;
Graph/Rerank 3/4. One provider failure and one header-quotation failure remain
in the denominator. Judge/source-identity concerns require human review;
no treatment is promoted. MCQ and retrieval proxies are separate diagnostics.

## Follow preprocessing and indexing

Use the [walkthrough](05_preprocessing_walkthrough.html) to inspect real source
cases. Compare body and editorial notes, verify source spans and list context,
and inspect unresolved references and identity warnings separately from legal
applicability. Snapshot/corpus status evidence does not certify current validity.

For a deliberate fresh local rebuild:

```bash
uv run --no-sync legal-rag preprocess --snapshot data/snapshots/current_20261010 --output data/corpora/manual01 --local-files-only
uv run --no-sync legal-rag report --corpus data/corpora/manual01 --output data/reports/manual01
```

`--local-files-only` requires the pinned tokenizer to be cached. Preprocessing
loads no embedding weights and calls no LLM. Inspect `quality.json` and the
final manifest before indexing; all structural gates and `ready_for_indexing`
must pass. Counts describe their stage, not completion of later stages.
[Spec 02](../specs/02_preprocessing.md) and [spec 03](../specs/03_metadata_graph.md)
define the contracts, including the three graph relation types.

Use the documented index command only for a ready corpus and a fresh output
path. Keep model revision, device and dtype explicit, run one model/index job
at a time, and reuse only verified cache entries. The manifest reconciles
payloads, IDs and corpus identity. Progress alone is not a completion gate.

## Launch only the intended experiment

The completed pilot used Luna only, local embeddings, four no-hint QIDs,
zero retries and no control judge. Enrichment's four proposals were not adopted.
The new work used 69 calls / 0.046987625 EUR; the cumulative ledger holds
115 calls / 0.068710875 EUR, including one uncertain reservation. These are
estimates/reservations, not invoices. No Sol/Utopia or paid BGE call was used.

A future explicitly authorized repeat can use the bounded settings below and
a new output directory; it still cannot justify promotion:

```bash
uv run --no-sync legal-rag evaluate --index data/indexes/validated-bge_20261011 --corpus data/corpora/validated_20261011 --profile hybrid --provider openai --model gpt-6-luna --judge-provider openai --judge-model gpt-6-luna --without-control --qids eval-0002 eval-0004 eval-0007 eval-0053 --retries 0 --budget-eur 0.5 --budget-reserve-eur 0.1 --local-files-only --output data/runs/manual01/hybrid
```

Refresh sourced prices/exchange rates and inspect the shared persistent ledger
before any paid batch. The ceiling is cumulative across clients/runs; a new
output directory does not reset it. Normal CLI defaults allow a larger budget
and Sol control calls, so they are not the bounded pilot configuration.
Utopia returned HTTP 403 in its earlier live preflight; verify access, actual
model identity and capacity before selecting it. There is no provider fallback.

Run one profile at a time and change one treatment while keeping question/gold,
corpus/index, generator, judge, filters and final context budget fixed.
`--resume` requires matching scientific identities and preserves saved answers.
Generation checkpoints are durable before their progress notification;
retrieval-only records are exported at completion. Verify final manifests.

Reports consume saved artifacts without retrieval or model calls. For new
reports, pass saved runs/diagnostics and `--comments` explicitly, with a fresh
output path. Notebook switches remain false unless you deliberately enable a
stage; all reusable logic stays in Python modules.

Next steps are human review of gold/source/judge disagreements, followed by an
explicitly authorized full 100-intent study with independent judge checks.
Approved revisions and oracle evidence have separate contracts; automatic
annotations are proposals, not approval. Preserve negative results and failures.

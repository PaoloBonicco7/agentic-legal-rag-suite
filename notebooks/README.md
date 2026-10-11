# Demonstration notebooks

Use the project environment with the `notebooks` extra, then start Jupyter from
the repository root. Edit the paths in each notebook's first code cell.

| Notebook | Purpose | Specifications |
| --- | --- | --- |
| `01_corpus.ipynb` | Inspect provenance, quality gates and records; optional deterministic rebuild. | 01–03 |
| `02_retrieval.ipynb` | Inspect saved retrieval proxies; optional local diagnostics. | 04, 05, 08 |
| `03_evaluation_reporting.ipynb` | Inspect benchmark/runs; optional evaluation and offline HTML. | 06, 07, 09 |

The companions point to corpus `data/corpora/validated_20261011`, index
`data/indexes/validated-bge_20261011` and report `data/reports/validated_20261011`.
Notebook 03 reads the seven saved profiles under `data/runs/validated_20261011/`:
`no_rag`, `dense`, `sparse`, `hybrid`, `multiquery`, `graph` and `rerank`.
Retrieval-only diagnostics for the same four QIDs are saved under
`data/diagnostics/validated_20261011/retrieval` when available.

All execution switches default to `False`. Running the notebooks with those
defaults only reads saved artifacts: no weights, index, embedding or API calls.
An enabled retrieval run loads embeddings and opens its index; an enabled
evaluation run calls configured providers. Use new output directories and avoid
concurrent index users or checkpoint writers.
The optional evaluation example uses only `gpt-6-luna` for generator and judge,
QIDs `eval-0002`, `eval-0004`, `eval-0007`, `eval-0053`, no control judge, zero
retries and a 0.50 EUR shared-ledger cap with 0.10 EUR reserved for validation.
This is an exploratory four-intent comparison, not the full benchmark.

Logic remains in `src/legal_rag/`; notebooks do not define parsers, retrieval,
metrics or prompts. Missing run directories mean results are unmeasured.
Source metadata and structural gates do not certify legal applicability, and
article matches do not establish answer correctness.

All three notebooks passed `nbformat` validation and top-to-bottom execution
in the project Jupyter kernel with all switches disabled. The budget ledger was
unchanged. Source notebooks keep outputs cleared; executed copies and HTML
previews are saved under `data/notebooks/validated_20261011`. Browser pixel-layout
inspection was unavailable. With the optional dependencies, validate locally:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyter nbconvert --execute --to notebook --output-dir /tmp/legal-rag-notebooks notebooks/*.ipynb
```

Keep execution switches disabled for that offline check. Optional live branches
need their own configured resources and were not executed during creation.

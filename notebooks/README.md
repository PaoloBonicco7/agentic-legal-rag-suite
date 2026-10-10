# Demonstration notebooks

Use the project environment with the `notebooks` extra, then start Jupyter from
the repository root. Edit the paths in each notebook's first code cell.

| Notebook | Purpose | Specifications |
| --- | --- | --- |
| `01_corpus.ipynb` | Inspect provenance, quality gates and records; optional deterministic rebuild. | 01–03 |
| `02_retrieval.ipynb` | Inspect saved retrieval proxies; optional local diagnostics. | 04, 05, 08 |
| `03_evaluation_reporting.ipynb` | Inspect benchmark/runs; optional evaluation and offline HTML. | 06, 07, 09 |

All execution switches default to `False`. Running the notebooks with those
defaults only reads saved artifacts: no weights, index, embedding or API calls.
An enabled retrieval run loads embeddings and opens its index; an enabled
evaluation run calls configured providers. Use new output directories and avoid
concurrent index users or checkpoint writers.

Logic remains in `src/legal_rag/`; notebooks do not define parsers, retrieval,
metrics or prompts. Missing run directories mean results are unmeasured.
Source metadata and structural gates do not certify legal applicability, and
article matches do not establish answer correctness.

Notebook JSON and code syntax are validated; default cells are checked in order
against local frozen artifacts. Stored outputs are intentionally cleared.
Full Jupyter-kernel execution requires the optional notebook dependencies:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyter nbconvert --execute --to notebook --output-dir /tmp/legal-rag-notebooks notebooks/*.ipynb
```

Keep execution switches disabled for that offline check. Optional live branches
need their own configured resources and were not executed during creation.

# Agentic Legal RAG Suite

Minimal, reproducible Legal RAG research codebase for a thesis project.

The project tests whether better retrieval improves legal question answering on a fixed benchmark of questions about Italian regional laws. It compares:

- **no-RAG**: the model answers without retrieved context;
- **simple RAG**: dense retrieval over the legal corpus;
- **advanced RAG**: hybrid dense + sparse retrieval with multi-query rewriting;
- **oracle context**: the model receives the expected legal articles, used as an upper reference.

The detailed workflow is in [`docs/specs/README.md`](docs/specs/README.md). The current result summary is in [`docs/results/00_overview.md`](docs/results/00_overview.md).

## Project Layout

- `src/legal_rag/`: reusable implementation for each pipeline step.
- `notebooks/`: demonstration notebooks; reusable logic should stay in `src/`.
- `docs/specs/`: compact contracts for each numbered step.
- `docs/results/`: run summaries and thesis-facing result notes.
- `data/evaluation/`: versioned source evaluation CSV files.
- `data/laws_html/`: local HTML legal corpus required by preprocessing, not tracked in Git.
- `data/*_clean`, `data/indexes`, `data/*_runs`, `data/reports`: generated artifacts.
- `OLD/`: historical reference only.

Generated datasets, indexes, caches, and benchmark runs are reproducible outputs. Do not treat them as source data.

## Setup

The project uses Python 3.11+ and `uv`.

Install the base development environment:

```bash
uv sync --group dev
```

Install notebook and UI dependencies when needed:

```bash
uv sync --group dev --group notebooks --group ui
```

Create a local `.env` file for Utopia-backed LLM calls:

```bash
UTOPIA_API_KEY=...
UTOPIA_BASE_URL=...
UTOPIA_CHAT_MODEL=SLURM.gpt-oss:120b
```

`UTOPIA_BASE_URL` and `UTOPIA_CHAT_MODEL` already match the project defaults, so only the API key is normally required.

## Basic Commands

Run tests:

```bash
PYTHONPATH=src uv run --no-sync pytest
```

Build the clean legal dataset from the local HTML corpus:

```bash
PYTHONPATH=src uv run --no-sync python -m legal_rag.laws_preprocessing
```

Build the clean evaluation dataset:

```bash
PYTHONPATH=src uv run --no-sync python -m legal_rag.evaluation_dataset
```

Run a smoke check for model connectivity and structured output:

```bash
PYTHONPATH=src uv run --no-sync python -m legal_rag.oracle_context_evaluation --smoke
PYTHONPATH=src uv run --no-sync python -m legal_rag.no_rag_baseline --smoke
```

Build a local Qdrant index:

```bash
PYTHONPATH=src uv run --no-sync python -m legal_rag.indexing --collection-name legal_chunks_bge_m3 --force-rebuild
```

Run simple and advanced RAG smoke checks:

```bash
PYTHONPATH=src uv run --no-sync python -m legal_rag.simple_rag --smoke
PYTHONPATH=src uv run --no-sync python -m legal_rag.advanced_graph_rag --smoke --run-name smoke --no-metadata-filters --no-graph-expansion --no-rerank --top-k 100
```

Generate the comparison report from existing run outputs:

```bash
PYTHONPATH=src uv run --no-sync python -m legal_rag.evaluation_reporting --allow-partial
```

Start the local Advanced RAG UI:

```bash
PYTHONPATH=src uv run --no-sync --group ui streamlit run src/legal_rag/ui/advanced_rag_app.py
```

# agentic-legal-rag-suite

Suite sperimentale per pipeline Legal RAG su normativa HTML: preprocessing deterministico, indexing Qdrant e baseline RAG con valutazione.

## Struttura repo
- `src/laws_ingestion/`: ingestion deterministic HTML -> dataset strutturato + pipeline notebook 03.
- `src/legal_indexing/`: chunk refinement, embeddings, indexing Qdrant e runtime RAG.
- `baselines/`: baseline retrieval/benchmarking (es. BM25).
- `notebooks/`: notebook sperimentali e walkthrough.
- `docs/`: documentazione operativa corrente.
- `docs/archive/`: documentazione storica/legacy (non source of truth).
- `data/evaluation/`: dataset benchmark (`questions.csv`, `questions_no_hint.csv`).

## Notebook principali
- `notebooks/03_laws_graph_pipeline.ipynb`
- `notebooks/04_qdrant_indexing_pipeline.ipynb`
- `notebooks/05_langgraph_rag_pipeline.ipynb`
- `notebooks/evaluation/02_evalutation_no_rag.ipynb`

## Setup rapido
```bash
poetry install
cp .env.example .env
```

## CLI / moduli Python
Il progetto e' notebook-centric e usa codice in `src/`.

Per eseguire i moduli direttamente:
```bash
PYTHONPATH=src poetry run python -m legal_indexing --help
PYTHONPATH=src poetry run python -m laws_ingestion --help
```

Esempio indexing:
```bash
UTOPIA_API_KEY=... PYTHONPATH=src poetry run python -m legal_indexing --subset-limit 500
```

## Config ambiente (Utopia)
Variabili principali:
- `UTOPIA_API_KEY`
- `UTOPIA_BASE_URL`
- `UTOPIA_EMBED_API_MODE` (`auto|openai|ollama`)
- `UTOPIA_EMBED_URL`
- `UTOPIA_EMBED_MODEL`
- `UTOPIA_CHAT_MODEL`
- `UTOPIA_JUDGE_MODEL`

## Documentazione
Indice documentazione corrente: `docs/index.md`.

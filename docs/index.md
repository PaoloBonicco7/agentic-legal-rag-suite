# Documentazione operativa

Questa cartella contiene la documentazione corrente del progetto. Il riferimento di overview generale e' in root.

## Overview
- `../OVERVIEW.md`: quadro completo di approccio, stato attuale e snapshot locale al 2 marzo 2026.

## Notebook pipeline ed evaluation
- `notebook_01_build_questions_no_hint.md`: preprocessing da MCQ a dataset no-hint.
- `notebook_02_evaluation_no_rag.md`: baseline senza retrieval (MCQ + no-hint + judge).
- `notebook_03_laws_graph_pipeline.md`: ingestion HTML -> dataset clean con grafo, eventi e QA gates.
- `notebook_04_qdrant_indexing_pipeline.md`: refinement chunk + embedding + indexing incrementale su Qdrant.
- `notebook_05_langgraph_rag_pipeline.md`: baseline Naive RAG con mini benchmark e artifact runtime.
- `notebook_06_advanced_rag_pipeline.md`: pipeline Advanced RAG con hybrid search (dense+sparse), failure taxonomy e dashboard comparativa Naive vs Advanced.

## Policy repository minima
- Versionare codice sorgente, test e documentazione.
- Non versionare artifact rigenerabili di run notebook (`notebooks/data/`, `notebooks/rag_pipeline/artifacts/`).
- `docs/archive/` e' storico e non va usato come source of truth operativo.

## Esecuzione moduli Python
Il codice e' sotto `src/`.

```bash
PYTHONPATH=src poetry run python -m legal_indexing --help
PYTHONPATH=src poetry run python -m laws_ingestion --help
```

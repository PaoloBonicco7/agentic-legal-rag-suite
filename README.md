# agentic-legal-rag-suite
Modular Agentic RAG for legal purposes: iterative retrieval loops (retrieve→self-evaluate→refine) over 3k HTML laws. Includes RAG-ready data extraction, hybrid search (lexical+semantic), MCQ benchmark evaluation, and full step-by-step tracing with citations.

## Dataset (HTML → RAG-ready JSONL)
The ingestion pipeline lives in `laws_ingestion/` and converts `data/leggi-html/*.html` into a flat JSONL dataset
(`laws/articles/passages/notes/edges/chunks` + `manifest.json`). See `laws_ingestion/README.md` for HTML structure
and extraction details.

## Quick setup (Poetry + Notebook Utopia)

1. Create environment and install dependencies:
   ```bash
   poetry install --no-root
   ```
2. Configure API key:
   ```bash
   cp .env.example .env
   # then set UTOPIA_API_KEY in .env
   ```
3. Start Jupyter:
   ```bash
   poetry run jupyter notebook
   ```
4. Open and run:
   `notebooks/utopia_api_smoke_test.ipynb`

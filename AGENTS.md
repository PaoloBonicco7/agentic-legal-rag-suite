# AGENTS.md

Agent-level context for working on this thesis codebase. The numbered workflow lives in `docs/specs/README.md`; each step has its own specification under `docs/specs/`. This file describes the project intent, the technology stack, and how a coding agent should behave when implementing or modifying any step.

## Project Goal

This repository is being refactored into a clean, reproducible Legal RAG research codebase for a thesis project. The thesis claim is that a more effective retrieval pipeline produces more accurate answers on a fixed set of legal questions. The codebase compares no-RAG, simple RAG, and advanced graph-aware RAG using a shared metric contract, so the comparison is reproducible and explainable.

The codebase must remain small and easy to read. A new reader should be able to clone the repository, inspect the real source data (legal HTML corpus and evaluation question sets), and follow the numbered specs to understand the experiment end to end.

## Repository Principles

- Prefer clarity over completeness when the two conflict.
- Do not reintroduce the broad complexity of the legacy `OLD/` project.
- Keep generated artifacts (indexes, run outputs, reports, caches) separate from source code and source data; they are reproducible and not committed by default.
- Treat source datasets (`data/laws_html/`, `data/evaluation/`) as first-class project assets — do not remove or replace them unless the project intent explicitly changes.
- Avoid hidden behavior: important inputs, outputs, assumptions, and limitations must be documented in the relevant spec or implementation note.
- Add complexity only when it directly improves the thesis workflow, reproducibility, or explanation quality.

## Coding Agent Workflow

- Before implementing any step, read the corresponding spec under `docs/specs/` end to end. The spec is the source of intent.
- Use `src/legal_rag/oracle_context_evaluation/` as the reference pattern for new step modules: `models.py` for Pydantic contracts and config, `prompts.py` for prompt strings + version constant, `llm.py` for the structured chat client, `io.py` for JSONL/JSON I/O, `runner.py` for orchestration, `cli.py` + `__main__.py` for the entry point, `env.py` for `.env` loading.
- Reuse existing utilities before writing new ones — in particular `UtopiaStructuredChatClient`, `load_env_file`, and the `_Record` / `to_json_record` pattern from `oracle_context_evaluation`.
- Put core logic in `src/legal_rag/<step>/`. Notebooks under `notebooks/` are demonstration and explanation only — they should not contain reusable logic.
- Use Pydantic v2 for every shared data contract: configuration models, output records, manifest schemas. Use `extra="forbid"` and JSON-mode dump for record exports.
- Each step exposes constant `*_SCHEMA_VERSION` and `*_PROMPT_VERSION` strings, and the run manifest must record both.
- Treat `OLD/` as historical reference only. Inspect it to understand intent, but do not copy its architecture.
- Add or update focused tests under `tests/` whenever a Contract or Quality Gate from a spec changes.

## Technology Stack

- **LLM provider**: Utopia (HPC4AI, Università di Torino), Ollama-compatible endpoint at `https://utopia.hpc4ai.unito.it/api`. Structured output via JSON schema (`format=schema`, `temperature=0`). Default model `SLURM.gpt-oss:120b`. Configurable per step via `chat_model` and `judge_model`. Credentials in `.env` (`UTOPIA_API_KEY`, `UTOPIA_BASE_URL`).
- **Vector store**: Qdrant in **local persistent file mode**, default at `data/indexes/qdrant`. No Docker, no server. Use the official `qdrant-client` Python SDK.
- **Embedding**: pluggable backend selected by config.
  - Default backend `local`: `BAAI/bge-m3` via `sentence-transformers` or `FlagEmbedding` (open weights, free, multilingual including Italian, native dense + sparse for hybrid).
  - Alternative backend `utopia`: HTTP call to `/ollama/api/embeddings` on the same Utopia endpoint, model identity from config.
  - Config fields: `embedding_backend: Literal["local","utopia"]`, `embedding_model: str`, `embedding_dim: int` (optional override).
- **Reranker**: LLM-as-reranker using the same Utopia client with a dedicated prompt that scores chunk relevance on a `0-2` scale. No separate cross-encoder dependency.
- **Hybrid retrieval**: dense + sparse vectors stored in the same Qdrant collection (named/sparse vectors). Fusion via Reciprocal Rank Fusion (RRF) using Qdrant's native `Query API` with `prefetch`.
- **Configuration**: per-step Pydantic `BaseModel` (e.g., `OracleEvaluationConfig`) with sane defaults. Credentials and per-machine paths come from `.env` via `load_env_file`. No YAML config files.
- **Core libraries** (already in `pyproject.toml`): `lxml`, `pydantic>=2.10`, `requests`. New additions to declare when needed: `qdrant-client`, plus the chosen embedding library (`sentence-transformers` or `FlagEmbedding`).

## When to Use Skills and MCP

- Use the available **Qdrant skills** when designing or modifying collection schemas, payload metadata, payload indexes, filters, named/sparse vectors, hybrid retrieval, or `Query API` usage. Do not use them for unrelated tasks.
- Use the **LangChain documentation MCP** only if a specific step explicitly introduces a LangChain abstraction (loader, splitter, retriever, chain). The default in this project is to call libraries directly; LangChain is welcome only where it removes glue code without obscuring the pipeline.
- Prefer well-known libraries over custom code for parsing, indexing, retrieval, evaluation, and visualization when the dependency is easy to justify in the thesis context.

## Documentation and Language

- **Specs and code in English.** Specs under `docs/specs/<NN>_<name>.md` follow the shared structure (Purpose, Inputs, Outputs, Pipeline, Contract, Quality Gates, Notebook Role, Acceptance Criteria); see `docs/specs/README.md`.
- **Implementation notes** under `docs/notes/<NN>_*.md` and **results** under `docs/results/<NN>_*.md` may be in Italian when they support the thesis narrative.
- **Code comments**: English, brief, only where intent is non-obvious or where a constraint, invariant, or workaround would otherwise be missed. Default to no comment.
- **Docstrings**: minimal, one or two lines, focused on the fundamental purpose. No multi-paragraph docstrings.
- Keep repository-level documentation minimal. Detailed step behavior lives in the step-specific spec, note, and result files.

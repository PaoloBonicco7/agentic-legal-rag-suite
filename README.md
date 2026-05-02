# Agentic Legal RAG Suite

A reproducible, thesis-oriented Legal RAG project that turns a legal corpus and evaluation questions into a clear end-to-end question answering workflow.

## What This Project Builds

This repository is being refactored into a minimal research codebase for building, running, and explaining a Legal RAG pipeline.

The goal is not to preserve the complexity of the previous implementation. The goal is to keep the same research flow while making each step understandable, reproducible, and easy to discuss with a research team.

The evaluation question datasets are part of the application and are intended to be versioned in the repository.
The HTML legal corpus is expected locally under `data/laws_html/`, but is not tracked in Git.

## Refactored Pipeline

1. Prepare evaluation questions.
2. Evaluate a no-retrieval baseline.
3. Prepare the legal corpus.
4. Build the retrieval index.
5. Run simple RAG.
6. Run advanced RAG.
7. Compare results.

## Repository Shape

- Core reusable code contains the implementation for each pipeline step.
- Pydantic v2 defines data contracts, configuration models, validation, and structured outputs.
- External libraries are welcome when they make the code simpler, clearer, and easier to reproduce.
- Notebooks provide short demonstration runs with explanatory text and visible outputs.
- Markdown specifications are the source of intent for each step.
- Markdown implementation notes record choices, results, and lessons learned.
- `data/laws_html/` contains the local source legal corpus and is ignored by Git.
- `data/evaluation/` contains the evaluation question datasets.
- `OLD/` is historical reference only; it is not the target architecture.

## Reproducibility Contract

The repository should include the full set of evaluation questions used by the application.
The legal HTML corpus should be kept locally under `data/laws_html/` and documented as an external input.

Runs should be reproducible from a fresh clone once the local corpus has been placed under `data/laws_html/`. Any generated dataset, retrieval index, benchmark output, or cache should be documented as a derived artifact, not as source data.

Outputs used in the thesis or shared with the research team should be traceable to their input data, configuration, and pipeline step.

## Development Setup

This project uses `uv` for dependency management.

Install the base development and notebook environment:

```bash
uv sync --group dev --group notebooks
```

Install the optional RAG dependencies when working on retrieval or generation:

```bash
uv sync --group dev --group notebooks --group rag
```

Run the basic checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Documentation Model

Each major pipeline step should have two documents:

- a specification file describing purpose, inputs, outputs, contracts, and acceptance criteria;
- an implementation/results note describing what was built, what choices were made, and what results were observed.

Repository-level documentation should remain minimal. Details belong in the step-specific Markdown files.

## Current Status

The refactor is in progress.

The previous implementation has been archived under `OLD/` and should be used only to understand the historical workflow, expected outputs, and prior experiments.

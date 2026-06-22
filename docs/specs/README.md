# Specification Map

This folder defines the thesis PoC workflow from source data to final evaluation.

The specifications are the source of intent. They describe what each step must prove, what it consumes, what it produces, and how the result is checked. They should stay compact enough to guide implementation and notebook demonstrations without becoming a second codebase.

`AGENTS.md` at the repository root is complementary, not a duplicate: it describes the project's technology stack, repository principles, and how a coding agent should behave. This file is the workflow map; each numbered file is the contract for one step.

## Workflow

1. `01_laws_preprocessing.md`: HTML laws to clean legal dataset with graph metadata.
2. `02_evaluation_dataset.md`: raw evaluation CSV files to validated MCQ and no-hint datasets.
3. `02b_oracle_context_evaluation.md`: controlled evaluation with and without source-of-truth article context.
4. `03_indexing_contract.md`: clean legal chunks to a retrieval-ready index contract.
5. `04_no_rag_baseline.md`: model-only evaluation without retrieval.
6. `05_simple_rag.md`: minimal retrieval, context, answer, citation, and evaluation loop.
7. `06_advanced_graph_rag.md`: explainable retrieval improvements (hybrid, query rewriting, graph expansion, reranking) compared by ablation; the promoted pipeline is hybrid + multi-query.
8. `06b_retrieval_diagnostics.md`: retrieval-only diagnostics for recall, ranking, and promotion decisions before advanced RAG runs.
9. `07_evaluation_reporting.md`: shared metrics, comparisons, failure analysis, and thesis-ready reporting.

## Shared Principles

- This is a reproducible thesis PoC, not a production system.
- Source data is versioned: `data/laws_html/` and `data/evaluation/`.
- Generated datasets, indexes, benchmark outputs, reports, caches, and notebook artifacts are reproducible outputs and are not committed by default.
- `OLD/` is historical reference only.
- Core logic belongs in reusable modules; notebooks demonstrate runs and explain choices.
- Use Pydantic v2 for shared data contracts, configuration models, validation, and structured outputs across the pipeline.
- External libraries are allowed and recommended when they make the PoC simpler, clearer, or more reliable.
- Prefer well-known libraries over custom implementations for specialized tasks when the dependency is easy to justify.
- Prefer clarity over completeness when a feature would make the PoC harder to explain.

## Common Spec Structure

Each numbered spec uses the same sections:

- Purpose
- Inputs
- Outputs
- Pipeline
- Contract
- Quality Gates
- Notebook Role
- Acceptance Criteria

## Notebook Mapping

- `notebooks/01_laws_preprocessing.ipynb`
- `notebooks/01b_laws_graph_exploration.ipynb` (graph structure analysis; no dedicated spec)
- `notebooks/02_evaluation_dataset.ipynb`
- `notebooks/02b_oracle_context_evaluation.ipynb`
- `notebooks/03_indexing_contract.ipynb`
- `notebooks/04_no_rag_baseline.ipynb`
- `notebooks/05_simple_rag.ipynb`
- `notebooks/06_advanced_graph_rag.ipynb`
- `notebooks/06b_retrieval_diagnostics.ipynb`

Each notebook should run one coherent demonstration, display the relevant artifacts, and explain the transformation. It should not contain the main implementation logic.

Step 07 (`07_evaluation_reporting.md`) is specified but not yet built as a notebook or report artifact. The cross-method headline comparison currently lives in `docs/results/00_overview.md`, derived by hand from the run summaries.

## Results Notes

After implementation, each step has a short result note under `docs/results/` with the same numbering, recording run configuration, observed counts, metrics, and known limitations. `docs/results/00_overview.md` holds the cross-method headline comparison. Result notes are in Italian, keeping technical terms in English.

# Specification Map

This folder defines the thesis PoC workflow from source data to final evaluation.

The specifications are the source of intent. They describe what each step must prove, what it consumes, what it produces, and how the result is checked. They should stay compact enough to guide implementation and notebook demonstrations without becoming a second codebase.

## Workflow

1. `01_laws_preprocessing.md`: HTML laws to clean legal dataset with graph metadata.
2. `02_evaluation_dataset.md`: raw evaluation CSV files to validated MCQ and no-hint datasets.
3. `02b_oracle_context_evaluation.md`: controlled evaluation with and without source-of-truth article context.
4. `03_indexing_contract.md`: clean legal chunks to a retrieval-ready index contract.
5. `04_no_rag_baseline.md`: model-only evaluation without retrieval.
6. `05_simple_rag.md`: minimal retrieval, context, answer, citation, and evaluation loop.
7. `06_advanced_graph_rag.md`: explainable improvements with filters, graph expansion, reranking, and optional hybrid retrieval.
8. `07_evaluation_reporting.md`: shared metrics, comparisons, failure analysis, and thesis-ready reporting.

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
- `notebooks/02_evaluation_dataset.ipynb`
- `notebooks/02b_oracle_context_evaluation.ipynb`
- `notebooks/03_indexing_contract.ipynb`
- `notebooks/04_no_rag_baseline.ipynb`
- `notebooks/05_simple_rag.ipynb`
- `notebooks/06_advanced_graph_rag.ipynb`
- `notebooks/07_evaluation_reporting.ipynb`

Each notebook should run one coherent demonstration, display the relevant artifacts, and explain the transformation. It should not contain the main implementation logic.

## Results Notes

After implementation, each step should have a short result note under `docs/results/` with the same numbering. The note should record the implementation choices, run configuration, observed counts, metrics, and known limitations.

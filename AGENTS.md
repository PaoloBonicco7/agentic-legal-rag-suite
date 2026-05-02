# AGENTS.md

## Project Goal

This repository is being refactored into a clean, reproducible Legal RAG research codebase for a thesis project.

The final project should preserve the end-to-end research flow from data preparation to advanced RAG, while making the codebase much smaller, easier to read, and easier to explain to a research team.

The repository should include the full source data needed by the application: the legal HTML corpus and the evaluation question datasets. A new reader should be able to clone the project and inspect the real input data without depending on the legacy implementation.

## Repository Principles

- Prefer clarity over completeness when the two conflict.
- Keep the core implementation small, reusable, and testable.
- Keep notebooks short, explicit, and focused on demonstrating each step.
- Do not reintroduce the broad complexity of the legacy project.
- Avoid hidden behavior: important assumptions, inputs, outputs, and limitations must be documented.
- Keep generated artifacts and full-run outputs separate from source code and step specifications.
- Treat source datasets as first-class project assets, not as optional examples.

## Expected Workflow

- Treat `OLD/` as historical reference only.
- Inspect `OLD/` to understand the previous flow, expected outputs, and experimental intent.
- Do not copy the old architecture, folder structure, or complexity by default.
- Put reusable logic in core Python modules.
- Use Pydantic v2 for explicit data contracts, configuration models, validation, and structured run outputs.
- Use external libraries when they make the implementation simpler, clearer, or more reliable for the PoC.
- Prefer well-known libraries over custom code for parsing, validation, indexing, retrieval, evaluation, and visualization when they reduce complexity.
- Use notebooks as demonstration and explanation layers, not as the place where most logic lives.
- Each notebook should show one coherent run with minimal code, visible inputs, visible outputs, and short explanations of the choices made.

## Data Policy

- Version the full source legal corpus under `data/laws_html/`.
- Version the evaluation question datasets under `data/evaluation/`.
- Do not remove or replace source data unless the project intent explicitly changes.
- Do not commit generated indexes, intermediate datasets, benchmark artifacts, or cache files unless a specification says they are canonical source assets.

## Documentation Rules

- Every major pipeline step must have a focused Markdown specification.
- Every implemented step should also have a Markdown note describing implementation choices and observed results.
- Specifications describe intent, contracts, inputs, outputs, and acceptance criteria.
- Implementation notes describe what was built, why key choices were made, and what results were obtained.
- Keep repository-level documentation minimal; put detailed step behavior in step-specific documents.
- Always document new code with minimal docstrings that clearly explain the fundamental purpose and processing steps.
- Keep code comments in English, concise, and focused on non-obvious decisions or data transformations.

## Quality Bar

- Changes should be small enough to review and explain.
- Prefer deterministic, reproducible behavior where practical.
- Add external dependencies deliberately: they should simplify the project and be easy to explain in the thesis context.
- Add or update focused tests when behavior changes.
- Validate that demo outputs are reproducible before presenting them as project results.
- Validate that source data remains available from a fresh clone.
- Make failures understandable: error messages and checks should help identify the broken step.
- Before adding complexity, confirm that it directly improves the thesis workflow, reproducibility, or explanation quality.

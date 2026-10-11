# Numbered research workflow

Each specification defines Purpose, Inputs, Outputs, Pipeline, Contract, Quality Gates, Notebook Role and Acceptance Criteria. Read it before changing the corresponding module. Shared data contracts live in `src/legal_rag/contracts.py`; there is one owner during parallel development.

| Step | Specification | Core responsibility |
|---|---|---|
| 00 | [Preservation](00_preservation.md) | Preserve the original research and untracked sources. |
| 01 | [Acquisition](01_acquisition.md) | Freeze complete official sources with provenance. |
| 02 | [Preprocessing](02_preprocessing.md) | Deterministic parsing and source coverage. |
| 03 | [Metadata and graph](03_metadata_graph.md) | Evidence, scope, conservative status and explicit edges. |
| 04 | [Indexing](04_indexing.md) | Contextual chunks and immutable dense/sparse index. |
| 05 | [Shared engine](05_rag.md) | Retrieval, context, providers, answers and stateless UI. |
| 06 | [Evaluation](06_evaluation.md) | Baselines, judgments and paired comparisons. |
| 07 | [Reporting](07_reporting.md) | Offline overview, experiment, question and corpus details. |
| 08 | [Retrieval diagnostics](08_retrieval_diagnostics.md) | Error localization and isolated ablations. |
| 09 | [Benchmark reviews](09_benchmark_reviews.md) | Human-approved revisions and frozen-answer rejudging. |
| 10 | [Reproducibility](10_reproducibility.md) | Verified packaging, restoration and delivery. |
| 11 | [Retrieval optimization — planned](11_retrieval_optimization.md) | Evidence-driven changes, isolated tests and a conditional combined candidate. |

Progress only after the preceding input gates pass. Technical tests do not establish a QA improvement. Negative or incomplete experiments remain visible. Runtime retrieval and generation cannot read benchmark gold. Notes explain decisions; results distinguish measured findings from future work.

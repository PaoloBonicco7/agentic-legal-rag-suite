# Agentic Legal RAG — Project Plan (High-level)
_Base architecture + documented task breakdown with tools/frameworks_

## 0) Project goals (recap)
- Build a **legal Agentic RAG** system over ~3000 HTML laws.
- Start with **MCQ benchmark** (A..F) + **citations + rationale**.
- Minimize **false-positive citations** (hallucinated / irrelevant norms).
- Ensure **full traceability** (audit/diagnostics) and **incremental extensibility**.

## 0.1) Benchmark dataset (source of truth)
- File: `questions.csv` (163 rows; 100 complete questions + 63 empty rows to ignore).
- Options: always 6 options `A)`..`F)` inside the `Domanda` cell (newline-separated).
- Ground truth:
  - `Risposta corretta` (label A..F)
  - `Riferimento legge per la risposta` (1..N references; 5 questions have multi-line refs).
- Parsing requirements: robust CSV parsing (embedded newlines) + UTF-8.
- Note: `LegalRAGatoni/` is baseline-only for later comparison; do not reuse it as implementation context.

## 1) Guiding principles
1. **Deterministic ingestion** (no LLM for parsing/metadata), LLM allowed only for **embeddings** (and later answer/eval).
2. **Baseline first**: build a measurable non-agentic baseline before adding loops.
3. **Hard filters are code**: abrogation/versioning/dedup/token budget handled deterministically.
4. **Everything traceable**: every run produces a `trace_id` and step-by-step logs/artifacts.
5. **Toggleable modules**: each step can be enabled/disabled via config.

---

# 2) Tech stack (planned)
## Core language & environment
- **Python 3.11+**
- **Poetry** (dependency management, venv)
- **Jupyter** notebooks for experiments & reporting (code lives in package)

## Parsing & data processing (no LLM)
- **BeautifulSoup4 / lxml**: HTML parsing
- **regex**: extracting references (abrogata/modificata, “art.” patterns)
- **pydantic**: strict schemas & validation
- Storage format: **JSONL** or **Parquet** (recommended for scale: Parquet)

## Retrieval & indexing
- **BM25**:
  - Option A: `rank-bm25` (simple)
  - Option B: Elasticsearch/OpenSearch (overkill for local PoC; optional)
- **Vector index**:
  - Option A: **FAISS** (local, fast)
  - Option B: **Chroma** (simple persistence + metadata)
- Embeddings provider: external service you call (configurable)

## Graph relations (legal links/versioning)
- **NetworkX** (local graph, easy)
- Optional later: Neo4j (only if graph querying becomes complex)

## Orchestration / agent workflow
- **LangGraph** (state machine + loops + node-level modularity)
- Alternative (fallback): custom orchestrator functions (same interfaces)

## Evaluation & observability
- Metrics: `accuracy` (overall + per-level), `retrieval_recall@k` (gold targets), `gold_citation_hit`, `citation_presence`, `faithfulness proxy`
- Logging: **structured JSON logs**
- Optional: **MLflow** / **Weights & Biases** for experiment tracking (nice-to-have)

---

# 3) Artifacts & directories (recommended)
```
agentic_legal_rag/
  requirements/
    REQUIREMENTS.md                # already defined (project requirements)
    PROJECT_PLAN.md                # this file
    ARCHITECTURE.md                # components + data flow + state machine
  data/
    raw_html/                      # input corpus (not in git)
    processed/                     # parsed nodes (jsonl/parquet)
    graphs/                        # edges/relations
    indexes/                       # bm25 + vector store persistence
    benchmarks/                    # MCQ dataset (questions + ground truth)
    runs/                          # traces/logs/results per run
  notebooks/
    01_ingestion_validation.ipynb
    02_index_building.ipynb
    03_baseline_benchmark.ipynb
    04_langgraph_pipeline.ipynb
  src/agentic_legal_rag/
    domain/                        # pydantic models
    ingestion/                     # html parsing + normalization
    storage/                       # load/save datasets
    indexing/                      # build bm25/vector/graph
    retrieval/                     # hybrid retrieval + filters + graph expand
    pipeline/                      # langgraph state + nodes
    evaluation/                    # benchmark runner + metrics
    utils/                         # logging, text utils
```

---

# 4) Work breakdown structure (WBS)
Each task lists: **Goal → Output → Tools → Done criteria**.

## Phase A — Foundation & documentation (architecture-first)
### A1) Define system boundaries & interfaces
- **Goal**: lock down component responsibilities and I/O contracts.
- **Output**:
  - `ARCHITECTURE.md` (components diagram + data flow)
  - public interfaces for: ingestion, indexing, retrieval, pipeline, evaluation
- **Tools**: Markdown, pydantic schemas
- **Done**: every module has a clear “input/output” contract.

### A2) Define data schemas (domain model)
- **Goal**: stable, validated schemas for legal nodes and outputs.
- **Output**: `domain/models.py` (pydantic)
  - `Law`, `Article`, `Note` (optional), `Edge`
  - `Citation`, `EvidenceItem`
  - `MCQResponse`, `EvalReport`
  - `AgentState` (for LangGraph)
- **Tools**: pydantic
- **Done**: sample objects validate; schema version field included.

### A3) Define experiment protocol & metrics
- **Goal**: decide what “better” means before coding loops.
- **Output**:
  - `evaluation/metrics.py` spec
  - `benchmarks/README.md` describing MCQ dataset format
- **Tools**: Python + markdown
- **Done**: metrics list fixed; output report format fixed.

---

## Phase B — Data ingestion (HTML → RAG-ready dataset) (no LLM)
### B0) Benchmark-driven subset (fast iteration)
- **Goal**: iterate ingestion on the 22 laws referenced by `questions.csv` before scaling to the full corpus.
- **Output**: a reproducible list of `law_id` / source files for the benchmark subset.
- **Tools**: Python (CSV + filename registry)
- **Done**: all gold targets (law+article) are resolvable in the subset.

### B1) HTML parser (Law + Articles + anchors)
- **Goal**: deterministic extraction of title/date/number + article segmentation.
- **Output**:
  - `processed/articles.parquet` (or jsonl)
  - `processed/laws.parquet`
- **Tools**: BeautifulSoup4, lxml, regex
- **Done**: on sample set, article count and anchors are stable; parsing errors logged.

### B2) Reference & status extraction (abrogation/amendments/links)
- **Goal**: extract legal relations and status signals.
- **Output**:
  - `graphs/edges.parquet` (ABROGATED_BY, AMENDED_BY, REFERS_TO, etc.)
  - metadata fields on laws/articles (status, amended_by, abrogated_by)
- **Tools**: regex + HTML href parsing, NetworkX (optional at this stage)
- **Done**: known patterns like “Abrogata da…” are detected; edges created.

### B3) Normalization & canonical IDs
- **Goal**: stable IDs across runs; consistent text cleanup for indexing.
- **Output**:
  - `law_id`, `article_id` canonical formats documented
  - normalized `clean_text` field (remove boilerplate, nav)
- **Tools**: custom text utils + regex
- **Done**: re-running ingestion produces identical IDs and near-identical text.

### B4) Data QA notebook
- **Goal**: validate ingestion quality visually and with checks.
- **Output**: `01_ingestion_validation.ipynb`
- **Tools**: pandas, rich display
- **Done**: notebook shows distributions (articles per law, missing fields), sample previews.

---

## Phase C — Indexing & retrieval baseline (non-agentic)
### C1) Build BM25 index
- **Goal**: lexical retrieval for exact references/numbers.
- **Output**: persisted BM25 index in `data/indexes/bm25/`
- **Tools**: rank-bm25 (or Whoosh), custom persistence
- **Done**: given a query, returns top-k articles with scores.

### C2) Build vector index (embeddings)
- **Goal**: semantic retrieval for paraphrases.
- **Output**: vector store in `data/indexes/vector/`
- **Tools**: FAISS or Chroma + embedding client wrapper
- **Done**: can retrieve top-k by cosine distance; metadata filters work.

### C3) Hybrid retrieval (merge + dedup)
- **Goal**: combine BM25 + vector results into unified candidates.
- **Output**: `retrieval/hybrid.py`
- **Tools**: Python
- **Done**: deterministic merge policy documented (weights, tie-breakers).

### C4) Hard filters (anti-hallucination)
- **Goal**: reduce false-positive citations before any LLM reasoning.
- **Output**: `retrieval/filters.py`
  - deprioritize (not exclude) abrogated laws by default on the benchmark (to avoid dropping gold)
  - prefer amended/updated versions if available
  - dedup same law/article, token budget selection
- **Tools**: Python + graph metadata
- **Done**: filters are unit-tested and produce explainable reasons.

---

## Phase D — Baseline answering for MCQ (measurable)
### D1) Evidence builder
- **Goal**: compact, citable evidence set (3–8 snippets).
- **Output**: `retrieval/evidence.py`
- **Tools**: text windowing, snippet extraction
- **Done**: every evidence item has `Citation` + snippet boundaries.

### D2) MCQ answerer (structured output)
- **Goal**: choose one of the provided option labels (benchmark: A..F) + rationale + citations.
- **Output**: `pipeline/nodes/answer_mcq.py` (even before LangGraph)
- **Tools**: LLM client wrapper, pydantic output validation
- **Done**: invalid JSON is handled; answer constrained to the provided labels (A..F); citations must come from evidence_set.

### D3) Benchmark runner
- **Goal**: run 100-Q dataset end-to-end and save reports.
- **Output**:
  - `runs/<timestamp>/results.jsonl`
  - `runs/<timestamp>/summary.json`
- **Tools**: evaluation/runner.py
- **Done**: reports include overall + per-level accuracy, retrieval_recall@k, gold_citation_hit, and per-question trace_id + retrieved citations.

---

## Phase E — LangGraph orchestration (agentic, incremental)
### E1) Define AgentState and graph skeleton
- **Goal**: create the workflow as a state machine.
- **Output**: `pipeline/langgraph_app.py`
- **Tools**: LangGraph
- **Done**: graph runs baseline path without loops.

### E2) Self-evaluation node (diagnostic, structured)
- **Goal**: evaluate context sufficiency + faithfulness; return actions.
- **Output**: `pipeline/nodes/self_eval.py`
- **Tools**: LLM + pydantic output
- **Done**: evaluator outputs `suggested_actions` (not just scores).

### E3) Correction loop (1st iteration)
- **Goal**: when eval fails threshold → apply corrective action → re-retrieve.
- **Output**: `pipeline/nodes/correct.py` + loop edges in LangGraph
- **Tools**: LangGraph
- **Done**: loop capped (e.g., max 2–3); improvements measured on benchmark.

### E4) Graph expansion node (optional)
- **Goal**: follow REFERS_TO / AMENDED_BY / ABROGATED_BY when needed.
- **Output**: `retrieval/graph_expand.py` + LangGraph node
- **Tools**: NetworkX
- **Done**: only triggers when eval suggests missing context; audit logs show followed edges.

---

## Phase F — Reporting & thesis-ready outputs
### F1) Experiment report notebook
- **Goal**: produce plots/tables for thesis (accuracy, error types, abrogation effects).
- **Output**: `notebooks/05_results_report.ipynb`
- **Tools**: pandas, matplotlib
- **Done**: reproducible report generation from `data/runs/`.

### F2) Documentation package
- **Goal**: clear documentation for repo and thesis.
- **Output**:
  - `README.md` (how to run)
  - `ARCHITECTURE.md` (diagram + explanation)
  - `DATA_MODEL.md` (schemas)
  - `EVAL_PROTOCOL.md` (benchmark & metrics)
- **Done**: a new contributor can run baseline in <30 min (local).

---

# 5) Definition of Done (project-level)
- Ingestion produces validated datasets + relations graph.
- Hybrid retrieval works with hard filters and is explainable.
- MCQ baseline runs on benchmark and logs per-question traces.
- LangGraph pipeline supports at least 1 loop (eval → correct → retry).
- Results are reproducible via notebooks + saved artifacts.

---

# 6) Immediate next actions (architecture-only, no heavy coding)
1. Create `ARCHITECTURE.md` with:
   - component diagram
   - data flow
   - state machine diagram (baseline path + loop)
2. Create `domain/models.py` (pydantic) drafts:
   - `Law`, `Article`, `Edge`, `Citation`, `MCQResponse`, `AgentState`
3. Define benchmark file format in `data/benchmarks/README.md`.

(Once those are frozen, implementation becomes mechanical.)

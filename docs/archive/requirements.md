# Requisiti
Status: Roadmap (target)  
Scope: goals, scope, success criteria  
Source of truth: questo documento

Questo documento e' la **single source of truth** per obiettivi, scope, requisiti e criteri di successo. Gli altri documenti devono rimandare qui quando parlano di requisiti.

## Obiettivo del progetto
Sviluppare un sistema **Agentic RAG** in ambito legale capace di:
1. Rispondere a domande a scelta multipla (MCQ; benchmark con opzioni A..F).
2. Fornire spiegazioni motivate con **citazioni normative**.
3. Ridurre **false-positive citations** (citazioni inventate/irrilevanti).
4. Supportare evoluzione incrementale della pipeline retrieval.
5. Garantire tracciabilita' completa (artefatti, diagnostica, riproducibilita').

## Input e output
### Input
- Domanda MCQ: testo + 6 opzioni (A..F), da `evaluation/questions.csv`.
- Corpus: ~3000 leggi regionali in HTML (es. `data/laws_html/*.html`).
- Notebook di pre-processing attuale: `notebooks/pre_processing_data/03_laws_graph_pipeline.ipynb` (notebook storici in `notebooks/old/`).

### Output (modalita' benchmark)
Output strutturato (JSON/Pydantic), minimo:
- `answer_label`: una tra A..F
- `rationale`: testo di spiegazione
- `citations`: lista di citazioni (vedi policy sotto)
- `trace_id`: identificatore esecuzione
- `eval`: campi diagnostici utili (facoltativi ma raccomandati)

## Requisiti funzionali (target)
### RF1 - Ingestion deterministica
- Parsing HTML e estrazione metadati senza LLM.
- Segmentazione coerente con struttura legale (articolo, passages, note).
- Estrazione relazioni tra leggi (almeno: `REFERS_TO`, `AMENDED_BY`, `ABROGATED_BY`) best-effort.

### RF2 - Dataset RAG-ready
- Generare un dataset indicizzabile (baseline JSONL flat).
- ID canonici stabili (es. `law_id` derivato dal filename, vedi `docs/data_model.md`).
- Metadati minimi disponibili per filtri e ranking (status, articolo, struttura, edges).

### RF3 - Retrieval (Qdrant-first)
- Vector store Qdrant come backend principale (persistenza locale).
- Supporto a varianti:
  - `plain`: testo senza intestazioni contestuali.
  - `with_metadata`: testo con prefissi/metadata utili al retrieval e alla citazione.

### RF4 - Ottimizzazione retrieval (incrementale)
Roadmap minima:
1. Baseline naive retrieval top-k (vector).
2. Hybrid retrieval (BM25 + vector) con merge deterministico.
3. Hard filtering (dedup, token budgeting, policy abrogazioni).
4. Reranking (cross-encoder o LLM reranker) su top-N.
5. Query rewrite (controllato) e loop agentico (self-eval -> azione -> re-retrieve).

### RF5 - Answering e citazioni
- Il modello deve selezionare una label A..F.
- Deve citare **solo evidenza recuperata** (no fonti esterne inventate).
- Deve poter produrre una spiegazione sintetica e verificabile.

### RF6 - Tracciabilita' e artefatti
Ogni run produce:
- configurazione usata (k, modelli, varianti)
- documenti recuperati + punteggi
- contesto costruito (o riferimenti a esso)
- output LLM grezzo e parseato
- metriche e report

## Policy “citazioni corrette” (decisione)
### Citazioni nel runtime
- Una “citation” e' **un `chunk_id`** presente nel contesto consegnato al modello.
- Il prompt deve vincolare `citations` a chunk_id presenti.

### Valutazione citazioni rispetto al gold
Dato che il benchmark fornisce riferimenti normativi (non chunk_id), usiamo il mapping via metadati del chunk:
- **Gold citation hit**: la risposta cita almeno un `chunk_id` che appartiene a una legge/articolo coerente col riferimento gold (match su `(law_id, article_label_norm)` quando possibile).
- Nota: se il gold non specifica articolo/comma, il match puo' degradare a `law_id` only (da esplicitare nel report).

## Policy abrogazioni (decisione)
- Nel benchmark **non escludere** documenti abrogati a priori: al massimo deprioritizzare, per evitare di eliminare gold.
- Fuori benchmark (modalita' “prod”), la policy puo' diventare piu' aggressiva (preferire vigente/ultima versione).

## Non-requisiti / fuori scope
- Knowledge graph come fonte primaria di retrieval (GraphRAG) e' fuori scope come baseline.
- Parsing o estrazione metadati guidata da LLM e' fuori scope per ingestion.

## Criteri di successo (metriche)
### Retrieval
- `recall@k` e `hit-rate@k` sul gold (definito in `docs/evaluation.md`).
### MCQ
- `accuracy` complessiva e per livello di difficolta'.
### Citazioni
- tasso di `gold_citation_hit` e riduzione di citazioni fuori evidenza (idealmente 0).

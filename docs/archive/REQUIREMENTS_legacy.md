# Agentic RAG for Legal Retrieval  
## Documento dei Requisiti e Specifiche Architetturali

---

# 1. Obiettivo del Progetto

Sviluppare un sistema **Agentic RAG (Retrieval-Augmented Generation)** in ambito legale capace di:

1. Rispondere a domande a scelta multipla (MCQ; nel benchmark attuale opzioni A..F).
2. Fornire spiegazione motivata con citazioni normative.
3. Ridurre allucinazioni (citazioni non pertinenti).
4. Supportare evoluzione incrementale della pipeline.
5. Consentire tracciabilità completa e analisi delle performance.

Il sistema opererà su un corpus di **~3000 leggi regionali in formato HTML**, strutturate e interconnesse tramite riferimenti ipertestuali.

---

# 2. Scope del Sistema

## 2.1 Input

- Domanda:
  - MCQ (nel benchmark attuale sempre 6 opzioni A..F)
  - Benchmark source of truth: `questions.csv` (ID 1..100); opzioni nel campo `Domanda` come righe `A) ...` … `F) ...`
  - In futuro: domanda aperta
- Corpus:
  - File `.html` contenenti testi normativi
  - Presenza di:
    - Articoli
    - Ancore
    - Note
    - Indicazioni di abrogazione
    - Collegamenti ad altre leggi

## 2.2 Output

### Modalità benchmark (MCQ)

Output strutturato (Pydantic/JSON):

- `answer_label`: string (deve essere una delle opzioni presenti, es. "A".."F")
- `confidence`: float
- `rationale`: spiegazione testuale
- `citations`: elenco fonti
- `eval`: report diagnostico
- `trace_id`: id tracciabilità

### Modalità aperta (fase successiva)

- Risposta testuale strutturata
- Citazioni normative puntuali
- Argomentazione supportata da evidenza

---

# 3. Dataset e Struttura del Corpus

## 3.1 Stato attuale

- ~3000 file HTML
- Presenza di:
  - Titolo legge
  - Data
  - Numero
  - Articoli (Art. 1, Art. 2, ecc.)
  - Ancore HTML
  - Note di modifica
  - Riferimenti a leggi successive
  - Indicazioni di abrogazione

## 3.2 Requisito critico

Trasformare gli HTML in un dataset **RAG-ready**, strutturato come:

### Entità principali

- `Law`
- `Article`
- (opzionale futuro) `Paragraph/Comma`
- `Note`

### Metadati minimi richiesti

Per ogni Law:
- id univoco
- titolo
- data
- numero
- stato (vigente / abrogata)
- abrogated_by
- amended_by
- riferimenti in uscita

Per ogni Article:
- article_number
- anchor_id
- testo
- law_id parent

---

# 4. Architettura Generale

Il sistema sarà organizzato in **tre livelli principali**:

---

## 4.1 Livello 1 – Ingestion & Normalization (Offline)

Funzioni:

- Parsing HTML (consentito usare librerie dedicate per semplificare e aumentare robustezza: es. BeautifulSoup4 + lxml; fallback stdlib possibile)
- Estrazione metadati
- Costruzione gerarchia legge → articolo
- Estrazione relazioni:
  - ABROGATED_BY
  - AMENDED_BY
  - REFERS_TO
- Creazione dataset strutturato

Output:
- Database documentale (JSON o DB locale)
- Grafo delle relazioni tra leggi

---

## 4.2 Livello 2 – Indexing (Offline)

Costruzione di:

1. Indice BM25 (keyword/lexical)
2. Indice vettoriale (embeddings)
3. Struttura grafo (NetworkX o simile)

Obiettivo:
Supportare retrieval ibrido:
- Lexical
- Semantic
- Metadata-aware
- Graph-based (fase evolutiva)

---

## 4.3 Livello 3 – Runtime Orchestrator (Online/Batch)

Pipeline modulare a step.

Tecnologia suggerita:
- LangGraph (per state machine agentica)
- Oppure orchestrazione custom Python

---

# 5. Workflow Agentico

Pipeline composta da moduli indipendenti e componibili.

---

## Step 0 – Query Understanding

Input:
- Domanda + opzioni

Output:
- Query multiple generate:
  - keyword-style
  - semantic paraphrase
  - query espanse con sinonimi

---

## Step 1 – Dual Retrieval

- BM25 retrieval
- Vector retrieval
- Merge risultati
- Deduplica

Output:
- candidates[]

---

## Step 2 – Hard Filtering

Regole deterministiche:

- Policy abrogazioni configurabile: nel benchmark **non escludere** leggi abrogate (al massimo deprioritizzare)
- Preferenza testi modificati vigenti
- Deduplica
- Rimozione contenuti non pertinenti

Obiettivo:
Ridurre falsi positivi prima dell’LLM.

---

## Step 3 – Reranking

- Cross-encoder o LLM reranker
- Ordinamento per rilevanza

---

## Step 4 – Evidence Builder

Costruzione set ristretto di evidenze:

- 3–8 estratti mirati
- Citazioni precise (legge + articolo + anchor)
- Snippet testuale

Obiettivo:
Contesto compatto e citabile.

---

## Step 5 – Answerer

Prompt vincolato:

- Deve scegliere una label tra le opzioni disponibili (nel benchmark attuale A..F)
- Deve citare solo evidenza disponibile
- Non può inventare fonti

Output:
- answer_label
- rationale
- citations

---

## Step 6 – Self-Evaluation

Valuta:

1. Sufficienza del contesto
2. Faithfulness della risposta
3. Copertura delle opzioni

Output strutturato:

- sufficiency_score
- faithfulness_score
- missing_aspects
- suggested_actions

---

## Step 7 – Correction Loop (max N iterazioni)

Azioni possibili:

- Query rewrite
- Aumento K
- Graph expansion
- Follow-the-citation
- Metadata filtering

---

# 6. Benchmark e Valutazione

## 6.1 Dataset

- File benchmark: `questions.csv`
- Record totali: 163
  - 100 domande complete (ID 1..100): usate per valutazione
  - 63 righe vuote (ID 101..163): ignorate
- Ground truth:
  - risposta corretta (label A..F)
  - riferimenti normativi (colonna `Riferimento legge per la risposta`) usati come gold target per retrieval/citations

## 6.2 Metriche principali

1. Accuracy MCQ
2. Citation Presence Rate
3. Gold citation hit (almeno un riferimento gold citato)
4. Retrieval Recall@K sui gold target (legge+articolo)
5. Faithfulness
6. (opzionale) Gold citation full coverage (tutti i riferimenti gold citati)

## 6.3 Errori critici

Errore più grave:
- Falso positivo (citazione non pertinente)

Errore meno grave:
- Falso negativo (mancata copertura)

---

# 7. Requisiti Non Funzionali

## 7.1 Tracciabilità

Il sistema deve registrare:

- Query generate
- Chunk recuperati
- Score BM25
- Score vector
- Score rerank
- Decisioni filtro
- Prompt inviati
- Output LLM

Ogni run deve avere un `trace_id`.

---

## 7.2 Explainability

Ogni risposta deve includere:

- Fonti citate
- Estratti testuali
- Motivazione collegata alla fonte

---

## 7.3 Modularità

Il sistema deve consentire:

- Inserimento nuovi step nella pipeline
- Sostituzione retriever
- Attivazione/disattivazione graph expansion
- Cambiare modello LLM senza refactoring globale

---

## 7.4 Dataset congelato (snapshot) e determinismo

Il sistema lavora su una snapshot fissa del corpus (nessun aggiornamento incrementale in questa fase).

Requisiti:
- `dataset_id`: hash deterministico della snapshot di `leggi-html/`
- `schema_version`: versione dello schema di output (es. `v1`)
- Determinismo: stessa snapshot e stessa versione parser ⇒ stessi ID/record (a parità di configurazione)

---

# 8. Estensioni Future

- Supporto domande aperte
- Versioning temporale (norma vigente in anno X)
- Inclusione giurisprudenza
- Multi-jurisdiction
- Retrieval gerarchico articolo → comma
- Cross-encoder locale

---

# 9. Milestone di Sviluppo

## Milestone 1 – Baseline deterministica

- Parsing HTML
- Indice BM25 + embeddings
- Retrieval + risposta MCQ
- Benchmark 100 domande

## Milestone 2 – Agentic loop

- Self-evaluation
- Query rewrite
- Rerank dinamico
- Misura delta performance

## Milestone 3 – Graph integration

- Costruzione grafo relazioni
- Graph expansion step
- Riduzione falsi positivi da abrogazioni

---

# 10. Principi Guida del Progetto

1. Prima baseline misurabile, poi agentic.
2. Ridurre allucinazioni prima di migliorare recall.
3. Ogni step deve essere loggabile.
4. Ogni miglioramento deve essere misurabile.
5. Architettura semplice, pipeline espandibile.

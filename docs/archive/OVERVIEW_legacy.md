## Documento di overview – RAG incrementale per QA su normativa (corpo in HTML)

### Problema e contesto
Le normative di riferimento sono disponibili in **formato HTML**, scelta utile perché:
- preserva la **struttura logica** (titoli, articoli, commi, rubriche);
- include **annotazioni** e attributi (es. **id legge**, **id articolo/comma**, riferimenti/citazioni verso altre leggi o articoli);
- consente di estrarre **metadati affidabili** da usare nel retrieval (filtri, ranking, contesto aggiuntivo).

**Problema principale:** un RAG “naive” applicato a testi normativi lunghi tende a soffrire di:
- recupero incompleto (missing del pezzo “giusto”);
- recupero rumoroso (chunk non pertinenti);
- perdita di contesto dovuta a chunking non coerente con la struttura legale;
- difficoltà nel gestire rinvii e dipendenze tra articoli/leggi.

**Obiettivo**: progettare un’architettura **incrementale e misurabile** che migliori progressivamente l’accuratezza del retrieval (e quindi delle risposte), sfruttando la ricchezza dell’HTML per costruire metadati e controllare meglio la selezione del contesto.

---

### Obiettivo del sistema
Costruire un sistema di **Question Answering** su normativa che:
- recuperi i passaggi normativi più pertinenti;
- generi risposte coerenti e verificabili, idealmente con riferimenti (articolo/comma);
- consenta una valutazione sperimentale comparabile tra diverse varianti di retrieval.

---

### Architettura a step (dal Naive RAG a retrieval avanzato)

#### Step 0 — Dataset e protocolli di valutazione
- Definizione del set di domande (generico: definizioni, obblighi, eccezioni, sanzioni, procedure, casi d’uso).
- Ground truth: riferimenti attesi (articoli/commi) e/o criteri di correttezza.
- Metriche minime:
  - Retrieval: Recall@k (presenza del chunk corretto nei top-k), precision qualitativa dei chunk.
  - Risposta: correttezza (rubrica), completezza, aderenza alle fonti (citazioni).

#### Step 1 — Naive RAG (baseline)
- Parsing HTML → testo “pulito”.
- Chunking semplice (per lunghezza fissa o per paragrafi).
- Embedding + vector store.
- Retrieval top-k per similarità.
- Prompt con chunk recuperati → generazione risposta.

**Output atteso:** baseline funzionante, utile per misurare miglioramenti successivi.

#### Step 2 — Chunking strutturale + metadati “di base” (sfrutto l’HTML)
- Chunking allineato a struttura legale: (es.) articolo → commi; o articolo con overlap controllato.
- Metadati estratti dall’HTML:
  - id legge / id documento
  - id articolo/comma
  - titolo/rubrica sezione
  - gerarchia (capo/titolo/sezione)
- Nel prompt, ogni chunk porta “intestazione” (es. `[Legge X | Art. 12 | comma 2]`).

**Obiettivo:** ridurre perdita di contesto e aumentare precisione/interpretabilità.

#### Step 3 — Retrieval ibrido (keyword + semantico)
- Aggiunta di ricerca lessicale (BM25) affiancata al vettoriale.
- Fusione risultati (union + scoring) o routing semplice:
  - query con riferimenti espliciti (“art. 5”, numeri, termini tecnici) → boost keyword
  - query concettuali/parafrasate → boost semantico

**Obiettivo:** coprire sia match “esatto” sia match “semantico”, aumentando recall.

#### Step 4 — Reranking e riduzione del rumore
- Recupero ampio (es. top-20) → **reranker** (cross-encoder) per scegliere top-n finali.
- Deduplica/anti-ridondanza (evitare chunk troppo simili).
- Eventuale “context window budgeting”: selezionare chunk massimizzando copertura tematica.

**Obiettivo:** aumentare precisione del contesto finale dato all’LLM.

#### Step 5 — Query enhancement (senza agent completo)
- Query rewriting controllato (1 passaggio):
  - estrazione termini chiave giuridici
  - espansione sinonimi/varianti (se utile)
- (Opzionale) decomposizione semplice in sotto-quesiti per query multi-parte, con merge dei contesti.

**Obiettivo:** migliorare retrieval quando domanda e testo normativo sono lessicalmente distanti.

#### Step 6 — Agentic retrieval “leggero” (orchestrazione guidata)
- L’LLM decide se:
  - riformulare la query,
  - fare un secondo retrieval mirato,
  - applicare filtri sui metadati (es. restringere a una legge/sezione),
  - verificare copertura (manca definizione? manca eccezione?).
- Produzione risposta finale solo dopo “sufficiente evidenza” nei chunk.

**Obiettivo:** robustezza su domande difficili con iterazioni minime e controllate.

---

### 5) Ruolo centrale dell’HTML e dei metadati
L’HTML non è solo un formato di input: è una fonte di **segnali strutturali** per:
- chunking coerente con la semantica legale;
- filtri/routing (per legge, sezione, articolo);
- ranking (priorità a chunk “più vicini” nella gerarchia o con rubriche rilevanti);
- tracciabilità (citazioni e provenienza della risposta).

---

### 6) Modelli da testare (overview)
- **Closed-source**: modelli generalisti forti per valutare “upper bound” di generazione con buon contesto.
- **Open-source**: modelli efficienti/controllabili; preferibilmente varianti con buona resa in italiano e, se disponibili, adattamenti legal.
- La variabile centrale del progetto rimane **retrieval + contesto**, quindi i modelli vanno testati in modo comparabile (stesso retrieval, stesso prompt template, stessa metrica).

---

### 7) Fuori scopo (ma citabile come evoluzione): GraphRAG
GraphRAG viene mantenuto **fuori scope** per l’implementazione attuale. Si cita come possibile estensione futura per:
- gestione avanzata di rinvii tra articoli/leggi;
- ragionamento multi-hop e aggregazione di conoscenza distribuita;
- costruzione di un knowledge graph da entità/relazioni estratte.

---

### 8) Deliverable atteso della tesi (in breve)
- Pipeline RAG incrementale con confronti sperimentali step-by-step.
- Evidenza quantitativa/qualitativa di quali migliorie aumentano recall/precision e accuratezza risposta.
- Discussione finale del trade-off tra performance e manutenibilità nel dominio normativo.

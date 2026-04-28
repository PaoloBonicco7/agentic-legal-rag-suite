# Retrieval & RAG
Status: Roadmap (target)  
Scope: retrieval, context building, answering  
Source of truth: `docs/requirements.md`

Questo documento definisce l’approccio target per retrieval e RAG, con stack **Qdrant-first** e ottimizzazioni incrementalmente attivabili.

## Indexing (Qdrant-first)
- Backend principale: Qdrant persistente su path locale (es. `data/indexes/qdrant`).
- Ogni chunk indicizzato deve includere metadati minimi: `chunk_id`, `law_id`, `article_label_norm`, `passage_label`, status e (quando disponibile) relazioni denormalizzate.

## Varianti dataset per retrieval
- `plain`: chunk “puliti” senza intestazioni.
- `with_metadata`: chunk con prefisso contestuale (es. `[law_id | Art. X | passage]`) per aumentare interpretabilita' e ridurre ambiguita'.

## Retrieval baseline
1. Query building (domanda + opzionalmente opzioni A..F).
2. Similarity search top-k su Qdrant.
3. Context builder con limiti di budget (max chars/tokens).

## Ottimizzazioni incrementalmente attivabili
### Hybrid retrieval (BM25 + vector)
- BM25 per match esatti (articoli, numeri, termini giuridici).
- Merge deterministico (policy documentata: pesi, tie-breakers, dedup).

### Hard filtering (anti-allucinazione “prima dell’LLM”)
- Dedup per articolo (`(law_id, article_label_norm)`).
- Token/context budgeting.
- Policy abrogazioni: nel benchmark non escludere, al massimo deprioritizzare.

### Reranking
- Recupero largo (es. top-20/50) seguito da reranker (cross-encoder o LLM reranker) per selezione finale top-n.
- Obiettivo: aumentare precisione del contesto finale.

### Query rewrite (controllato)
- Riscrittura in 1 passaggio, con vincoli:
  - estrazione termini chiave
  - normalizzazione riferimenti (art., comma, numeri)
  - evitare drift semantico

## Evidence builder e policy citazioni
- Evidence set: 3–8 estratti (chunk/snippet) con intestazione e `chunk_id`.
- Policy runtime: `citations` = lista di `chunk_id` presenti nel contesto (vincolo forte nel prompt).
- Policy valutazione: vedi `docs/requirements.md` e `docs/evaluation.md`.


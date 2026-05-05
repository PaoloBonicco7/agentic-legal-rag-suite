# 06b - UI Advanced Graph RAG

Questa UI Streamlit serve a testare una domanda libera contro la pipeline Advanced Graph RAG e a visualizzare, passo per passo, cosa accade durante il retrieval. Non sostituisce i notebook e non produce artifact in `data/rag_runs/`: è uno strumento locale di ispezione e demo.

## Prerequisiti

- Dataset pulito e indice Qdrant già generati dagli step 01 e 03.
- File `.env` o variabili d'ambiente con credenziali Utopia, in particolare `UTOPIA_API_KEY`.
- Dipendenze UI installate:

```bash
uv sync --group dev --group notebooks --group ui
```

Se l'indice corrente è dense-only, l'opzione hybrid viene mostrata ma resta disabilitata finché manifest, collection Qdrant ed embedder non espongono sparse vectors.

## Avvio

Da root repository:

```bash
uv run --group ui streamlit run src/legal_rag/ui/advanced_rag_app.py
```

Streamlit apre la UI su un indirizzo locale, normalmente:

```text
http://localhost:8501
```

Se Qdrant locale segnala che la cartella dell'indice è già in uso, chiudere notebook, processi Python o altre UI che stanno accedendo a `data/indexes/qdrant`, poi riavviare Streamlit.

## Uso

1. Controllare nella sidebar i path principali: evaluation dir, laws dir, index dir, manifest e collection.
2. Verificare nel pannello "Stato runtime" che collection, modello embedding e modello chat siano corretti.
3. Impostare i flag del retrieval:
   - metadata filters;
   - hybrid retrieval, solo se disponibile;
   - graph expansion;
   - LLM rerank.
4. Scrivere una domanda nel campo chat in basso.
5. Leggere la risposta e aprire i tab:
   - `retrieval iniziale`: chunk recuperati da Qdrant;
   - `graph expansion`: edge usati e chunk aggiunti dal grafo;
   - `rerank`: ordine finale e score del reranker;
   - `contesto finale`: chunk effettivamente passati al modello;
   - `citazioni`: fonti citate dalla risposta.

La tabella delle metriche mostra i risultati già presenti nei run salvati, mentre i timing indicano solo la singola domanda eseguita nella UI.

# 01 — Preprocessing delle leggi

Pipeline deterministica HTML → dataset pulito in `src/legal_rag/laws_preprocessing/`. Legge il
corpus versionato `data/laws_html/`, estrae struttura legale e relazioni esplicite, costruisce i
chunk per il retrieval e scrive gli artifact in `data/laws_dataset_clean/`. Il corpus sorgente non
viene modificato; l'output è rigenerabile. Metodologia in
[note/01](../notes/01_laws_preprocessing_methodology.md).

## Configurazione

- Sorgente / output: `data/laws_html` → `data/laws_dataset_clean`
- Chunk: 600 parole, overlap 80 (word-based)
- Parser: `lxml.html`

## Risultati osservati

- File HTML validi: 3.145 (1 ignorato, `.DS_Store`)
- Leggi 3.145 · articoli 17.774 · passaggi 76.390 · note 8.380 · edge 35.159 · **chunk 76.467**
- Riferimenti non risolti: 277 · ID duplicati scartati: 0
- Tutti i quality gate superati; `manifest.json` espone `ready_for_indexing: true`.

## Limiti noti

- Le relazioni derivano solo da hyperlink e citazioni esplicite: nessuna inferenza legale.
- La risoluzione dei riferimenti è limitata alle leggi presenti nel corpus e ai pattern di citazione
  deterministici.
- `lxml.html` conserva alcune strutture di note/link malformate che il parser stdlib precedente
  saltava: i conteggi di note, edge e riferimenti non risolti risultano di poco superiori.

## Riproduzione

```bash
PYTHONPATH=src python -m legal_rag.laws_preprocessing --source data/laws_html --output data/laws_dataset_clean
```

Notebook: `notebooks/01_laws_preprocessing.ipynb`.

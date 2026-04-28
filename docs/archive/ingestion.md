# Ingestion (HTML -> dataset)
Status: Roadmap (target)  
Scope: ingestion pipeline  
Source of truth: `docs/data_model.md`

Questa sezione descrive la pipeline di ingestion che trasforma HTML normativi in un dataset RAG-ready.

## Requisiti chiave
- Parsing e metadati **deterministici** (no LLM).
- Segmentazione coerente con struttura legale (articoli, passages).
- Estrazione note e relazioni tra leggi best-effort.
- Output riproducibile con un `manifest` di QA e unresolved refs.

## Riferimento implementativo
Il dettaglio implementativo della pipeline e' documentato in:
- `laws_ingestion/README.md`

## Output atteso (baseline)
Dataset flat (JSONL) con almeno:
- `laws.jsonl`
- `articles.jsonl`
- `passages.jsonl`
- `notes.jsonl`
- `edges.jsonl`
- `chunks.jsonl` (input principale per embeddings/retrieval)
- `manifest.json` (config, counts, warnings/errors, unresolved refs)

## Note su struttura e metadati
- La struttura del testo e' preservata tramite:
  - `structure_path` (Parte/Titolo/Capo/Sezione) quando disponibile.
  - `article_label_norm` e `passage_label` per chunk/passage.
- Lo status (abrogazioni/modifiche) e' best-effort e deve essere tracciato come metadato, non “cotto” nel testo.


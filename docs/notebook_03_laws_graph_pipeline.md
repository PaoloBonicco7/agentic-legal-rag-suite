# Notebook 03 - Laws Graph Pipeline

## Obiettivo
`notebooks/03_laws_graph_pipeline.ipynb` costruisce un dataset legale deterministico e pronto al retrieval a partire da HTML normativi:
- parsing strutturato,
- status normativo,
- relazioni e eventi,
- chunk con viste `historical/current`,
- quality gates per handoff all'indexing.

## Input e prerequisiti
Input principale:
- `data/laws_html/`

Prerequisiti:
- ambiente Python del progetto attivo
- codice `src/laws_ingestion` importabile

Configurazione principale (via `PipelineConfig`):
- `html_dir`
- `output_dir`
- `run_root_dir`
- `sample_size` (opzionale)
- `backend`, `strict`, parametri chunking base ingest

## Flusso step-by-step
1. Setup root detection e bootstrap notebook.
2. Esecuzione pipeline `run_pipeline(config)`.
3. Step 01: inventory corpus + segnali lessicali.
4. Step 02: parsing strutturato (laws/articles/notes/chunks).
5. Step 03: classificazione status (`current/past/unknown/index_or_empty`) con evidenze.
6. Step 04: normalizzazione relazioni (`relation_type`, dedup, rimozione self-loop nel clean).
7. Step 05: estrazione eventi (`REPEAL/AMEND/REPLACE/INSERT`).
8. Step 06: arricchimento chunk con metadati grafo + `index_views`.
9. Step 07-08: export dataset clean + quality report e gate `ready_to_embedding`.

## Output e artifact
### Output pubblico
Cartella `data/laws_dataset_clean/`:
- `laws.jsonl`
- `articles.jsonl`
- `notes.jsonl`
- `edges.jsonl`
- `events.jsonl`
- `chunks.jsonl`
- `manifest.json`

### Artifact di run
Cartella `notebooks/data/laws_graph_pipeline/<run_id>/`:
- report step (`stepXX_report.json`)
- file intermedi step 01..06
- `step08_quality_report.md`
- metriche/grafici di QA

## Contratti con notebook precedente/successivo
- Contratto verso notebook 04:
- `manifest.json` valido
- `ready_to_embedding=true`
- `chunks.jsonl` con campi richiesti (`index_views`, metadati status/relazioni, ecc.)

Il notebook include un controllo handoff esplicito verso `04_qdrant_indexing_pipeline.ipynb`.

## Note operative essenziali
- Pipeline conservativa: nessuna legge viene rimossa, storico sempre preservato.
- In caso di ambiguita' lo status resta `unknown` (no inferenze aggressive).
- Se parquet engine non e' disponibile, alcuni artifact step usano fallback JSONL mantenendo estensione `.parquet`; il dataset clean resta usabile.

## Riferimenti codice
- Notebook: `notebooks/03_laws_graph_pipeline.ipynb`
- Pipeline: `src/laws_ingestion/data_preparation/laws_graph/pipeline.py`
- Status: `src/laws_ingestion/data_preparation/laws_graph/status.py`
- Relazioni: `src/laws_ingestion/data_preparation/laws_graph/relations.py`
- Eventi: `src/laws_ingestion/data_preparation/laws_graph/events.py`
- Views: `src/laws_ingestion/data_preparation/laws_graph/views.py`
- Quality report: `src/laws_ingestion/data_preparation/laws_graph/reporting.py`

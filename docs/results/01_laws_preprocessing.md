# 01 — Preprocessing delle leggi

La pipeline deterministica HTML → dataset pulito vive in `src/legal_rag/laws_preprocessing/`.
Legge il corpus versionato `data/laws_html/`, estrae struttura, relazioni ed eventi espliciti di
cessazione, quindi costruisce i chunk per il retrieval. Il corpus sorgente non viene modificato.
Metodologia in [note/01](../notes/01_laws_preprocessing_methodology.md).

## Run v2 per l'audit di vigenza

- Run: `2026-07-23T10:14:00Z`
- Output: `data/laws_dataset_clean_status_v2`
- Contratti: `laws-preprocessing-v2`, `legal-status-rules-v1`
- Chunking: 600 parole, overlap 80
- Git: `43ea1c990bf44cab1e0217dd0358b100ad05f7db`, worktree pulito
- Corpus SHA-256: `aa46ea3758c1de5596a8902b95e90cdfc6d08fbc2956086495a020680be22459`
- Manifest SHA-256: `65eb3cdca461a2a8d894959be9df3a34f3e03cc975d6df08c40505111da51b23`
- Chunk SHA-256: `dacbb03fd3debfd272031226ad539100cc228141acca94fba304fa28c663cb95`
- Eventi SHA-256: `ee740ae7b403a2bf6f4b666e59e6b243f493963ea047436bcead322fd4973917`

Il run contiene 3.145 leggi, 17.774 articoli, 76.422 passaggi, 8.395 note, 76.499 chunk,
35.159 edge e 3.310 eventi di stato. Tutti i quality gate sono verdi e
`ready_for_indexing=true`.

| Entità | current | partial | past | unknown |
|---|---:|---:|---:|---:|
| Leggi | 836 | 184 | 1.944 | 181 |
| Articoli | 14.063 | 361 | 3.040 | 310 |
| Passaggi | 68.352 | 4 | 7.935 | 131 |

La disponibilità del contenuto è separata dalla vigenza:

| Entità | substantive | unstructured | metadata_only | empty |
|---|---:|---:|---:|---:|
| Leggi | 1.239 | 1.905 | 1 | 0 |
| Articoli | 15.673 | 1.905 | 48 | 148 |
| Passaggi | 74.517 | 1.905 | 0 | 0 |

Gli eventi comprendono 3.290 `repeal` e 20 `expiration`: 2.931 sono risolti, 119 ambigui e
260 non applicati. Gli scope sono 1.948 `all`, 1.351 `only` e 11 `all_except`.

## Transizione v1 → v2

Tutti i 3.145 ID legge e i 17.774 ID articolo sono stabili fra i due artefatti. Il confronto
effettivo produce 1.211 leggi `current → past`, non 1.196: quest'ultimo era un conteggio
preliminare ottenuto prima del run definitivo.

| Stato legge v1 → v2 | Conteggio |
|---|---:|
| current → current | 836 |
| current → partial | 179 |
| current → past | 1.211 |
| current → unknown | 178 |
| past → past | 730 |
| unknown → partial / past / unknown | 5 / 3 / 3 |

La correzione più rilevante per i filtri riguarda il verso opposto: dei 938 articoli v1
etichettati `past`, 496 non risultano più interamente cessati in v2 (`40 current`,
`357 partial`, `99 unknown`). È l'evidenza quantitativa della propagazione eccessiva da commi o
lettere all'intero articolo.

Le viste v2 contengono 76.499 chunk `historical`, 49.131 `current` e 68.557
`not_explicitly_past`. `historical` è inclusiva, non una ricostruzione temporale.

## Baseline storico v1

L'artefatto `data/laws_dataset_clean/` resta necessario per riprodurre i risultati precedenti:
3.145 leggi, 17.774 articoli, 76.390 passaggi, 8.380 note e 76.467 chunk; chunk SHA-256
`9609ece84033a653f824a0fb7f31193a5a036176e8f82be65994a0c7cac82520`.
La sua distribuzione legge era 2.404 `current`, 730 `past`, 11 `unknown`; non deve più essere
usata per concludere che il corpus sia dominato da norme correnti.

## Limiti interpretativi

- `current` significa “nessuna cessazione esplicita rilevata nel corpus”, non certificazione
  giuridica.
- Modifiche, inserimenti e sostituzioni non implicano automaticamente che il testo consolidato
  sia `past`.
- Ambiguità editoriali o backlink insufficienti restano `unknown`.
- L'audit è corpus-only: non ricostruisce intervalli `valid_from` / `valid_to` e non esegue una
  verifica legale esterna.

## Riproduzione

```bash
PYTHONPATH=src python -m legal_rag.laws_preprocessing \
  --source data/laws_html \
  --output data/laws_dataset_clean_status_v2
```

Notebook: `notebooks/01_laws_preprocessing.ipynb`.

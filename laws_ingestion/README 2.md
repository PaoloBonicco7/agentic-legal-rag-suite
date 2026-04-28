# laws_ingestion

Pipeline deterministica (no LLM) per trasformare un corpus di leggi regionali in HTML (`data/laws_html/*.html`) in un dataset **RAG-ready** (JSONL flat) con:
- leggi e articoli strutturati
- segmentazione “passages” (intro/comma/lettera) per retrieval più preciso
- note (`nota_*`) estratte e collegate a articoli/passages
- relazioni tra leggi del corpus (edges) derivate da link e testo

> Obiettivo: partire da leggi strutturate in file HTML e ottenere un dataset indicizzabile (vector store/BM25) + metadati utili (status, connessioni).

## Posizione codice
- Codice Python runtime: `src/laws_ingestion/`
- Namespace pipeline notebook 03: `laws_ingestion.data_preparation.laws_graph`

## Nota notebook
- Il percorso operativo aggiornato per il pre-processing esteso e' `notebooks/03_laws_graph_pipeline.ipynb`.
- I notebook legacy restano in archivio sotto `notebooks/old/` e non devono essere rimossi in modo distruttivo.

---

## Punto di partenza: cosa contengono gli HTML
### 1) Filename = metadati affidabili (id/data/numero)
I file hanno naming standard tipo:
`<prefix>_LR-<day>-<month>-<year>-n<num>.html`  
Esempio: `1967_LR-5-settembre-1991-n44.html`

Da cui deriviamo:
- `law_date = 1991-09-05`
- `law_number = 44`
- `law_id = vda:lr:1991-09-05:44` (ID canonico del dataset)

Questo è la **source of truth** per identificare la legge: niente matching fuzzy sul contenuto.

### 2) Struttura interna tipica
Nel corpus reale, il contenuto utile è quasi sempre in:
- wrapper `<article> ... </article>` (se manca, facciamo fallback sull’intero documento)
- blocchi lineari `<h1>`, `<h2>`, `<p>` e talvolta `<table>`

Noi “linearizziamo” l’HTML in una sequenza ordinata di **blocchi** con:
- `text` (whitespace normalizzato)
- `anchors` (`<a name="...">...</a>`)
- `links` (`<a href="...">...</a>`)

### 3) Segnali strutturali che sfruttiamo
**Articoli**
- Articoli marcati da anchor:
  - `name="articolo_..."` con testo tipo `Art. 1`
- Fallback per HTML storici:
  - righe tipo `ARTICOLO 1 ...` / `ART. 1 ...`

**Indice/TOC (da scartare dal preambolo)**
- spesso compare `INDICE` + righe con link interni `href="#articolo_..."` (navigazione, non contenuto normativo)

**Note**
- Citazioni di note nel testo: link interni `href="#nota_01"` (o simili)
- Definizioni note: anchor `name="nota_01"` con testo della nota (modifiche/abrogazioni/insert)

**Relazioni tra leggi (link esterni “relativi”)**
Molti riferimenti ad altre leggi sono codificati come link relativi alla webapp sorgente, ad es.:
`href="/app/leggieregolamenti/dettaglio?tipo=L&numero_legge=45%2F95&versione=V"`

Questi URL **non vengono seguiti** (offline non esistono), ma li usiamo come segnali strutturali per costruire il grafo delle relazioni.

**Heading di struttura**
Riconosciamo intestazioni tipo `PARTE / TITOLO / CAPO / SEZIONE ...` per costruire `structure_path` (contesto gerarchico dell’articolo).

---

## Come trasformiamo gli HTML (high-level)
1) **Parsing HTML → Blocks** (`stdlib` o `bs4+lxml` se disponibile)  
2) **Segmentation**:
   - `preamble_text`: testo prima del primo articolo (scartando TOC)
   - `articles`: lista articoli (anchor `articolo_*` o fallback testuale)
3) **Passages** (dentro ogni articolo):
   - `intro`: prima del primo comma numerato
   - `comma`: pattern tipo `1.` / `1bis.` / `2.`
   - `lettera`: pattern tipo `a)` / `b)` (annidata nel comma corrente)
4) **Note**:
   - estraiamo le definizioni `nota_*`
   - colleghiamo note → articoli/passages usando i link `href="#nota_*"` trovati nel testo
5) **Relazioni (edges)**:
   - da `href` con `numero_legge=...` + da regex nel testo (es. `L.R. 24/2002`, `Legge regionale <data>, n. <num>`)
   - classifichiamo l’edge in `REFERS_TO / AMENDED_BY / ABROGATED_BY` usando keyword e contesto (preambolo/note/passages)
6) **Status**:
   - `Law.status` e `Law.abrogated_by` (best-effort) da marker nel preambolo + edges
   - `Article.is_abrogated/amended_by_law_ids` (best-effort) da note collegate
7) **Chunking**:
   - chunk = split word-based dei **passages** (`max_words=600`, `overlap_words=80`)
   - ogni chunk include prefisso contestuale + metadati (incl. `related_law_ids` / `relation_types`)

---

## Come interpretiamo e risolviamo i link `href` verso altre leggi
Esempio (offline “non cliccabile” ma utile):
`/app/leggieregolamenti/dettaglio?tipo=L&numero_legge=45%2F95&versione=V`

Risoluzione deterministica a `law_id` (solo intra-corpus), in ordine:
1) riferimento completo nel testo del link (es. `L.R. 23 ottobre 1995, n. 45`)
2) riferimento corto nel testo (es. `l.r. 24/2002`) → `(year, num)`
3) fallback dal parametro `numero_legge=NUM/YY|YYYY`:
   - `YY>=50 => 19YY`, altrimenti `20YY`
   - risoluzione via indice registry `(year, num) -> law_id` (solo se univoca)

Se non risolvibile/ambiguo, non creiamo l’edge e incrementiamo `manifest.json: unresolved_refs`.

---

## Output: dataset RAG-ready (JSONL flat)
I file vengono scritti in `out_dir/`:
- `laws.jsonl`: 1 riga = 1 legge (metadati + preambolo + status)
- `articles.jsonl`: 1 riga = 1 articolo (testo + struttura + note collegate + status)
- `passages.jsonl`: 1 riga = 1 passage (`intro/cX/cX.lit_a/...`) + `related_law_ids` + `relation_types`
- `notes.jsonl`: 1 riga = 1 nota (`nota_*`) + `linked_article_ids` + `linked_passage_ids`
- `edges.jsonl`: relazioni tra leggi del corpus (`REFERS_TO/AMENDED_BY/ABROGATED_BY`) con `extraction_method=href|text_regex`
- `chunks.jsonl`: input principale per embeddings/retrieval (chunk per passage, con metadati denormalizzati)
- `manifest.json`: `dataset_id`, config, counts, warning/error sample, `unresolved_refs`

---

## CLI
```bash
python -m laws_ingestion ingest --html-dir data/laws_html --out-dir data/laws_dataset
# Variante rapida: ingest solo delle leggi referenziate dal benchmark MCQ
python -m laws_ingestion ingest --scope benchmark --csv data/evaluation/questions.csv --html-dir data/laws_html --out-dir data/laws_dataset_benchmark
python -m laws_ingestion qa --out-dir data/laws_dataset
python -m laws_ingestion debug-law --html-dir data/laws_html --law-id vda:lr:1991-09-05:44
```

## Note
- Best-effort by default: non blocchiamo l’intera run per singoli HTML “strani”; gli errori finiscono nel manifest.
- `--strict` rende la run fail-fast su errori strutturali.
- Moduli non-ingestion (benchmark/BM25/eval) sono in `baselines/` per mantenere `laws_ingestion/` minimale.

# Modello dati
Status: Roadmap (target)  
Scope: data model and IDs  
Source of truth: `docs/requirements.md`

Questo progetto parte da HTML normativi e produce un dataset RAG-ready con entita' e metadati stabili. La terminologia e i campi qui sotto devono rimanere coerenti in ingestion, indexing e evaluation.

## Entita' principali
### Law
Metadati minimi:
- `law_id`: ID canonico della legge (deterministico, stabile).
- `law_date`: data legge (ISO).
- `law_number`: numero legge.
- `title`: titolo (best-effort).
- `status`: es. `vigente` / `abrogata` (best-effort).
- `abrogated_by`: lista di `law_id` (best-effort).
- `amended_by`: lista di `law_id` (best-effort).

### Article
Metadati minimi:
- `article_id`: ID stabile (derivabile da `law_id` + label/anchor).
- `law_id`: parent.
- `article_label_norm`: label normalizzata (es. `Art. 12`).
- `anchor_id`: anchor HTML se disponibile.
- `structure_path`: percorso gerarchico (Parte/Titolo/Capo/Sezione) best-effort.
- `text`: testo articolo.

### Passage
Segmento piu' fine dentro un articolo, allineato a:
- `intro` (testo introduttivo),
- `comma` (1., 2., 1bis.),
- `lettera` (a), b), ...).

Campi:
- `passage_id`: ID stabile.
- `article_id`, `law_id`
- `passage_label`: es. `intro`, `c1`, `c1.lit_a`
- `text`

### Note
Campi:
- `note_id`: es. `nota_01`
- `text`
- collegamenti a `article_id` / `passage_id` dove viene citata (best-effort).

### Edge (relazioni tra leggi)
Campi:
- `src_law_id`
- `dst_law_id`
- `relation_type`: `REFERS_TO` / `AMENDED_BY` / `ABROGATED_BY`
- `extraction_method`: `href` / `text_regex` (o simili)

### Chunk (unita' indicizzabile)
E' l'unita' di retrieval ed embedding. Campi minimi (coerenti con l'implementazione corrente in `src/legal_indexing/`):
- `chunk_id`
- `law_id`
- `article_id` (opzionale)
- `passage_id` (opzionale)
- `text` (contenuto mostrato come evidenza)
- `text_for_embedding` (contenuto “pulito” per embedding)
- `article_label_norm` (opzionale ma raccomandato)
- `passage_label` (opzionale ma raccomandato)
- `law_status`, `article_is_abrogated` (best-effort)
- `related_law_ids`, `relation_types` (best-effort, denormalizzati per retrieval)

## ID canonici (decisione)
L’ID canonico della legge deve essere derivato in modo deterministico dal filename, come descritto in `laws_ingestion/README.md`.
Esempio:
- filename: `1967_LR-5-settembre-1991-n44.html`
- `law_id`: `vda:lr:1991-09-05:44`

Gli ID derivati (`article_id`, `passage_id`, `chunk_id`) devono essere stabili e ricostruibili a partire da `law_id` e dalla struttura (anchor/label), evitando contatori non deterministici.

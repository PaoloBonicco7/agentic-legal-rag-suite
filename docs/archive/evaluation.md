# Valutazione
Status: Roadmap (target)  
Scope: benchmark MCQ + retrieval metrics  
Source of truth: `docs/requirements.md`

## Dataset benchmark
- File: `evaluation/questions.csv`
- Ogni riga rappresenta una domanda MCQ con:
  - testo domanda con opzioni A..F (nel campo “Domanda” con newline embedded)
  - label corretta (A..F)
  - riferimento/i normativi gold (testuali)

## Metriche retrieval
Obiettivo: misurare se il retrieval recupera l’evidenza corretta.
- `recall@k`: frazione di domande per cui almeno un documento gold e' presente nei top-k.
- `hit-rate@k`: variante “binary” (hit/no-hit) aggregata.

Definizione di “hit” (decisione):
- mappiamo il riferimento gold (testuale) su entita' del dataset quando possibile;
- un retrieved doc contribuisce a “hit” se i metadati matchano almeno `(law_id, article_label_norm)` (quando disponibile).

## Metriche MCQ
- `accuracy` complessiva e per livello.
- breakdown errori:
  - retrieval miss (gold non recuperato)
  - answer miss (gold recuperato ma risposta sbagliata)
  - citation issues (citazioni non valide)

## Metriche citazioni
Coerenti con `docs/requirements.md`:
- `gold_citation_hit`: almeno una citation (chunk_id) corrisponde al gold via mapping metadati.
- `invalid_citations`: citations non presenti nel contesto (target: 0).

## Artefatti di run (target)
Ogni esecuzione salva in una directory per run:
- config (modelli, k, varianti)
- retrieved docs (chunk_id, score, metadata)
- output raw LLM e parse JSON
- summary metriche + report per domanda con `trace_id`


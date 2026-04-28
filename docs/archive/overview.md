# Overview
Status: Roadmap (target)  
Scope: project overview  
Source of truth: `docs/requirements.md`

## Problema
Le normative di riferimento sono disponibili in **formato HTML**: questo preserva struttura (titoli, articoli, commi), ancore, note e collegamenti tra leggi. Un RAG “naive” su testi lunghi soffre tipicamente di:
- recupero incompleto (manca il pezzo giusto);
- recupero rumoroso (chunk non pertinenti);
- chunking non allineato alla struttura legale;
- difficolta' nel gestire rinvii e dipendenze tra articoli/leggi.

## Obiettivo
Costruire una pipeline **RAG incrementale e misurabile** per Question Answering su normativa che:
- estragga dai HTML un dataset **RAG-ready** con metadati affidabili;
- migliori progressivamente la qualita' del retrieval con tecniche di ottimizzazione (hybrid, rerank, rewrite, loop agentico);
- produca risposte verificabili con citazioni puntuali e tracciabilita'.

## Principi di progetto
- **Determinismo** in ingestion e metadati (no LLM per parsing/metadata).
- **Incrementalita'**: un cambiamento per volta, confronto diretto rispetto alla baseline.
- **Misurabilita'**: metriche su retrieval + accuratezza/qualita' citazioni in benchmark MCQ.
- **Tracciabilita'**: ogni run produce artefatti e identificatori per audit.

## Fuori scope (ma citabile come evoluzione)
- GraphRAG/knowledge graph “pesante” come soluzione primaria. Si considera al massimo come estensione dopo che ingestion, retrieval e benchmark sono solidi.


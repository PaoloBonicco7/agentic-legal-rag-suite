# Compound legal reference review

The observations below concern frozen `data/corpora/current-final`, snapshot
`vda-efe319281d52e13a`, acquired on 10 October 2026. The correction is confined
to source code and offline regressions. Corpus, manifests, indexes and gold
remain unchanged; this review is not a legal certification.

## LR10/2026, article 9, paragraph 1

Source: [official page](https://www.consiglio.vda.it/app/leggieregolamenti/dettaglio?pk_lr=11824&versione=V),
local source `data/snapshots/current_20261010/sources/11824.html` and existing
regression fixture `tests/fixtures/current_11824.html` (ISO-8859-1).

The operative paragraph reads:

> I Comuni, entro novanta giorni dalla data di approvazione delle deliberazioni della Giunta regionale di cui agli articoli 6, comma 3, e 13, comma 2, della l.r. 20/1999, come modificati, rispettivamente, dal comma 1 e dal comma 4 dell'articolo 5 della presente legge, provvedono conseguentemente ad adeguare le deliberazioni di cui all'articolo 11 della medesima l.r. 20/1999.

In frozen `relations.jsonl`, record `rel:2923813afe31549c180d734a` has source
`vda:lr:2026-06-17:10#art:9#p:1`, target `vda:lr:1999-08-02:20`, resolution
`confirmed`, block `vda:lr:2026-06-17:10#b:00131`, span `[3,378)`.
This law-level edge does not identify articles 6 and 13 for article expansion.

Inspection of the retained HTML confirms two separate links to LR20/1999:
`articoli 6, comma 3, e 13, comma 2, della` governs the first link;
`articolo 11 della medesima` governs the second. Article 5 belongs to the
explicit internal expression `della presente legge` and must not become an
LR20/1999 target. The new regression resolves external articles 6, 13 and 11;
comma numbers 3 and 2 and internal article 5 never become those targets.

## LR54/1998, article 21, paragraph 3

Source: [official page](https://www.consiglio.vda.it/app/leggieregolamenti/dettaglio?pk_lr=2517&versione=V),
local source `data/snapshots/current_20261010/sources/2517.html` and existing
fixture `tests/fixtures/2517_LR-7-dicembre-1998-n54.html`.

The operative paragraph reads:

> Fermo restando quanto disposto dal comma 1 e dall'articolo 23 e nel rispetto del principio della separazione tra funzioni di direzione politica e funzioni di direzione amministrativa di cui agli articoli 4 e 5 della legge regionale 23 ottobre 1995, n. 45 (Riforma dell'organizzazione dell'Amministrazione regionale della Valle d'Aosta e revisione della disciplina del personale), lo statuto può attribuire al Consiglio la competenza di ulteriori atti, tra i quali, in particolare:

Frozen record `rel:7173e271632db8366652c082` has source
`vda:lr:1998-12-07:54#art:21#p:3`, target
`vda:lr:1995-10-23:45#art:23`, resolution `confirmed`; span
`vda:lr:1998-12-07:54#b:00393 [3,483)`.

The original HTML link encloses only `legge regionale 23 ottobre 1995, n. 45`;
the immediately governing text is `agli articoli 4 e 5 della`. The earlier
article 23 has a different scope. The old extractor ignored plural articles
and borrowed the last singular article anywhere in the clause prefix. The new
real-fixture regression resolves articles 4 and 5, never LR45/1995 article 23.
Registry counterparts in these tests are minimal synthetic laws with real
identities and decoy article nodes; they do not replace source fixtures.

## Deterministic correction and validation

`explicit-graph-3` considers only a supported article expression immediately
governing the specific law mention, after the previous citation. It supports
explicit article lists with single comma qualifiers and named law modifiers;
ranges, compound qualifiers and intervening prose remain unresolved. Missing
or duplicated target articles remain unresolved too. Passive interventions do
not silently substitute the law as actor for an unresolved cited article.
Cessation logic is unchanged.

On 11 October 2026 the following focused offline command passed 29 tests:

```bash
.venv/bin/python -m pytest tests/test_corpus_status_graph.py tests/test_corpus_parser.py::test_real_lr42_keeps_exceptions_and_historical_headings_out_of_body -q
```

Coverage includes both real compound references, note 40 of LR21/2003,
LR42/1992, unsupported lists, citation boundaries, missing/duplicate targets,
article suffixes and unresolved passive actors. These are software
regressions, not a corpus-wide semantic audit or a measured retrieval gain.

A future authorized rebuild must create a new corpus and index, recording the
new graph version. Expected local changes are replacement of the wrong LR54
edge by articles 4 and 5, and explicit LR10 targets 6, 13 and 11. Aggregate
edge counts, retrieval effects and benchmark scores have not been measured.
Existing-node and valid-span gates alone cannot establish semantic accuracy.

The [preprocessing walkthrough](05_preprocessing_walkthrough.html) shows a
separate supported multi-target note backlink in LR42/1992. That example does
not certify all graph relations; human review remains necessary for unsupported
expressions and evaluation evidence.

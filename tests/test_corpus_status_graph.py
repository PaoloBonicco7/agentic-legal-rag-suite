import pytest

from legal_rag.corpus.graph import extract_relations
from legal_rag.corpus.status import assess_status

from test_corpus_parser import FIXTURES, fixture, law, parse


@pytest.mark.parametrize("clause", [
    "Non abrogata dalla L.R. 2 gennaio 2001, n. 2.",
    "Abrogata dalla L.R. 2 gennaio 2001, n. 2, eccetto l'articolo 1.",
    "Abrogata dalla L.R. 2 gennaio 2001, n. 2 dal 2030.",
    "Abrogata dalla L.R. 2 gennaio 2001, n. 2 dal 1 gennaio 2030.",
    "Abrogata dalla L.R. 2 gennaio 2001, n. 2 dal 1° gennaio 2030.",
    "Abrogata dalla L.R. 2 gennaio 2001, n. 2; restano salvi alcuni effetti.",
    "Abrogata dalla L.R. 2 gennaio 2001, n. 2. Dal 2030.",
    "La legge regionale 3 gennaio 1999, n. 3 è abrogata.",
])
def test_unsafe_formulas_never_exclude(clause):
    document, blocks, units = parse(law(f"<p>{clause}</p><p>Art. 1</p><p>1. Testo normativo utile.</p>"))
    assess_status(document, blocks, units, "2026-10-10")
    assert document.status != "ceased_confirmed"
    assert all(u.status != "ceased_confirmed" for u in units)


def test_dated_cessation_is_confirmed_only_after_effect_date():
    text = law("<p>Abrogata dalla L.R. 2 gennaio 2001, n. 2 dal 1 gennaio 2025.</p><p>Art. 1</p><p>1. Testo normativo utile.</p>")
    document, blocks, units = parse(text)
    events = assess_status(document, blocks, units, "2024-12-31")
    assert document.status == "unknown"
    assert events[0].effective_at == "2025-01-01"
    document, blocks, units = parse(text)
    assess_status(document, blocks, units, "2026-10-10")
    assert document.status == "ceased_confirmed"
    assert all(u.status == "ceased_confirmed" for u in units)


def test_separate_conflicting_editorial_statement_vetoes_exclusion():
    document, blocks, units = parse(law('<p>Abrogata dalla L.R. 2 gennaio 2001, n. 2.</p><p>Non abrogata integralmente.</p><p>Art. 1</p><p>1. Testo utile.</p>'))
    events = assess_status(document, blocks, units)
    assert document.status == "unknown"
    assert all(event.resolution == "unresolved" for event in events)
    assert all(u.status != "ceased_confirmed" for u in units)


def test_real_repealed_law_retains_body_but_propagates_status():
    document, blocks, units = fixture("1967_LR-5-settembre-1991-n44.html")
    assert document.availability == "body"
    assess_status(document, blocks, units, "2026-10-10")
    assert document.status == "ceased_confirmed"
    assert all(u.status == "ceased_confirmed" for u in units)


def test_duplicate_note_anchor_and_multi_backlinks_abstain():
    text = law('<p>Art. 1</p><p>1. Primo testo. <a href="#nota_1">(1)</a></p><p>2. Secondo testo. <a href="#nota_1">(1)</a></p><p><a name="nota_1">(1)</a> Comma abrogato dalla L.R. 2 gennaio 2001, n. 2.</p><p><a name="nota_1">(1)</a> Comma abrogato dalla L.R. 2 gennaio 2001, n. 2.</p>')
    document, blocks, units = parse(text)
    events = assess_status(document, blocks, units, "2026-10-10")
    assert all(e.resolution == "unresolved" for e in events)
    assert all(u.status != "ceased_confirmed" for u in units)
    assert "duplicate_note_anchor:nota_1" in document.warnings


def test_article_note_cannot_bind_a_paragraph_marker():
    document, blocks, units = parse(law('<p>Art. 1</p><p>1. Primo testo. <a href="#nota_1">(1)</a></p><p><a name="nota_1">(1)</a> Articolo abrogato dalla L.R. 2 gennaio 2001, n. 2.</p>'))
    events = assess_status(document, blocks, units)
    assert events[0].resolution == "unresolved"
    assert all(u.status != "ceased_confirmed" for u in units)


def test_constitutional_ruling_against_old_formulation_never_excludes_new_body():
    document, blocks, units = parse(law('<p>Art. 22 <a href="#nota_1">(1)</a></p><p>1. Nuova disposizione del 2026.</p><p><a name="nota_1">(1)</a> La Corte costituzionale ha dichiarato illegittimo il testo sostituito dalla L.R. 1 gennaio 2025, n. 1.</p>'))
    events = assess_status(document, blocks, units, "2026-10-10")
    assert events[0].kind == "constitutional_ruling"
    assert events[0].resolution == "unresolved"
    assert all(u.status != "ceased_confirmed" for u in units)


def test_real_note40_assigns_only_clause_local_intervention():
    document, blocks, units = fixture("2711_LR-15-dicembre-2003-n21.html")
    repeal_doc, _, repeal_units = parse('<article><h1>Legge regionale 1 agosto 2011, n. 21</h1><p>Disposizioni regionali.</p><p>Art. 8</p><p>1. Abrogazioni.</p></article>', "2011")
    old_doc, _, old_units = parse('<article><h1>Legge regionale 27 novembre 1990, n. 75</h1><p>Disciplina regionale.</p><p>Art. 3bis</p><p>1. Testo regionale.</p></article>', "1990")
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, repeal_doc, old_doc]}, units + repeal_units + old_units)
    note40 = next(b for b in blocks if b.anchor == "nota_40")
    edges = [r for r in relations if r.span and r.span.block_id == note40.block_id]
    repeals = [r for r in edges if r.kind == "repeals"]
    assert len(repeals) == 1
    assert repeals[0].source_id == repeal_doc.document_id + "#art:8"
    assert not any(r.kind == "repeals" and old_doc.document_id in r.source_id for r in edges)
    assert any(r.kind == "amends" and r.target_id == old_doc.document_id + "#art:3bis" for r in edges)
    assert all(r.evidence in note40.text for r in edges)


def test_unresolved_reference_is_preserved_and_historical_intervention_is_not_emitted():
    document, blocks, units = parse(law('<p>Art. 1</p><p>1. Testo utile. <a href="#nota_1">(1)</a></p><p><a name="nota_1">(1)</a> Comma modificato dalla L.R. 2 gennaio 2001, n. 2.</p><p>Nella formulazione originaria recitava:</p><p>Abroga l\'art. 9 della L.R. 3 gennaio 1999, n. 3.</p>'))
    relations = extract_relations(document, blocks, units, {document.document_id: document})
    historical = [r for r in relations if "3 gennaio 1999" in r.evidence]
    assert historical and all(r.kind == "references" and r.resolution == "unresolved" for r in historical)


def target_law(identity, labels):
    body = "".join(f"<p>Art. {label}</p><p>1. Disposizione regionale.</p>" for label in labels)
    document, _, units = parse(f"<article><h1>{identity}</h1><p>Disciplina regionale.</p>{body}</article>")
    return document, units


def test_real_lr54_compound_reference_never_binds_unrelated_article23():
    document, blocks, units = fixture("2517_LR-7-dicembre-1998-n54.html")
    target, target_units = target_law("Legge regionale 23 ottobre 1995, n. 45", ["4", "5", "23"])
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, target]}, units + target_units)
    source_id = document.document_id + "#art:21#p:3"
    edges = [r for r in relations if r.kind == "references" and r.source_id == source_id]
    assert {r.target_id for r in edges} == {target.document_id + "#art:4", target.document_id + "#art:5"}
    assert all(r.resolution == "confirmed" for r in edges)
    block = next(b for b in blocks if "Fermo restando quanto disposto dal comma 1 e dall'articolo 23" in b.text)
    assert all(r.span.block_id == block.block_id and block.text[r.span.start:r.span.end] == r.evidence for r in edges)


def test_real_lr10_list_with_comma_qualifiers_and_medesima_law():
    document, blocks, units = parse((FIXTURES / "current_11824.html").read_bytes().decode("iso-8859-1"))
    target, target_units = target_law("Legge regionale 2 agosto 1999, n. 20", ["2", "3", "5", "6", "11", "13"])
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, target]}, units + target_units)
    edges = [r for r in relations if r.kind == "references" and r.source_id == document.document_id + "#art:9#p:1"]
    assert len(edges) == 3
    assert {r.target_id for r in edges} == {target.document_id + f"#art:{label}" for label in ["6", "13", "11"]}
    assert all(r.resolution == "confirmed" for r in edges)
    block = next(b for b in blocks if "I Comuni, entro novanta giorni" in b.text)
    assert all(r.span.block_id == block.block_id and block.text[r.span.start:r.span.end] == r.evidence for r in edges)


@pytest.mark.parametrize("expression", [
    "articoli da 4 a 5 della",
    "articoli 4, commi 1 e 2, e 5 della",
    "articolo 4, comma 1 e 2 della",
    "articoli 4, comma 1, 5 della",
    "articolo 4 del regolamento adottato ai sensi della",
])
def test_unsupported_article_attachment_preserves_unresolved_literal(expression):
    citation = "Legge regionale 23 ottobre 1995, n. 45"
    document, blocks, units = parse(law(f"<p>Art. 1</p><p>1. Si applicano gli {expression} {citation}.</p>"))
    target, target_units = target_law(citation, ["1", "2", "4", "5"])
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, target]}, units + target_units)
    edges = [r for r in relations if r.kind == "references"]
    assert len(edges) == 1
    assert edges[0].target_id is None and edges[0].resolution == "unresolved"
    assert expression in edges[0].target_reference and citation in edges[0].target_reference


def test_article_attachment_does_not_cross_previous_law_mention():
    document, blocks, units = parse(law("<p>Art. 1</p><p>1. Si applica l'articolo 4 della legge regionale 23 ottobre 1995, n. 45, richiamata dalla legge regionale 2 agosto 1999, n. 20.</p>"))
    first, first_units = target_law("Legge regionale 23 ottobre 1995, n. 45", ["4"])
    second, second_units = target_law("Legge regionale 2 agosto 1999, n. 20", ["4"])
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, first, second]}, units + first_units + second_units)
    assert {r.target_id for r in relations if r.kind == "references"} == {first.document_id + "#art:4", second.document_id}


@pytest.mark.parametrize("duplicate", [False, True])
def test_explicit_article_with_missing_or_duplicate_target_stays_unresolved(duplicate):
    document, blocks, units = parse(law("<p>Art. 1</p><p>1. Si applica l'articolo 4 della legge regionale 23 ottobre 1995, n. 45.</p>"))
    target, target_units = target_law("Legge regionale 23 ottobre 1995, n. 45", ["4"] if duplicate else ["5"])
    if duplicate:
        target_units.append(next(u for u in target_units if u.kind == "article").model_copy())
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, target]}, units + target_units)
    edge = next(r for r in relations if r.kind == "references")
    assert edge.target_id is None and edge.resolution == "unresolved"
    assert edge.target_reference == target.document_id + "#art:4"


def test_supported_article_suffix_and_missing_law_retain_precise_references():
    document, blocks, units = parse(law("<p>Art. 1</p><p>1. Si applica l'articolo 4 sexies della legge regionale 23 ottobre 1995, n. 45. Si applicano gli articoli 6, comma 3, e 13, comma 2, della legge regionale 2 agosto 1999, n. 20.</p>"))
    target, target_units = target_law("Legge regionale 23 ottobre 1995, n. 45", ["4sexies"])
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, target]}, units + target_units)
    edges = [r for r in relations if r.kind == "references"]
    assert any(r.target_id == target.document_id + "#art:4sexies" for r in edges)
    missing = [r for r in edges if r.resolution == "unresolved"]
    assert {r.target_reference for r in missing} == {"vda:lr:1999-08-02:20#art:6", "vda:lr:1999-08-02:20#art:13"}


def test_passive_intervention_with_ambiguous_article_actor_never_falls_back_to_law():
    document, blocks, units = parse(law("<p>Art. 1</p><p>1. Testo utile.</p><p>Articolo abrogato dall'articolo 4 del regolamento adottato ai sensi della legge regionale 23 ottobre 1995, n. 45.</p>"))
    target, target_units = target_law("Legge regionale 23 ottobre 1995, n. 45", ["4", "5"])
    relations = extract_relations(document, blocks, units, {d.document_id: d for d in [document, target]}, units + target_units)
    assert any(r.kind == "references" and r.resolution == "unresolved" for r in relations)
    assert not any(r.kind in {"repeals", "amends"} for r in relations)

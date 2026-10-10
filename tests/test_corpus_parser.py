from pathlib import Path

from legal_rag.contracts import SourceRecord
from legal_rag.corpus.parser import extract_blocks, parse_source, span_text
from legal_rag.corpus.status import assess_status

FIXTURES = Path(__file__).parent / "fixtures"


def parse(text, source_id="1"):
    source = SourceRecord(source_id=source_id, url="https://example.org/law", path="source.html", sha256="unused")
    return parse_source(source, "snapshot-test", text)


def law(body):
    return '<article><h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1><p>Disciplina dei contributi regionali.</p><p>(B.U. 3 gennaio 2000).</p>' + body + "</article>"


def fixture(name):
    return parse((FIXTURES / name).read_text(encoding="utf-8"))


def test_full_page_excludes_navigation_and_preserves_nested_table_text_once():
    text = "<html><body><nav>Menu che non è normativa</nav>" + law('<p>Art. 1</p><p>(Requisiti)</p><p>1. Primo.<br>2. Secondo.</p><table><tr><td>Prima<p>Cella</p>Dopo</td></tr></table>') + "<footer>Footer</footer></body></html>"
    document, blocks, units = parse(text)
    assert document.title == "Disciplina dei contributi regionali."
    assert document.published_at == "2000-01-03"
    assert [b.text for b in blocks][-5:] == ["1. Primo.", "2. Secondo.", "Prima", "Cella", "Dopo"]
    assert not any("Menu" in b.text or "Footer" in b.text for b in blocks)
    by_block = {b.block_id: b for b in blocks}
    assert all(span_text(u.spans, by_block) == u.text for u in units)
    assert next(u for u in units if u.kind == "article").heading == "(Requisiti)"


def test_real_stub_has_title_and_no_invented_article():
    document, blocks, units = fixture("273_LR-10-ottobre-1950-n1.html")
    assert document.title == "Classificazione delle strade di interesse regionale."
    assert document.availability == "metadata_only"
    assert not units
    events = assess_status(document, blocks, units, "2026-10-10")
    assert document.status == "ceased_confirmed"
    assert len(events) == 1 and events[0].scope == "law"


def test_real_lr42_keeps_exceptions_and_historical_headings_out_of_body():
    document, blocks, units = fixture("2065_LR-19-agosto-1992-n42.html")
    events = assess_status(document, blocks, units, "2026-10-10")
    articles = {u.label: u for u in units if u.kind == "article"}
    assert set(articles) == {str(n) for n in range(1, 16)}
    assert articles["2"].status == "no_cessation_evidence"
    assert not articles["4"].text
    assert articles["4"].status == "ceased_confirmed"
    assert document.status == "unknown"
    assert any(e.scope == "law" and e.resolution == "unresolved" for e in events)
    assert "Personale straordinario" in " ".join(b.text for b in blocks if b.region == "historical")
    assert "Personale straordinario" not in " ".join(u.text for u in units)
    paragraphs = {u.label: u for u in units if u.kind == "paragraph" and u.article_label == "5"}
    assert paragraphs["4"].status == "ceased_confirmed"
    assert paragraphs["1"].status == "no_cessation_evidence"


def test_real_omitted_annex_is_separate_from_last_article():
    document, blocks, units = fixture("2144_LR-13-maggio-1993-n33.html")
    article = next(u for u in units if u.kind == "article" and u.label == "25")
    annex = next(u for u in units if u.kind == "annex")
    assert "Omissis" not in article.text
    assert "Omissis" in annex.text
    assert "omitted_annex" in document.warnings
    assert any(b.exclusion_reason == "omitted_annex" for b in blocks)


def test_list_without_numbered_intro_belongs_to_article():
    _, _, units = parse(law('<p>Art. 1</p><p>Sono ammesse le seguenti spese:</p><p>a) Acquisto di strumenti.</p><p>b) Formazione del personale.</p>'))
    letters = [u for u in units if u.kind == "letter"]
    assert all(u.parent_id.endswith("#p:intro:4") for u in letters)


def test_index_links_and_marker_only_paragraphs_are_not_normative():
    _, blocks, units = parse(law('<p>Indice</p><p><a href="#articolo_1__">Art. 1</a></p><p><a name="articolo_1__">Art. 1</a></p><p>1.<sup><a href="#nota_1">(1)</a></sup></p><p><a name="nota_1">(1)</a> Comma abrogato dalla L.R. 2 gennaio 2001, n. 2.</p>'))
    assert len([u for u in units if u.kind == "article"]) == 1
    assert next(u for u in units if u.kind == "paragraph").text == "1."
    assert sum(b.region == "index" for b in blocks) == 2


def test_extraction_splits_line_breaks_without_duplicate_links():
    blocks = extract_blocks('<article><p>Primo <a href="/x">riferimento</a>.<br>Secondo <a href="/y">rinvio</a>.</p></article>', "d")
    assert [b.text for b in blocks] == ["Primo riferimento.", "Secondo rinvio."]
    assert blocks[0].links[0]["start"] == "6"
    assert blocks[1].links[0]["start"] == "8"


def test_ordinal_first_day_and_conflicting_secondary_header_keep_primary_identity():
    document, blocks, units = parse('<article><h1 id="tmpl-banche-dati-titolo-secondario">Legge regionale 1<sup>°</sup> aprile 1987, n. 20 - Testo vigente</h1><h2>Legge regionale 18 marzo 1987, n. 19</h2><p>Legge regionale n. 20 del 01 04 1987</p><p>Bollettino ufficiale 30 04 1987 n. 8</p><p>Finanziamento delle spese per assistenza.</p></article>')
    assert document.document_id == "vda:lr:1987-04-01:20"
    assert document.identity == "Legge regionale 1° aprile 1987, n. 20"
    assert document.title == "Finanziamento delle spese per assistenza."
    assert document.published_at == "1987-04-30"
    assert "conflicting_identity_heading:vda:lr:1987-03-18:19" in document.warnings


def test_quoted_amended_article_does_not_become_a_duplicate_root_article():
    document, blocks, units = parse(law('<p>Art. 1</p><p>1. L\'articolo 7 è sostituito dal seguente:</p><p>"Art. 7</p><p>(Disciplina regionale)</p><p>1. Nuovo testo normativo.</p><p>2. Altra disposizione.".</p><p>Art. 7</p><p>1. Disposizione della legge modificatrice.</p>'))
    assert [u.label for u in units if u.kind == "article"] == ["1", "7"]
    assert next(u for u in units if u.kind == "article" and u.label == "1").text.count("Nuovo testo") == 1
    assert any(b.region == "body_quote" for b in blocks)
    assert not document.warnings


def test_inline_article_rubric_and_thousands_separator_are_structural():
    _, blocks, units = parse(law('<p>ARTICOLO 40. Compensi.</p><p>1. Sono dovute le seguenti somme.</p><p>1.000.000 lire per ogni beneficiario.</p><p>Art. 41 - (Omissis)</p>'))
    articles = [u for u in units if u.kind == "article"]
    assert [u.label for u in articles] == ["40", "41"]
    assert articles[0].heading == "Compensi."
    assert len([u for u in units if u.kind == "paragraph"]) == 1
    assert "1.000.000" in articles[0].text
    assert not articles[1].text


def test_current_repeated_paragraph_identity_is_not_the_thematic_title():
    document, blocks, units = parse((FIXTURES / 'current_11621.html').read_bytes().decode('iso-8859-1'))
    assert document.title == 'Disposizioni urgenti in materia di istruzione e di cultura. Modificazioni di leggi regionali.'
    assert blocks[1].exclusion_reason == 'identity_heading'
    assert not any(document.title in unit.text for unit in units)
    assert not any('Cookies' in block.text for block in blocks)


def test_current_multiline_title_and_unpunctuated_bu_are_metadata():
    document, blocks, units = parse((FIXTURES / 'current_11541.html').read_bytes().decode('iso-8859-1'))
    assert document.title.startswith('Modificazioni alla legge regionale 16 luglio 2024, n. 11 (Disciplina dell\'organizzazione')
    assert document.title.endswith('e di altre disposizioni in materia di lavoro e formazione professionale).')
    assert document.published_at == '2025-06-27'
    assert all(block.region == 'metadata' for block in blocks[:5])
    assert not any(unit.kind == 'unstructured' for unit in units)


def test_current_unclosed_leaf_quote_recovers_anchored_outer_article_without_rewriting_source():
    document, blocks, units = parse((FIXTURES / 'current_11824.html').read_bytes().decode('iso-8859-1'))
    articles = {unit.label: unit for unit in units if unit.kind == 'article'}
    assert set(articles) == {str(n) for n in range(1, 12)}
    assert 'Art. 19bis' in articles['2'].text
    assert '1.000.000' in articles['11'].text
    assert any(warning.startswith('source_quote_boundary_recovered:') for warning in document.warnings)
    assert 'unterminated_body_quote' not in document.warnings
    malformed = next(block for block in blocks if block.text.startswith('"2bis. Nei casi'))
    assert malformed.text.count('"') == 1
    assert any(span.block_id == malformed.block_id for span in articles['6'].spans)
    assert all(span_text(unit.spans, {block.block_id: block for block in blocks}) == unit.text for unit in units)


def test_numeric_identity_heading_variants_do_not_swallow_real_law_titles():
    text = '<article><h1>Legge regionale 28 luglio 2025, n. 24 - Testo vigente</h1><p>Legge regionale del 28 luglio 2025, n. 24</p><p>Legge regionale in materia di lavori pubblici.</p><p>(BU del 8 agosto 2025, n. 44)</p><p>Art. 1</p><p>1. Disposizione.</p></article>'
    document, blocks, units = parse(text)
    assert document.title == 'Legge regionale in materia di lavori pubblici.'
    assert document.published_at == '2025-08-08'
    assert not any(unit.kind == 'unstructured' for unit in units)

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('streamlit')

import streamlit as st
from streamlit.testing.v1 import AppTest

from legal_rag import runtime
from legal_rag.artifacts import write_json
from legal_rag.contracts import AnswerResult, Candidate, Chunk, Citation, Context, Evidence

APP = Path(__file__).parents[1] / 'src/legal_rag/ui/app.py'


@pytest.fixture(autouse=True)
def clear_engine_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def source_paths(tmp_path, *, date='2026-10-10T12:00:00+00:00'):
    corpus, index = tmp_path / 'corpus', tmp_path / 'index'
    write_json(corpus / 'manifest.json', {'snapshot_id': 'snapshot-fixture', 'snapshot_date': date})
    write_json(index / 'manifest.json', {'snapshot_id': 'snapshot-fixture', 'corpus_dir': str(corpus)})
    return corpus, index


def result(question, *, source_url='https://www.consiglio.vda.it/leggi/item/1'):
    chunk = Chunk(chunk_id='chunk-1', document_id='law-1', snapshot_id='snapshot-fixture',
        article_label='12', title='Competenze regionali', text='Il Consiglio esercita le funzioni legislative.',
        embedding_text='Il Consiglio esercita le funzioni legislative.', unit_ids=['law-1#art:12'],
        spans=[], source_url=source_url)
    evidence = Evidence(context_id='E1', chunk=chunk, rendered='[E1] ' + chunk.text)
    return AnswerResult(question=question, profile='hybrid', snapshot_id='snapshot-fixture',
        answer='Il Consiglio esercita le funzioni legislative.', status='answered',
        citations=[Citation(context_id='E1', quote=chunk.text, source_url=source_url, chunk_id=chunk.chunk_id)],
        candidates=[Candidate(chunk=chunk, rank=1, score=1, origin='hybrid')],
        context=Context(text=evidence.rendered, evidence=[evidence]))


def fake_factory(monkeypatch, answer=None, error=None):
    calls, questions = [], []

    def answer_question(question):
        questions.append(question)
        if error:
            raise error
        return answer(question) if answer else result(question)

    def create(config, policy, **kwargs):
        calls.append({'config': config, 'policy': policy, **kwargs})
        return SimpleNamespace(answer=answer_question), {}

    monkeypatch.setattr(runtime, 'create_engine', create)
    return calls, questions


def start_app(monkeypatch, *arguments):
    monkeypatch.setattr(sys, 'argv', [str(APP), *map(str, arguments)])
    return AppTest.from_file(str(APP)).run()


def test_snapshot_discovery_lazy_engine_and_stateless_answer_with_citation(monkeypatch, tmp_path):
    corpus, index = source_paths(tmp_path)
    calls, questions = fake_factory(monkeypatch)
    app = start_app(monkeypatch, '--index', index, '--timeout', 37, '--max-output-tokens', 512)
    assert not app.exception and not calls
    assert any('snapshot-fixture' in caption.value for caption in app.caption)
    assert any('2026-10-10' in caption.value for caption in app.caption)

    app.text_area[0].input('  Quali funzioni esercita il Consiglio?  ')
    app.button[0].click().run()
    assert not app.exception and questions == ['Quali funzioni esercita il Consiglio?']
    assert calls[0]['corpus_dir'] == str(corpus)
    assert calls[0]['config'].timeout_seconds == 37
    assert calls[0]['config'].max_output_tokens == 512
    assert any('funzioni legislative' in text.value for text in app.markdown)
    assert any('E1' in item.label and 'articolo 12' in item.label for item in app.expander)
    assert app.get('link_button')[0].url == 'https://www.consiglio.vda.it/leggi/item/1'

    app.text_area[0].input('Una seconda domanda').run()
    assert len(calls) == 1 and len(questions) == 1
    assert not any('funzioni legislative' in text.value for text in app.markdown)
    app.button[0].click().run()
    assert not app.exception and questions[-1] == 'Una seconda domanda'
    assert len(calls) == 1


def test_legacy_provenance_and_empty_question_do_not_load_engine(monkeypatch, tmp_path):
    corpus, _ = source_paths(tmp_path, date=None)
    calls, _ = fake_factory(monkeypatch)
    app = start_app(monkeypatch, '--corpus', corpus)
    assert any('snapshot-fixture' in caption.value for caption in app.caption)
    assert any('sconosciuta' in warning.value for warning in app.warning)
    app.button[0].click().run()
    assert not app.exception and not calls
    assert any('Scrivi una domanda' in warning.value for warning in app.warning)


def test_missing_or_mismatched_provenance_is_safe_and_does_not_load_engine(monkeypatch, tmp_path):
    corpus, index = source_paths(tmp_path)
    write_json(index / 'manifest.json', {'snapshot_id': 'another-snapshot', 'corpus_dir': str(corpus)})
    calls, _ = fake_factory(monkeypatch)
    app = start_app(monkeypatch, '--index', index)
    assert not app.exception and not calls and app.error
    assert not app.text_area
    app = start_app(monkeypatch, '--corpus', tmp_path / 'private-secret-path')
    assert not app.exception and not calls and app.error
    assert 'private-secret-path' not in app.error[0].value


@pytest.mark.parametrize('returned_error', [False, True])
def test_provider_failure_never_exposes_exception_body_or_result_errors(monkeypatch, tmp_path, returned_error):
    _, index = source_paths(tmp_path)
    if returned_error:
        fake_factory(monkeypatch, answer=lambda question: AnswerResult(question=question, profile='hybrid',
            status='error', errors=['Authorization: secret-key; private response body']))
    else:
        fake_factory(monkeypatch, error=RuntimeError('Authorization: secret-key; private response body'))
    app = start_app(monkeypatch, '--index', index)
    app.text_area[0].input('Domanda')
    app.button[0].click().run()
    assert not app.exception and app.error
    assert all('secret-key' not in item.value and 'private response' not in item.value for item in app.error)


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'https://[invalid'])
def test_unsafe_source_link_is_not_rendered(monkeypatch, tmp_path, url):
    _, index = source_paths(tmp_path)
    fake_factory(monkeypatch, answer=lambda question: result(question, source_url=url))
    app = start_app(monkeypatch, '--index', index)
    app.text_area[0].input('Domanda')
    app.button[0].click().run()
    assert not app.exception and not app.get('link_button')
    assert any('collegamento' in warning.value for warning in app.warning)


def test_no_rag_without_corpus_has_explicit_unknown_snapshot(monkeypatch):
    calls, _ = fake_factory(monkeypatch)
    app = start_app(monkeypatch, '--profile', 'no_rag')
    assert not app.exception and not calls
    assert any('Snapshot: non specificato' in caption.value for caption in app.caption)
    assert any('senza consultazione' in item.value for item in app.info)

from pathlib import Path

import pytest

from legal_rag.artifacts import read_json
from legal_rag.corpus import acquisition as a


def catalogue(source_id='11', page=1, last=1):
    law = f'<a href="dettaglio?pk_lr={source_id}&amp;versione=V">Legge regionale 1 gennaio 2026, n. {source_id}</a>'
    return law + f'<a href="risultatiricerca?pagina={last}">last</a>'


def detail(source_id='11'):
    return f'<h1 id="tmpl-banche-dati-titolo-secondario">Legge regionale 1 gennaio 2026, n. {source_id}</h1>'


class FakeDownloader:
    calls = []
    fail = False
    drift = False

    def __init__(self, config):
        self.config = config
        self.session = type('Session', (), {'close': lambda self: None})()

    def get(self, url):
        self.calls.append(url)
        if 'risultatiricerca' in url:
            value = catalogue('12' if self.drift and self.calls.count(url) > 1 else '11')
        else:
            if self.fail:
                raise RuntimeError('interrupted')
            value = detail()
        return type('Response', (), {'content': value.encode(), 'encoding': 'utf-8'})()


def test_charset_mismatch_keeps_curly_apostrophe():
    value, encoding = a.decode_source(b'<p>l\x92articolo</p>', 'iso-8859-1')
    assert value == '<p>l’articolo</p>' and encoding == 'windows-1252'
    with pytest.raises(ValueError):
        a.decode_source('bad\u0085text'.encode(), 'utf-8')


def test_pagination_and_foreign_links():
    entries, last = a.parse_catalogue(catalogue(last=11) + '<a href="https://bad.invalid/dettaglio?pk_lr=12">Legge regionale wrong</a>')
    assert list(entries) == ['11'] and last == 11
    with pytest.raises(ValueError):
        a.parse_catalogue('<p>No results</p>')


def test_identity_validation_rejects_another_law():
    a.validate_detail(detail(), 'Legge regionale 1 gennaio 2026, n. 11')
    with pytest.raises(ValueError):
        a.validate_detail(detail('12'), 'Legge regionale 1 gennaio 2026, n. 11')
    with pytest.raises(ValueError):
        a.validate_detail(detail('111'), 'Legge regionale 1 gennaio 2026, n. 11')


def test_reported_pagination_beyond_visible_navigation():
    _, last = a.parse_catalogue(catalogue(last=2) + '<p>1 di 318 (Pagina corrente)</p>')
    assert last == 318


def test_resume_and_immutable_snapshot(tmp_path, monkeypatch):
    FakeDownloader.calls, FakeDownloader.fail, FakeDownloader.drift = [], True, False
    monkeypatch.setattr(a, '_Downloader', FakeDownloader)
    root = tmp_path / 'snapshot'
    with pytest.raises(RuntimeError):
        a.acquire_snapshot(root)
    assert read_json(root / 'manifest.json')['complete'] is False
    FakeDownloader.fail = False
    manifest = a.acquire_snapshot(root, resume=True)
    assert manifest['complete'] and manifest['source_count'] == 1
    assert manifest['current_completeness_verified'] is True
    assert len([url for url in FakeDownloader.calls if 'risultatiricerca' in url]) == 2
    with pytest.raises(FileExistsError):
        a.acquire_snapshot(root, resume=True)


def test_catalogue_drift_never_promotes(tmp_path, monkeypatch):
    FakeDownloader.calls, FakeDownloader.fail, FakeDownloader.drift = [], False, True
    monkeypatch.setattr(a, '_Downloader', FakeDownloader)
    root = tmp_path / 'snapshot'
    with pytest.raises(ValueError, match='changed'):
        a.acquire_snapshot(root)
    assert not read_json(root / 'manifest.json')['complete']


def test_resume_refetches_final_catalogue_instead_of_trusting_checkpoint(tmp_path, monkeypatch):
    FakeDownloader.calls, FakeDownloader.fail, FakeDownloader.drift = [], False, True
    monkeypatch.setattr(a, '_Downloader', FakeDownloader)
    root = tmp_path / 'snapshot'
    with pytest.raises(ValueError):
        a.acquire_snapshot(root)
    FakeDownloader.drift = False
    manifest = a.acquire_snapshot(root, resume=True)
    assert manifest['complete']
    assert len([url for url in FakeDownloader.calls if 'risultatiricerca' in url]) == 3


def test_legacy_preserves_bytes_and_records_unknown_date(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    raw = 'bad\u0085text'.encode()
    (source / '11_law.html').write_bytes(raw)
    target = tmp_path / 'legacy'
    result = a.import_legacy(source, target)
    assert (target / 'sources/11_law.html').read_bytes() == raw
    assert result['acquisition_started_at'] is None and result['legacy']
    assert result['encoding_anomalies'] == ['11_law.html']
    assert not result['current_completeness_verified']

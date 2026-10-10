import json
from pathlib import Path

import pytest

from legal_rag.artifacts import file_hash, write_json, write_jsonl
from legal_rag.evaluation import load_benchmark, run_evaluation, run_retrieval_diagnostics
from legal_rag.reporting import ReportComment, build_report, load_comments, write_comments
from test_evaluation import FakeCounter, FakeEngine, FakeJudge, FakeRetriever, identity, sources


def corpus(path: Path) -> Path:
    path.mkdir()
    write_json(path / "manifest.json", {"snapshot_id": "s", "counts": {"documents": 1}})
    write_jsonl(path / "documents.jsonl", [{"document_id": "vda:lr:2000-01-25:5", "title": '<script>alert("x")</script>',
        "status": "unknown", "availability": "body", "source_url": 'javascript:alert("x")', "api_key": "MUST_NOT_EXPORT"}])
    write_jsonl(path / "units.jsonl", [{"unit_id": "u1", "document_id": "vda:lr:2000-01-25:5", "kind": "article", "label": "12", "text": "<img src=x onerror=alert(1)>", "spans": []}])
    return path


def test_corpus_only_report_escapes_source_and_does_not_leak_secrets(tmp_path):
    source = corpus(tmp_path / "corpus")
    page = build_report(source, [], tmp_path / "report")
    assert page.exists()
    documents = list((page.parent / "corpus").glob("*.html"))
    assert len(documents) == 1
    text = documents[0].read_text()
    assert '<script>alert("x")</script>' not in text
    assert "&lt;script&gt;" in text
    assert "href=\"javascript:" not in text
    assert "<img src=x" not in text
    assert "MUST_NOT_EXPORT" not in text
    assert "https://cdn" not in text
    assert "no_cessation_evidence" in (page.parent / "corpus.html").read_text()
    manifest = json.loads((page.parent / "manifest.json").read_text())
    assert "corpus.html" in manifest["output_hashes"]
    with pytest.raises(FileExistsError):
        build_report(source, [], page.parent)


def test_report_preserves_failed_judgments_and_reuses_only_saved_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("legal_rag.evaluation.runner.code_identity", lambda: {"source_sha256": "test"})
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    engine, judge = FakeEngine(), FakeJudge(fail=True)
    run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity())
    before = len(engine.calls), len(judge.calls)
    page = build_report(None, [tmp_path / "run"], tmp_path / "report")
    assert before == (len(engine.calls), len(judge.calls))
    assert "incomplete" in page.read_text()
    experiment = (page.parent / "experiments/run-1.html").read_text()
    assert "missing_scores" in experiment and "eval-0001" in experiment
    detail = (page.parent / "questions/run-1-eval-0001-no_hint.html").read_text()
    assert "RuntimeError" in detail and "Contesto realmente inviato" in detail
    assert "Filtra" in experiment


def test_report_does_not_claim_comparable_results_with_different_models(tmp_path, monkeypatch):
    monkeypatch.setattr("legal_rag.evaluation.runner.code_identity", lambda: {"source_sha256": "test"})
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "a", identity=identity())
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "b", identity=identity(generator={"provider": "utopia", "model": "other"}))
    page = build_report(None, [tmp_path / "a", tmp_path / "b"], tmp_path / "report")
    assert "Confronto non compatibile" in page.read_text()
    assert "incompatible_identity" in page.read_text()


def test_single_run_report_discloses_artifact_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr("legal_rag.evaluation.runner.code_identity", lambda: {"source_sha256": "test"})
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "run", identity=identity())
    metrics = tmp_path / "run/metrics.json"
    metrics.write_text(metrics.read_text() + "\n", encoding="utf-8")
    page = build_report(None, [tmp_path / "run"], tmp_path / "report")
    assert "artifact_hash_mismatch:metrics.json" in page.read_text()


def test_retrieval_only_report_has_separate_proxy_pages_without_generation(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    retriever = FakeRetriever()
    run_retrieval_diagnostics(benchmark, retriever, FakeCounter(), tmp_path / "diagnostic", profiles=("hybrid", "graph"))
    calls = len(retriever.calls)
    page = build_report(None, [], tmp_path / "report", diagnostic_dirs=[tmp_path / "diagnostic"])
    assert len(retriever.calls) == calls
    assert "Diagnosi del retrieval" in page.read_text()
    assert "nessuno score di correttezza" in page.read_text()
    profile = (page.parent / "diagnostics/diagnostic-1-graph.html").read_text()
    assert "candidate_pool_article_success" in profile and "context_article_success" in profile
    detail = (page.parent / "questions/diagnostic-1-graph-eval-0001.html").read_text()
    assert "Pool originale e selezione finale" in detail and "relation-1" in detail


def test_stopped_run_report_keeps_unvisited_qids_missing(tmp_path):
    from legal_rag.providers.errors import ProviderError

    benchmark = load_benchmark(sources(tmp_path / "sources", count=3), expected_count=3)

    class ForbiddenEngine(FakeEngine):
        def answer(self, *args, **kwargs):
            raise ProviderError("http_403", "Forbidden")

    run_evaluation(benchmark, ForbiddenEngine(), FakeJudge(), tmp_path / "run", identity=identity())
    page = build_report(None, [tmp_path / "run"], tmp_path / "report")
    experiment = (page.parent / "experiments/run-1.html").read_text()
    assert "eval-0003" in experiment and "not_evaluated" in experiment
    detail = (page.parent / "questions/run-1-eval-0003-no_hint.html").read_text()
    assert "Quanto dura?" in detail and "Nessun checkpoint salvato" in detail
    assert "limiti inferiori" in page.read_text()


def test_comments_persist_separately_and_respect_run_scope(tmp_path):
    source = corpus(tmp_path / "corpus")
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    first = run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "first", profile="dense", identity=identity())
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "second", profile="hybrid", identity=identity())
    comments_path = write_comments(tmp_path / "notes.jsonl", [
        ReportComment(comment_id="right", author="Researcher", text="Scoped interpretation", qid="eval-0001", run_hash=first["run_hash"]),
        ReportComment(comment_id="wrong", author="Researcher", text="Different run interpretation", qid="eval-0001", run_hash="different-run"),
        ReportComment(comment_id="document", author="Researcher", text="Document interpretation", document_id="vda:lr:2000-01-25:5"),
        ReportComment(comment_id="escaped", author="<svg onload=alert(1)>", text="<script>COMMENT_PAYLOAD</script>")])
    before = comments_path.read_bytes()
    page = build_report(source, [tmp_path / "first", tmp_path / "second"], tmp_path / "report", comments_path=comments_path)
    assert comments_path.read_bytes() == before and len(load_comments(comments_path)) == 4
    home = page.read_text()
    assert "Different run interpretation" in home and "different-run" in home
    assert "<script>COMMENT_PAYLOAD</script>" not in home and "&lt;script&gt;COMMENT_PAYLOAD&lt;/script&gt;" in home
    assert "<svg onload=" not in home
    first_page = (page.parent / "questions/run-1-eval-0001-no_hint.html").read_text()
    second_page = (page.parent / "questions/run-2-eval-0001-no_hint.html").read_text()
    assert "Scoped interpretation" in first_page and "Different run interpretation" not in first_page
    assert "Scoped interpretation" not in second_page and "Different run interpretation" not in second_page
    document_page = next((page.parent / "corpus").glob("*.html")).read_text()
    assert "Document interpretation" in document_page
    manifest = json.loads((page.parent / "manifest.json").read_text())
    assert manifest["comments_path"] == str(comments_path)
    assert manifest["input_hashes"][str(comments_path)] == file_hash(comments_path)
    assert "Scoped interpretation" not in json.dumps(manifest) and "COMMENT_PAYLOAD" not in json.dumps(manifest)
    build_report(source, [], tmp_path / "second-report", comments_path=comments_path)
    assert comments_path.read_bytes() == before


def test_comment_writes_require_explicit_append_and_unique_ids(tmp_path):
    path = write_comments(tmp_path / "notes.jsonl", [ReportComment(comment_id="one", author="Researcher", text="First interpretation")])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_comments(path, [ReportComment(comment_id="two", author="Researcher", text="Second interpretation")])
    with pytest.raises(ValueError, match="unique"):
        write_comments(path, [ReportComment(comment_id="one", author="Researcher", text="Duplicate")], append=True)
    assert path.read_bytes() == before
    write_comments(path, [ReportComment(comment_id="two", author="Researcher", text="Second interpretation")], append=True)
    assert path.read_bytes().startswith(before)
    assert [comment.comment_id for comment in load_comments(path)] == ["one", "two"]


def test_answers_are_side_by_side_with_pending_checkpoint_and_compatibility(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "dense", profile="dense", identity=identity())

    class InterruptedJudge(FakeJudge):
        def complete(self, *args):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_evaluation(benchmark, FakeEngine(wrong=True), InterruptedJudge(), tmp_path / "hybrid", profile="hybrid", identity=identity())
    page = build_report(None, [tmp_path / "dense", tmp_path / "hybrid"], tmp_path / "report")
    assert "Risposte affiancate per domanda" in page.read_text() and 'data-filter="answer-comparisons"' in page.read_text()
    detail = (page.parent / "questions/compare-eval-0001-no_hint.html").read_text()
    assert "Confronto descrittivo" in detail and "incomplete_run" in detail
    assert "Presidente e Consiglio." in detail and "wrong" in detail and "Semantic match" in detail
    assert 'run-1-eval-0001-no_hint.html' in detail and 'run-2-eval-0001-no_hint.html' in detail
    assert "&quot;score&quot;: 2" in detail and "&quot;score&quot;: null" in detail
    missing = (page.parent / "questions/compare-eval-0002-no_hint.html").read_text()
    assert "not_evaluated" in missing and "Nessun checkpoint di generazione salvato" in missing


def test_retrieval_profiles_are_side_by_side_without_answer_scores(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    run_retrieval_diagnostics(benchmark, FakeRetriever(), FakeCounter(), tmp_path / "diagnostic", profiles=("hybrid", "graph"))
    page = build_report(None, [], tmp_path / "report", diagnostic_dirs=[tmp_path / "diagnostic"])
    assert "Retrieval affiancato per domanda" in page.read_text() and 'data-filter="retrieval-comparisons"' in page.read_text()
    detail = (page.parent / "questions/compare-retrieval-eval-0001.html").read_text()
    assert "candidate_pool_article_success" in detail and "context_article_success" in detail
    assert '<h2>hybrid</h2>' in detail and '<h2>graph</h2>' in detail
    assert "diagnostic-1-hybrid-eval-0001.html" in detail and "diagnostic-1-graph-eval-0001.html" in detail
    assert "&quot;score&quot;" not in detail and "Score e giudici" not in detail


@pytest.mark.parametrize('mismatch', [None, 'snapshot_id', 'manifest_sha256'])
def test_snapshot_date_is_visible_only_for_exact_reconciled_corpus(tmp_path, mismatch):
    from lxml import html
    from legal_rag.providers.errors import ProviderError

    source = corpus(tmp_path / 'corpus')
    snapshot_date = '2026-10-10T13:26:27+00:00'
    write_json(source / 'manifest.json', {'snapshot_id': 's', 'snapshot_date': snapshot_date, 'counts': {'documents': 1}})
    frozen = {'snapshot_id': 's', 'manifest_sha256': file_hash(source / 'manifest.json')}
    if mismatch:
        frozen[mismatch] = 'historical-snapshot' if mismatch == 'snapshot_id' else 'f' * 64
    benchmark = load_benchmark(sources(tmp_path / 'sources'), expected_count=2)
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / 'complete', identity=identity(corpus=frozen))

    class ForbiddenEngine(FakeEngine):
        def answer(self, *args, **kwargs):
            raise ProviderError('http_403', 'Forbidden')

    run_evaluation(benchmark, ForbiddenEngine(), FakeJudge(), tmp_path / 'pending', identity=identity(corpus=frozen))
    retriever = FakeRetriever()
    retriever.identity = {'snapshot_id': frozen['snapshot_id'], 'corpus': frozen, 'index_identity': 'fake'}
    run_retrieval_diagnostics(benchmark, retriever, FakeCounter(), tmp_path / 'diagnostic', profiles=('dense', 'graph'))
    page = build_report(source, [tmp_path / 'complete', tmp_path / 'pending'], tmp_path / 'report', diagnostic_dirs=[tmp_path / 'diagnostic'])
    expected_date = 'data non disponibile' if mismatch else snapshot_date
    for path in [*page.parent.joinpath('questions').glob('*.html'), *page.parent.joinpath('experiments').glob('*.html'),
                 *page.parent.joinpath('diagnostics').glob('*.html')]:
        labels = html.fromstring(path.read_text()).xpath('//p[@class="snapshot"]')
        assert labels, path.name
        for label in labels:
            text = label.text_content()
            assert frozen['snapshot_id'] in text and expected_date in text, path.name
            if mismatch:
                assert snapshot_date not in text, path.name
    home = html.fromstring(page.read_text())
    for label in home.xpath('//table//p[@class="snapshot"]'):
        assert frozen['snapshot_id'] in label.text_content() and expected_date in label.text_content()
    for path in [page.parent / 'corpus.html', *page.parent.joinpath('corpus').glob('*.html')]:
        label = html.fromstring(path.read_text()).xpath('//header/p[@class="snapshot"]')[0]
        assert 'Snapshot: s' in label.text_content() and snapshot_date in label.text_content()

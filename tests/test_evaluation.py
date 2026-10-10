import csv
import json
from pathlib import Path

import pytest

from legal_rag.artifacts import file_hash, write_json, write_jsonl
from legal_rag.contracts import AnswerResult, Candidate, Chunk, Citation, Context, Evidence, Usage
from legal_rag.evaluation import compare_runs, load_benchmark, load_diagnostics, load_results, run_evaluation, run_rejudge, run_retrieval_diagnostics, select_control_qids, write_review_proposals
from legal_rag.evaluation.models import Review

REVIEW_PROVENANCE = {"note": "Synthetic fixture: the cited source supports this review decision.",
                     "source_urls": ["https://example.test/primary-source"], "evidence": "Synthetic supporting passage for a contract test."}


def sources(path: Path, count: int = 2) -> Path:
    path.mkdir()
    headers = ["Domanda", "Livello", "Risposta corretta", "Riferimento legge per la risposta"]
    questions = [("Chi decide?", "Consiglio", "L1"), ("Quali organi?", "Presidente", "L2"), ("Quanto dura?", "Un anno", "L3")][:count]
    with (path / "questions_no_hint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for question, answer, level in questions:
            writer.writerow([question, level, answer, "Legge regionale 25 gennaio 2000, n. 5 - Art. 12"])
    with (path / "questions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for question, answer, level in questions:
            choices = "\n".join(f"{label}) {answer if label == 'A' else label}" for label in "ABCDEF")
            writer.writerow([question + "\n" + choices, level, "A", "Legge regionale 25 gennaio 2000, n. 5 - Art. 12"])
    return path


def identity(**updates):
    return {"generator": {"provider": "utopia", "model": "answer"}, "judge": {"provider": "openai", "model": "judge"},
            "answer_prompt_version": "test-v1", "filters": {"exclude_ceased": True}, "token_budget": 4096,
            "candidate_budget": 100, "corpus": "snapshot-hash", "embedding": "bge-revision", **updates}


class FakeEngine:
    def __init__(self, *, fail=False, bad_quote=False, wrong=False):
        self.calls, self.fail, self.bad_quote, self.wrong = [], fail, bad_quote, wrong

    def answer(self, question, *, profile, options=None):
        self.calls.append((question, profile, options))
        if self.fail:
            raise RuntimeError("A sensitive provider body must not enter the artifact")
        text = "Presidente e Consiglio."
        chunk = Chunk(chunk_id="c1", document_id="vda:lr:2000-01-25:5", snapshot_id="s", article_label="12",
                      title="Organizzazione", text=text, embedding_text=text, unit_ids=["u1"], spans=[], source_url="https://example.org/law")
        context = Context() if profile == "no_rag" else Context(text=text, token_count=5,
                    evidence=[Evidence(context_id="E1", chunk=chunk, rendered=text)])
        citations = [] if profile == "no_rag" else [Citation(context_id="E1", chunk_id="c1", quote="Inventato" if self.bad_quote else "Consiglio")]
        return AnswerResult(question=question, profile=profile, snapshot_id="s", status="answered", answer="wrong" if self.wrong else text,
                            selected_option="A", context=context, citations=citations,
                            candidates=[] if profile == "no_rag" else [Candidate(chunk=chunk, rank=1, score=.9, origin=profile)],
                            usage=[Usage(provider="utopia", requested_model="answer", returned_model="answer")])


class FakeJudge:
    identity = {"provider": "openai", "model": "judge"}

    def __init__(self, *, fail=False):
        self.calls, self.fail = [], fail

    def complete(self, prompt, schema):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("secret=do-not-export")
        data = json.loads(prompt.split("\n")[-1])
        assert "options" not in data
        return schema(correctness=0 if data["candidate_answer"] == "wrong" else 2, explanation="Semantic match",
                      grounding="supported" if data["grounding_applicable"] else "not_applicable", grounding_explanation="Evidence checked"), Usage(
                          provider="openai", requested_model="judge", returned_model="judge", cost_eur=.001)


class FakeControlJudge(FakeJudge):
    identity = {"provider": "openai", "model": "control"}

    def complete(self, *args):
        payload, usage = super().complete(*args)
        return payload, usage.model_copy(update={"requested_model": "control", "returned_model": "control"})


@pytest.fixture(autouse=True)
def fixed_code_identity(monkeypatch):
    monkeypatch.setattr("legal_rag.evaluation.runner.code_identity", lambda: {"commit": "test", "source_sha256": "test", "dirty": False})


def test_original_100_intents_align_and_references_remain_diagnostic():
    source = Path(__file__).resolve().parents[1] / "data/evaluation"
    original = {path.name: path.read_bytes() for path in source.glob("questions*.csv")}
    benchmark = load_benchmark(source)
    assert len(benchmark.questions) == 100
    assert len({question.qid for question in benchmark.questions}) == 100
    assert sum(len(question.expected_references) for question in benchmark.questions) == 106
    assert benchmark.questions[4].expected_references[0].article_label == "4bis"
    assert "documented_reference_mismatch" in benchmark.questions[12].flags
    assert original == {path.name: path.read_bytes() for path in source.glob("questions*.csv")}


def test_alignment_rejects_wrong_gold(tmp_path):
    source = sources(tmp_path / "sources")
    path = source / "questions_no_hint.csv"
    path.write_text(path.read_text().replace("Consiglio", "Sbagliato"), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_benchmark(source, expected_count=2)


def test_only_attributable_approved_reviews_enter_current_view(tmp_path):
    source = sources(tmp_path / "sources")
    benchmark = load_benchmark(source, expected_count=2)
    review = write_review_proposals(benchmark, [{"qid": "eval-0001", "correct_answer": "New answer"}], tmp_path / "proposals.jsonl")
    assert load_benchmark(source, expected_count=2, review_path=review).questions[0].correct_answer == "Consiglio"
    with pytest.raises(ValueError, match="no approved"):
        load_benchmark(source, expected_count=2, review_path=review, view="current_approved")
    with pytest.raises(ValueError, match="reviewer"):
        Review(qid="eval-0001", status="current_revised", human_approved=True)
    approved = write_review_proposals(benchmark, [{"qid": "eval-0001", "status": "current_revised", "correct_answer": "New answer",
                "human_approved": True, "reviewer": "Researcher", **REVIEW_PROVENANCE}, {"qid": "eval-0002", "status": "historical_only", "human_approved": True, "reviewer": "Researcher", **REVIEW_PROVENANCE}], tmp_path / "approved.jsonl")
    current = load_benchmark(source, expected_count=2, view="current_approved", review_path=approved)
    assert len(current.questions) == 1
    assert current.questions[0].correct_label is None
    assert benchmark.questions[0].correct_answer == "Consiglio"


def test_failed_judge_resume_reuses_answer_and_blocks_complete_comparison(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    engine, judge = FakeEngine(), FakeJudge(fail=True)
    manifest = run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity())
    assert manifest["complete"] is False
    assert all(row.score is None for row in load_results(tmp_path / "run"))
    assert "secret=do-not-export" not in (tmp_path / "run/results.jsonl").read_text()
    judge.fail = False
    manifest = run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity(), resume=True)
    assert manifest["complete"] is True
    assert len(engine.calls) == 2
    assert len(load_results(tmp_path / "run")) == 2
    with pytest.raises(FileExistsError):
        run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity())
    with pytest.raises(ValueError, match="identities"):
        run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity(token_budget=2048), resume=True)


def test_generation_failure_is_zero_and_not_a_judge_failure(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    judge = FakeJudge()
    run_evaluation(benchmark, FakeEngine(fail=True), judge, tmp_path / "run", identity=identity())
    rows = load_results(tmp_path / "run")
    assert all(row.score == 0 and row.answer.status == "error" for row in rows)
    assert not judge.calls
    assert "sensitive provider" not in (tmp_path / "run/results.jsonl").read_text()


def test_citations_and_reference_proxies_are_separate_from_correctness(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    engine = FakeEngine(bad_quote=True)
    run_evaluation(benchmark, engine, FakeJudge(), tmp_path / "run", identity=identity())
    row = load_results(tmp_path / "run")[0]
    assert row.score == 2 and row.citation_checks[0]["valid"] is False
    assert row.retrieval_proxies["article_success"]["5"] is True
    assert row.retrieval_proxies["article_mrr"]["100"] == 1
    assert all(set(call[2] or {}) <= set("ABCDEF") for call in engine.calls)
    assert all(isinstance(call[0], str) for call in engine.calls)


def test_comparisons_reject_different_budgets_and_compute_paired_effect(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    run_evaluation(benchmark, FakeEngine(wrong=True), FakeJudge(), tmp_path / "dense", profile="dense", identity=identity(), judge_control=FakeControlJudge())
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "hybrid", profile="hybrid", identity=identity(), judge_control=FakeControlJudge())
    compared = compare_runs([tmp_path / "dense", tmp_path / "hybrid"], resamples=100)["comparisons"][0]
    assert compared["comparable"] is True
    values = compared["formats"]["no_hint"]
    assert values["fully_correct"]["delta"] == 1
    assert values["law_cluster_sensitivity"]["clusters"] == 1
    assert values["promoted"] is True
    run_evaluation(benchmark, FakeEngine(), FakeJudge(), tmp_path / "different", profile="hybrid", identity=identity(token_budget=1024))
    compared = compare_runs([tmp_path / "dense", tmp_path / "different"], resamples=100)["comparisons"][0]
    assert compared["comparable"] is False
    assert "incompatible_identity" in compared["issues"]


def test_control_sample_is_stable_and_balanced():
    benchmark = load_benchmark(Path(__file__).resolve().parents[1] / "data/evaluation")
    selected = select_control_qids(benchmark.questions)
    assert selected == select_control_qids(list(reversed(benchmark.questions)))
    assert len(selected) == 20
    assert {level: sum(question.qid in selected for question in benchmark.questions if question.level == level)
            for level in ("L1", "L2", "L3", "L4")} == {level: 5 for level in ("L1", "L2", "L3", "L4")}


def test_control_failure_resume_retries_only_control(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    engine, judge, control = FakeEngine(), FakeJudge(), FakeJudge(fail=True)
    manifest = run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity(), judge_control=control)
    assert manifest["complete"] and not manifest["control_complete"]
    control.fail = False
    manifest = run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity(), judge_control=control, resume=True)
    assert manifest["control_complete"]
    assert len(engine.calls) == len(judge.calls) == 2
    assert len(control.calls) == 4


def test_mcq_is_exact_and_invalid_labels_are_generation_errors(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class InvalidLabelEngine(FakeEngine):
        def answer(self, *args, **kwargs):
            answer = super().answer(*args, **kwargs)
            answer.selected_option = "invalid"
            return answer

    judge = FakeJudge()
    run_evaluation(benchmark, InvalidLabelEngine(), judge, tmp_path / "run", identity=identity(), formats=("mcq",))
    rows = load_results(tmp_path / "run")
    assert all(row.score == 0 and "invalid_mcq_label" in row.flags for row in rows)
    assert not judge.calls
    assert json.loads((tmp_path / "run/metrics.json").read_text())["mcq"]["generation_errors"] == 2


def test_oracle_uses_only_approved_chunks_from_exact_corpus(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    chunk = FakeEngine().answer("Question", profile="hybrid").candidates[0].chunk
    write_jsonl(corpus_dir / "chunks.jsonl", [chunk])
    write_json(corpus_dir / "manifest.json", {"snapshot_id": "s", "files": {"chunks.jsonl": file_hash(corpus_dir / "chunks.jsonl")}})
    evidence = {"qid": "eval-0001", "snapshot_id": "s", "corpus_manifest_sha256": file_hash(corpus_dir / "manifest.json"),
                "chunk_ids": [chunk.chunk_id], "human_approved": True, "reviewer": "Researcher"}
    write_jsonl(tmp_path / "oracle.jsonl", [evidence])

    class OracleEngine(FakeEngine):
        def answer(self, question, *, profile, options=None, oracle_chunks=None):
            assert oracle_chunks == [chunk]
            assert question == "Chi decide?"
            return super().answer(question, profile=profile, options=options)

    manifest = run_evaluation(benchmark, OracleEngine(), FakeJudge(), tmp_path / "run", identity=identity(corpus={"snapshot_id": "s"}),
        profile="oracle", qids=["eval-0001"], corpus_dir=corpus_dir, oracle_evidence_path=tmp_path / "oracle.jsonl")
    assert manifest["inputs"]["oracle_policy"]["human_verified"]
    assert "oracle_policy" not in manifest["inputs"]["identity"]
    write_jsonl(tmp_path / "unapproved.jsonl", [{**evidence, "human_approved": False}])
    with pytest.raises(ValueError, match="human approval"):
        run_evaluation(benchmark, OracleEngine(), FakeJudge(), tmp_path / "bad", identity=identity(corpus={"snapshot_id": "s"}),
            profile="oracle", qids=["eval-0001"], corpus_dir=corpus_dir, oracle_evidence_path=tmp_path / "unapproved.jsonl")
    write_jsonl(corpus_dir / "chunks.jsonl", [chunk.model_copy(update={"text": "modified"})])
    with pytest.raises(ValueError, match="manifest hash"):
        run_evaluation(benchmark, OracleEngine(), FakeJudge(), tmp_path / "tampered", identity=identity(corpus={"snapshot_id": "s"}),
            profile="oracle", qids=["eval-0001"], corpus_dir=corpus_dir, oracle_evidence_path=tmp_path / "oracle.jsonl")


def test_current_snapshot_gate_and_frozen_gold_rejudge(tmp_path):
    source = sources(tmp_path / "sources")
    benchmark = load_benchmark(source, expected_count=2)
    engine = FakeEngine()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    write_jsonl(corpus_dir / "chunks.jsonl", [FakeEngine().answer("Fixture", profile="hybrid").candidates[0].chunk])
    write_json(corpus_dir / "manifest.json", {"snapshot_id": "s", "ready_for_indexing": True,
        "files": {"chunks.jsonl": file_hash(corpus_dir / "chunks.jsonl")}})
    run_identity = identity(corpus={"snapshot_id": "s", "manifest_sha256": file_hash(corpus_dir / "manifest.json")})
    run_evaluation(benchmark, engine, FakeJudge(), tmp_path / "original", identity=run_identity)
    review = write_review_proposals(benchmark, [{"qid": "eval-0001", "status": "current_revised", "correct_answer": "Corrected gold",
        "human_approved": True, "reviewer": "Researcher", "snapshot_id": "s", **REVIEW_PROVENANCE}], tmp_path / "approved.jsonl")
    current = load_benchmark(source, expected_count=2, view="current_approved", review_path=review)
    with pytest.raises(ValueError, match="snapshot"):
        run_evaluation(current, engine, FakeJudge(), tmp_path / "wrong-snapshot", identity=identity(corpus={"snapshot_id": "other"}))
    manifest = run_rejudge(current, FakeJudge(), tmp_path / "original", tmp_path / "corrected", identity=run_identity, corpus_dir=corpus_dir)
    row = load_results(tmp_path / "corrected")[0]
    assert row.correct_answer == "Corrected gold" and not row.answer.usage
    assert len(engine.calls) == 2
    assert load_results(tmp_path / "original")[0].correct_answer == "Consiglio"
    assert manifest["inputs"]["frozen_generation"]["source_run_hash"]
    revised = current.model_copy(deep=True)
    revised.questions[0].question = "Changed question"
    with pytest.raises(ValueError, match="changed question"):
        run_rejudge(revised, FakeJudge(), tmp_path / "original", tmp_path / "changed", identity=run_identity, corpus_dir=corpus_dir)
    judge = FakeJudge()
    other = tmp_path / "other-corpus"
    other.mkdir()
    write_json(other / "manifest.json", {"snapshot_id": "different", "ready_for_indexing": True, "files": {}})
    with pytest.raises(ValueError, match="corpus identity"):
        run_rejudge(current, judge, tmp_path / "original", tmp_path / "wrong-corpus", identity=run_identity, corpus_dir=other)
    (corpus_dir / "chunks.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_rejudge(current, judge, tmp_path / "original", tmp_path / "tampered-corpus", identity=run_identity, corpus_dir=corpus_dir)
    assert not judge.calls and not (tmp_path / "wrong-corpus").exists() and not (tmp_path / "tampered-corpus").exists()


def test_correctness_does_not_imply_grounding_and_invalid_judge_usage_is_preserved(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class UnsupportedJudge(FakeJudge):
        def complete(self, *args):
            payload, usage = super().complete(*args)
            return payload.model_copy(update={"grounding": "unsupported"}), usage

    run_evaluation(benchmark, FakeEngine(), UnsupportedJudge(), tmp_path / "unsupported", identity=identity())
    metrics = json.loads((tmp_path / "unsupported/metrics.json").read_text())["no_hint"]
    assert metrics["fully_correct_rate"] == 1 and metrics["correct_and_grounded_rate"] == 0

    class InvalidGroundingJudge(FakeJudge):
        def complete(self, *args):
            payload, usage = super().complete(*args)
            return payload.model_copy(update={"grounding": "not_applicable"}), usage

    run_evaluation(benchmark, FakeEngine(), InvalidGroundingJudge(), tmp_path / "invalid", identity=identity())
    row = load_results(tmp_path / "invalid")[0]
    assert row.score is None and len(row.judge_usage) == 1


def test_empty_no_hint_output_is_zero_without_a_judge_call(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class EmptyEngine(FakeEngine):
        def answer(self, *args, **kwargs):
            answer = super().answer(*args, **kwargs)
            answer.answer = "  "
            return answer

    judge = FakeJudge()
    run_evaluation(benchmark, EmptyEngine(), judge, tmp_path / "run", identity=identity())
    assert all(row.score == 0 and "empty_generated_answer" in row.flags for row in load_results(tmp_path / "run"))
    assert not judge.calls


def test_resume_rejects_modified_saved_answers(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    engine, judge = FakeEngine(), FakeJudge()
    run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity())
    path = tmp_path / "run/answers.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity(), resume=True)
    assert len(engine.calls) == 2


class FakeCounter:
    name, revision = "fake-counter", "v1"

    def count(self, text):
        return len(text.split())


class FakeRetriever:
    identity = {"snapshot_id": "s", "index_identity": "fake"}
    manifest = {"snapshot_id": "s"}

    def __init__(self, *, fail_mode=None):
        self.calls, self.fail_mode = [], fail_mode
        self.target = FakeEngine().answer("Question", profile="hybrid").candidates[0].chunk
        self.other = self.target.model_copy(update={"chunk_id": "other", "article_label": "13"})

    def search(self, question, *, mode, config, trace):
        self.calls.append((question, mode))
        if mode == self.fail_mode:
            raise RuntimeError("private provider detail")
        chunk = self.target if mode == "sparse" else self.other
        trace.append({"query": question, "mode": mode})
        return [Candidate(chunk=chunk, rank=1, score=.8, origin=mode)]

    def expand_graph(self, candidates, *, config):
        return [Candidate(chunk=self.target, rank=2, score=0, origin="graph")], ["relation-1"]


def test_retrieval_diagnostics_keep_pool_selection_and_context_separate(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    retriever = FakeRetriever()
    manifest = run_retrieval_diagnostics(benchmark, retriever, FakeCounter(), tmp_path / "diagnostic")
    rows = load_diagnostics(tmp_path / "diagnostic")
    graph = next(row for row in rows if row.answer.profile == "graph")
    assert graph.retrieval_proxies["candidate_pool_article_success"]["100"] is False
    assert graph.retrieval_proxies["article_success"]["100"] is True
    assert graph.retrieval_proxies["context_article_success"] is True
    assert graph.answer.retrieval_trace["graph_relation_ids"] == ["relation-1"]
    assert all(row.retrieval_only and not row.answer.answer and not row.answer.usage for row in rows)
    assert all(question in {"Chi decide?", "Quali organi?"} for question, _ in retriever.calls)
    assert len(rows) == 8 and manifest["errors"] == 0
    assert "correctness" not in (tmp_path / "diagnostic/metrics.json").read_text().replace("not correctness", "")
    with pytest.raises(FileExistsError):
        run_retrieval_diagnostics(benchmark, retriever, FakeCounter(), tmp_path / "diagnostic")


def test_failed_retrieval_diagnostics_preserve_denominator_and_unavailable_proxies(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)
    run_retrieval_diagnostics(benchmark, FakeRetriever(fail_mode="dense"), FakeCounter(), tmp_path / "diagnostic", profiles=("dense",))
    metrics = json.loads((tmp_path / "diagnostic/metrics.json").read_text())["dense"]
    assert metrics["denominator"] == metrics["retrieval_errors"] == 2
    assert metrics["article_success"]["100"]["proxy_coverage"] == 0
    assert metrics["article_success"]["100"]["available_value"] is None
    assert "private provider detail" not in (tmp_path / "diagnostic/records.jsonl").read_text()


def test_promotion_requires_complete_control_and_complete_current_review(tmp_path):
    source = sources(tmp_path / "sources")
    benchmark = load_benchmark(source, expected_count=2)
    for name, wrong in (("dense", True), ("hybrid", False)):
        run_evaluation(benchmark, FakeEngine(wrong=wrong), FakeJudge(), tmp_path / name, identity=identity(), profile=name)
    pair = compare_runs([tmp_path / "dense", tmp_path / "hybrid"], resamples=100)["comparisons"][0]["formats"]["no_hint"]
    assert pair["fully_correct"]["delta"] == 1 and not pair["promoted"]
    assert not pair["promotion_gates"]["control_present"]
    run_identity = identity(corpus={"snapshot_id": "s"})
    approved = {"qid": "eval-0001", "status": "current_supported", "human_approved": True, "reviewer": "Researcher", "snapshot_id": "s", **REVIEW_PROVENANCE}
    for complete_review in (False, True):
        reviews = [approved]
        if complete_review:
            reviews.append({**approved, "qid": "eval-0002", "status": "historical_only"})
        review_path = write_review_proposals(benchmark, reviews, tmp_path / f"review-{complete_review}.jsonl")
        current = load_benchmark(source, expected_count=2, review_path=review_path, view="current_approved")
        paths = []
        for profile, wrong in (("dense", True), ("hybrid", False)):
            path = tmp_path / f"{complete_review}-{profile}"
            manifest = run_evaluation(current, FakeEngine(wrong=wrong), FakeJudge(), path, profile=profile,
                identity=run_identity, judge_control=FakeControlJudge())
            assert manifest["view_provisional"] is not complete_review
            assert manifest["original_intents"] == 2 and manifest["view_intents"] == manifest["selected_intents"] == 1
            paths.append(path)
        pair = compare_runs(paths, resamples=100)["comparisons"][0]["formats"]["no_hint"]
        assert pair["promoted"] is complete_review


def test_judge_budget_stop_leaves_missing_cases_and_resume_allows_new_accounting(tmp_path):
    from legal_rag.providers.errors import BudgetExceeded

    benchmark = load_benchmark(sources(tmp_path / "sources", count=3), expected_count=3)

    class BudgetJudge(FakeJudge):
        blocked = True
        accounting = {"price_date": "2026-10-10", "cap_eur": 20}

        def complete(self, prompt, schema):
            if self.blocked and len(self.calls) == 1:
                self.calls.append(prompt)
                raise BudgetExceeded()
            return super().complete(prompt, schema)

    engine, judge = FakeEngine(), BudgetJudge()
    manifest = run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity())
    assert not manifest["complete"] and manifest["stopped_reason"] == "judge:budget_exceeded"
    assert len(engine.calls) == len(judge.calls) == 2
    rows = load_results(tmp_path / "run")
    assert [row.score for row in rows] == [2, None]
    metrics = json.loads((tmp_path / "run/metrics.json").read_text())["no_hint"]
    assert metrics["denominator"] == 3 and metrics["missing_scores"] == 2
    assert metrics["fully_correct_rate"] == 1 / 3 and metrics["fully_correct_rate_status"] == "lower_bound_missing_scores"
    assert metrics["generation_errors"] == 0 and metrics["generation_not_attempted"] == 1
    judge.blocked, judge.accounting = False, {"price_date": "2026-10-11", "cap_eur": 20}
    resumed = run_evaluation(benchmark, engine, judge, tmp_path / "run", identity=identity(), resume=True)
    assert resumed["complete"] and len(engine.calls) == 3
    assert [record["judge"]["price_date"] for record in resumed["accounting_history"]] == ["2026-10-10", "2026-10-11"]
    assert resumed["run_hash"] == manifest["run_hash"]


def test_generation_budget_is_not_attempted_but_http403_is_failed_once(tmp_path):
    from legal_rag.providers.errors import BudgetExceeded, ProviderError

    benchmark = load_benchmark(sources(tmp_path / "sources", count=3), expected_count=3)

    class BlockedEngine(FakeEngine):
        blocked = True

        def answer(self, question, **kwargs):
            if self.blocked and len(self.calls) == 1:
                self.calls.append((question, kwargs["profile"], None))
                raise BudgetExceeded()
            return super().answer(question, **kwargs)

    engine = BlockedEngine()
    manifest = run_evaluation(benchmark, engine, FakeJudge(), tmp_path / "budget", identity=identity())
    row = load_results(tmp_path / "budget")[1]
    assert row.score is None and not row.generation_attempted
    assert manifest["stopped_reason"] == "generator:budget_exceeded" and len(engine.calls) == 2
    engine.blocked = False
    resumed = run_evaluation(benchmark, engine, FakeJudge(), tmp_path / "budget", identity=identity(), resume=True)
    assert resumed["complete"] and len(engine.calls) == 4

    class ForbiddenEngine(FakeEngine):
        def answer(self, question, **kwargs):
            self.calls.append((question, kwargs["profile"], None))
            raise ProviderError("http_403", "Forbidden")

    forbidden = ForbiddenEngine()
    manifest = run_evaluation(benchmark, forbidden, FakeJudge(), tmp_path / "forbidden", identity=identity())
    rows = load_results(tmp_path / "forbidden")
    assert len(forbidden.calls) == len(rows) == 1 and rows[0].score == 0 and rows[0].generation_attempted
    assert manifest["stopped_reason"] == "generator:http_403" and not manifest["complete"]


def test_unverified_context_stops_before_model_attempt_without_zero_score(tmp_path):
    from legal_rag.providers.errors import ProviderError

    benchmark = load_benchmark(sources(tmp_path / 'sources', count=3), expected_count=3)

    class UnverifiedContextEngine(FakeEngine):
        preflight_checks = 0

        def answer(self, question, **kwargs):
            self.preflight_checks += 1
            raise ProviderError('context_unverified', 'Model context has not been verified')

    engine, judge = UnverifiedContextEngine(), FakeJudge()
    manifest = run_evaluation(benchmark, engine, judge, tmp_path / 'run', identity=identity())
    rows = load_results(tmp_path / 'run')
    assert engine.preflight_checks == 1 and not engine.calls and not judge.calls
    assert not manifest['complete'] and manifest['status'] == 'incomplete'
    assert manifest['stopped_reason'] == 'generator:context_unverified'
    assert len(rows) == 1 and rows[0].score is None and not rows[0].generation_attempted
    checkpoint = json.loads((tmp_path / 'run/answers.jsonl').read_text())
    assert not checkpoint['attempted'] and checkpoint['answer']['usage'] == []
    metrics = json.loads((tmp_path / 'run/metrics.json').read_text())['no_hint']
    assert metrics['denominator'] == metrics['missing_scores'] == 3
    assert metrics['score_distribution']['0'] == 0 and metrics['generation_errors'] == 0


def test_cost_summary_labels_conservative_usage_and_unknown_costs(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class EstimatedJudge(FakeJudge):
        def complete(self, *args):
            payload, usage = super().complete(*args)
            return payload, usage.model_copy(update={"cost_eur": .01, "cost_basis": "conservative_reservation", "ledger_entry_id": "test-ledger"})

    run_evaluation(benchmark, FakeEngine(), EstimatedJudge(), tmp_path / "run", identity=identity())
    usage = json.loads((tmp_path / "run/metrics.json").read_text())["no_hint"]["usage"]
    assert usage["known_cost_eur"] == .02 and usage["unknown_cost_calls"] == 2
    assert usage["cost_eur_by_basis"]["conservative_reservation"] == .02
    assert "not invoiced costs" in usage["cost_interpretation"]


def test_control_reversal_uses_full_run_winner_instead_of_primary_sample(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources", count=3), expected_count=3)

    class SelectiveEngine(FakeEngine):
        def __init__(self, wrong_questions):
            super().__init__()
            self.wrong_questions = wrong_questions

        def answer(self, question, **kwargs):
            self.wrong = question in self.wrong_questions
            return super().answer(question, **kwargs)

    class FlippedControl(FakeControlJudge):
        def complete(self, *args):
            payload, usage = super().complete(*args)
            return payload.model_copy(update={"correctness": 2 if payload.correctness == 0 else 0}), usage

    for flip in (True, False):
        paths = []
        for profile, wrong_questions in (("dense", {"Quali organi?", "Quanto dura?"}), ("hybrid", {"Chi decide?"})):
            path = tmp_path / f"{flip}-{profile}"
            run_evaluation(benchmark, SelectiveEngine(wrong_questions), FakeJudge(), path, profile=profile,
                identity=identity(), judge_control=FlippedControl() if flip else FakeControlJudge(), control_size=1)
            paths.append(path)
        compared = compare_runs(paths, resamples=100)["comparisons"][0]["formats"]["no_hint"]
        assert compared["fully_correct"]["delta"] > 0
        assert compared["control_primary_sample_delta"] == -1
        assert compared["promotion_gates"]["control_does_not_reverse_winner"] is flip


def test_same_model_control_does_not_authorize_promotion(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class SameModelDifferentTimeout(FakeJudge):
        identity = {"provider": "openai", "model": "judge", "timeout_seconds": 90}

    for profile, wrong in (("dense", True), ("hybrid", False)):
        run_evaluation(benchmark, FakeEngine(wrong=wrong), FakeJudge(), tmp_path / profile,
                       identity=identity(), profile=profile, judge_control=SameModelDifferentTimeout())
    compared = compare_runs([tmp_path / "dense", tmp_path / "hybrid"], resamples=100)["comparisons"][0]
    values = compared["formats"]["no_hint"]
    assert compared["comparable"] and values["fully_correct"]["delta"] == 1
    assert values["promotion_gates"]["control_present"] and values["promotion_gates"]["control_sample_coverage"]
    assert not values["promotion_gates"]["distinct_control_model"] and not values["promoted"]


def test_review_approval_requires_reason_primary_source_and_revision_evidence():
    base = {"qid": "eval-0001", "status": "current_revised", "human_approved": True,
            "reviewer": "Researcher", **REVIEW_PROVENANCE}
    for changes, error in (({"note": " "}, "reason"), ({"source_urls": []}, "source URLs"),
                           ({"source_urls": ["javascript:alert(1)"]}, "HTTP"),
                           ({"source_urls": ["https://user:secret@example.test/source"]}, "HTTP"),
                           ({"evidence": " "}, "supporting evidence")):
        with pytest.raises(ValueError, match=error):
            Review(**{**base, **changes})
    assert Review(**{**base, "status": "current_supported", "evidence": ""}).human_approved
    assert not Review(qid="eval-0001", note="Pending source verification").human_approved


@pytest.mark.parametrize("role", ["generator", "judge", "control_judge"])
@pytest.mark.parametrize("both_unknown", [False, True])
def test_successful_unknown_model_outputs_are_not_controlled(tmp_path, role, both_unknown):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class UnknownEngine(FakeEngine):
        def answer(self, *args, **kwargs):
            answer = super().answer(*args, **kwargs)
            answer.usage[-1].returned_model = "unknown"
            return answer

    class UnknownJudge(FakeJudge):
        def complete(self, *args):
            payload, usage = super().complete(*args)
            return payload, usage.model_copy(update={"returned_model": "unknown"})

    class UnknownControl(FakeControlJudge):
        def complete(self, *args):
            payload, usage = super().complete(*args)
            return payload, usage.model_copy(update={"returned_model": "unknown"})

    for profile in ("dense", "hybrid"):
        unknown = both_unknown or profile == "hybrid"
        run_evaluation(benchmark, UnknownEngine() if unknown and role == "generator" else FakeEngine(),
            UnknownJudge() if unknown and role == "judge" else FakeJudge(), tmp_path / profile, profile=profile, identity=identity(),
            judge_control=UnknownControl() if unknown and role == "control_judge" else FakeControlJudge())
    compared = compare_runs([tmp_path / "dense", tmp_path / "hybrid"], resamples=100)["comparisons"][0]
    assert not compared["comparable"] and f"unknown_effective_{role}" in compared["issues"]


def test_known_success_after_unknown_failed_attempt_and_resolved_alias_is_comparable(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / "sources"), expected_count=2)

    class PriorUnknownEngine(FakeEngine):
        def answer(self, *args, **kwargs):
            answer = super().answer(*args, **kwargs)
            answer.usage[-1].returned_model = "answer-resolved"
            answer.usage.insert(0, Usage(provider="utopia", requested_model="answer", returned_model="unknown"))
            return answer

    class RetryJudge(FakeJudge):
        blocked = True

        def complete(self, *args):
            if self.blocked:
                error = RuntimeError("A failed transport attempt")
                error.usage = [Usage(provider="openai", requested_model="judge", returned_model="unknown")]
                raise error
            return super().complete(*args)

    for profile in ("dense", "hybrid"):
        judge = RetryJudge()
        run_evaluation(benchmark, PriorUnknownEngine(), judge, tmp_path / profile, profile=profile, identity=identity())
        judge.blocked = False
        run_evaluation(benchmark, PriorUnknownEngine(), judge, tmp_path / profile, profile=profile, identity=identity(), resume=True)
    assert compare_runs([tmp_path / "dense", tmp_path / "hybrid"], resamples=100)["comparisons"][0]["comparable"]


def test_evaluation_progress_observes_persisted_checkpoints_and_final_export(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / 'sources'), expected_count=2)
    output = tmp_path / 'run'
    engine, judge, events = FakeEngine(), FakeJudge(), []

    def progress(event):
        events.append(event)
        if event['stage'] == 'generation_checkpoint':
            checkpoints = [json.loads(line) for line in (output / 'answers.jsonl').read_text().splitlines()]
            assert len(checkpoints) == event['processed']
            assert checkpoints[-1]['key'] == [event['qid'], 'no_hint']
            assert checkpoints[-1]['answer']['answer'] == 'Presidente e Consiglio.'
            assert len(judge.calls) == event['processed'] - 1
        elif event['stage'] == 'evaluation_record':
            saved = load_results(output)
            assert len(saved) == event['processed'] and saved[-1].qid == event['qid']
        elif event['stage'] == 'evaluation_complete':
            saved_manifest = json.loads((output / 'manifest.json').read_text())
            assert saved_manifest['complete'] and event['status'] == 'complete'
            assert all(file_hash(output / name) == expected for name, expected in saved_manifest['output_hashes'].items())
            metrics = json.loads((output / 'metrics.json').read_text())['no_hint']
            assert metrics['denominator'] == 2 and metrics['missing_scores'] == 0

    run_evaluation(benchmark, engine, judge, output, identity=identity(), progress=progress)
    assert events[0]['stage'] == 'evaluation_start' and events[0]['processed'] == 0
    assert events[-1]['stage'] == 'evaluation_complete' and events[-1]['processed'] == 2
    assert [event['processed'] for event in events if event['stage'] == 'generation_checkpoint'] == [1, 2]
    assert [event['processed'] for event in events if event['stage'] == 'evaluation_record'] == [1, 2]


def test_diagnostic_progress_counts_completed_ram_records_without_new_metrics(tmp_path):
    benchmark = load_benchmark(sources(tmp_path / 'sources'), expected_count=2)
    profiles = ('dense', 'sparse')
    baseline = tmp_path / 'baseline'
    run_retrieval_diagnostics(benchmark, FakeRetriever(), FakeCounter(), baseline, profiles=profiles)
    output, retriever, events = tmp_path / 'monitored', FakeRetriever(), []

    def progress(event):
        events.append(event)
        if event['stage'] == 'retrieval_record':
            assert len(retriever.calls) == event['processed']
            assert not output.exists()
        elif event['stage'] == 'diagnostic_complete':
            saved_manifest = json.loads((output / 'manifest.json').read_text())
            assert event['status'] == 'complete' and saved_manifest['complete']
            assert all(file_hash(output / name) == expected for name, expected in saved_manifest['output_hashes'].items())
            rows = load_diagnostics(output)
            assert len(rows) == event['processed'] == 4
            assert all(row.retrieval_only and not row.answer.answer and not row.answer.usage for row in rows)

    run_retrieval_diagnostics(benchmark, retriever, FakeCounter(), output, profiles=profiles, progress=progress)
    assert events[0]['stage'] == 'diagnostic_start' and events[0]['total'] == 4
    assert [event['processed'] for event in events if event['stage'] == 'retrieval_record'] == [1, 2, 3, 4]
    assert [event['profile'] for event in events if event['stage'] == 'retrieval_record'] == ['dense', 'dense', 'sparse', 'sparse']
    assert events[-1]['stage'] == 'diagnostic_complete' and events[-1]['processed'] == 4
    baseline_metrics = json.loads((baseline / 'metrics.json').read_text())
    monitored_metrics = json.loads((output / 'metrics.json').read_text())
    for profile in profiles:
        baseline_metrics[profile].pop('timings_seconds')
        monitored_metrics[profile].pop('timings_seconds')
    assert monitored_metrics == baseline_metrics

from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from legal_rag.artifacts import file_hash, write_jsonl
from legal_rag.contracts import AnswerPayload, Block, Candidate, Chunk, Citation, Document, EngineConfig, Relation, SourceSpan, Unit, Usage
from legal_rag.generation import RagEngine
from legal_rag.generation.engine import QueryVariants, RerankPayload, validate_citations
from legal_rag.generation.enrichment import ProposalPayload, propose_enrichment
from legal_rag.retrieval import Retriever, build_context, fuse_rankings


class Words:
    name, revision = "fixture", "v1"

    def count(self, text):
        return len(text.split())


def chunk(identifier, *, text=None, unit=None, status="no_cessation_evidence"):
    return Chunk(chunk_id=identifier, document_id="law", snapshot_id="snapshot", title="Law", article_label=identifier,
                 text=text or f"Evidence from {identifier}.", embedding_text=text or f"Evidence from {identifier}.",
                 unit_ids=[unit or f"law#art:{identifier}"], spans=[], source_url="https://example.org/law", status=status)


def candidates(chunks, origin="hybrid"):
    return [Candidate(chunk=c, rank=i + 1, score=1 / (i + 1), origin=origin) for i, c in enumerate(chunks)]


class Chat:
    identity = {"provider": "fake", "model": "test"}

    def __init__(self, responses):
        self.responses, self.prompts = list(responses), []

    def complete(self, prompt, schema):
        self.prompts.append((prompt, schema))
        output = self.responses.pop(0)
        if callable(output):
            output = output(prompt)
        return schema.model_validate(output), Usage(provider="fake", requested_model="test", returned_model="test")


class FixedRetriever:
    identity = {"index": "fixture"}
    manifest = {"snapshot_id": "snapshot"}

    def __init__(self, chunks):
        self.hits, self.calls = candidates(chunks), []

    def search(self, question, *, mode, config, trace):
        self.calls.append(question)
        trace.append({"query": question, "mode": mode})
        return self.hits


def answer(quote="Evidence", context_id="E1"):
    return AnswerPayload(answer="Risposta fondata.", status="answered", citations=[Citation(context_id=context_id, quote=quote)])


def test_context_counts_headers_and_keeps_complete_source_units():
    short = chunk("short", text="Complete short text.")
    oversized = chunk("long", text="word " * 300)
    one = build_context(candidates([short]), Words())
    context = build_context(candidates([oversized, short, short, chunk("last")]), Words(), max_tokens=one.token_count, max_units=1)
    assert [e.chunk.chunk_id for e in context.evidence] == ["short"]
    assert context.evidence[0].chunk.text in context.text
    assert context.token_count == Words().count(context.text)
    assert {item["reason"] for item in context.omissions} == {"token_limit", "unit_limit"}


@pytest.mark.parametrize("warning", ["source_quote_boundary_recovered", "source_quote_boundary_recovered:117"])
def test_context_shows_law_identity_and_specific_parsing_uncertainty_within_budget(warning):
    ordinary = chunk("1")
    fallback = build_context(candidates([ordinary]), Words())
    assert fallback.text.startswith("[E1] law\nLaw\nArticle:")
    identified = ordinary.model_copy(update={"identity": "Legge regionale 3 dicembre 2007, n. 31",
                                           "warnings": ["duplicate_unit_id", "unrelated_document_warning"]})
    identified_context = build_context(candidates([identified]), Words())
    assert identified_context.text.startswith("[E1] Legge regionale 3 dicembre 2007, n. 31\nLaw\nArticle:")
    assert "Parsing uncertainty:" not in identified_context.text
    recovered = identified.model_copy(update={"warnings": [*identified.warnings, warning]})
    context = build_context(candidates([recovered]), Words())
    assert "Parsing uncertainty: source_quote_boundary_recovered\n" in context.text
    assert "duplicate_unit_id" not in context.text and "unrelated_document_warning" not in context.text
    assert context.token_count == Words().count(context.text)
    assert context.token_count > identified_context.token_count > fallback.token_count
    omitted = build_context(candidates([recovered]), Words(), max_tokens=identified_context.token_count)
    assert not omitted.evidence and omitted.omissions == [{"chunk_id": "1", "reason": "token_limit"}]
    exact = build_context(candidates([recovered]), Words(), max_tokens=context.token_count)
    assert exact.text == context.text and exact.token_count == context.token_count


def test_quote_warning_prefix_does_not_accept_unrelated_marker_names():
    record = chunk("1").model_copy(update={"warnings": ["source_quote_boundary_recovered_unrelated"]})
    assert "Parsing uncertainty:" not in build_context(candidates([record]), Words()).text


def test_rrf_uses_explicit_zero_based_k60_and_matches_native_qdrant():
    client = QdrantClient(":memory:")
    client.create_collection("chunks", vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
                             sparse_vectors_config={"sparse": models.SparseVectorParams()})
    records = [chunk("a"), chunk("b"), chunk("c")]
    for i, record in enumerate(records):
        client.upsert("chunks", [models.PointStruct(id=str(uuid5(NAMESPACE_URL, record.chunk_id)),
            vector={"dense": [[1, 0], [.5, .5], [0, 1]][i], "sparse": models.SparseVector(indices=[1], values=[1, 3, 2][i:i + 1])},
            payload=record.to_json_record())])
    try:
        prefetch = [models.Prefetch(query=[1., 0.], using="dense", limit=3),
                    models.Prefetch(query=models.SparseVector(indices=[1], values=[1]), using="sparse", limit=3)]
        native = client.query_points("chunks", prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=60)), limit=3).points
        fused = fuse_rankings([candidates(records, "dense"), candidates([records[1], records[2], records[0]], "sparse")])
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 60)
        assert {c.chunk.chunk_id: c.score for c in fused} == pytest.approx({p.payload["chunk_id"]: p.score for p in native})
    finally:
        client.close()


def test_context_deduplicates_source_spans_across_distinct_chunk_ids():
    original = chunk("first").model_copy(update={"spans": [SourceSpan(block_id="block", start=0, end=40)]})
    duplicate = original.model_copy(update={"chunk_id": "different-id"})
    subset = original.model_copy(update={"chunk_id": "subset", "spans": [SourceSpan(block_id="block", start=10, end=20)]})
    partial = original.model_copy(update={"chunk_id": "partial", "spans": [SourceSpan(block_id="block", start=20, end=60)]})
    context = build_context(candidates([original, duplicate, subset, partial]), Words())
    assert [e.chunk.chunk_id for e in context.evidence] == ["first", "partial"]
    assert context.omissions == [{"chunk_id": "different-id", "reason": "source_already_shown"}, {"chunk_id": "subset", "reason": "source_already_shown"}]
    assert context.text.count(original.text) == 2


def test_citation_validation_requires_displayed_exact_source_quote():
    context = build_context(candidates([chunk("one")]), Words())
    checked = validate_citations(answer(), context)
    assert checked[0].source_url == "https://example.org/law"
    assert checked[0].chunk_id == "one"
    for invalid in (answer("invented"), answer(context_id="E2"), AnswerPayload(answer="Unsupported", status="answered")):
        with pytest.raises(ValueError):
            validate_citations(invalid, context)


def test_no_rag_does_not_retrieve_and_rag_empty_context_does_not_generate():
    retriever = FixedRetriever([])
    client = Chat([AnswerPayload(answer="Knowledge answer", status="uncertain")])
    engine = RagEngine(client, retriever, Words())
    no_rag = engine.answer("Question", profile="no_rag")
    empty = engine.answer("Question", profile="hybrid")
    assert no_rag.status == "uncertain" and not no_rag.context.text
    assert empty.status == "insufficient_evidence"
    assert retriever.calls == ["Question"]
    assert len(client.prompts) == 1
    assert "using your knowledge" in client.prompts[0][0]


def test_snapshot_date_is_rag_provenance_and_no_rag_output_only():
    client = Chat([answer(), AnswerPayload(answer="Knowledge answer", status="uncertain")])
    retriever = FixedRetriever([chunk("one")])
    date = "2026-10-10T12:00:00Z"
    engine = RagEngine(client, retriever, Words(), snapshot_id="snapshot", snapshot_date=date)
    grounded = engine.answer("Question", profile="hybrid")
    no_rag = engine.answer("Question", profile="no_rag")
    assert grounded.status == "answered" and no_rag.status == "uncertain"
    assert grounded.snapshot_id == no_rag.snapshot_id == "snapshot"
    assert grounded.snapshot_date == no_rag.snapshot_date == date
    assert f"Corpus acquisition date: {date}" in client.prompts[0][0]
    assert "does not certify legal applicability" in client.prompts[0][0]
    assert date not in client.prompts[1][0] and "Corpus acquisition date:" not in client.prompts[1][0]
    assert retriever.hits[0].chunk.text not in client.prompts[1][0] and not no_rag.context.text
    assert retriever.calls == ["Question"]
    assert engine.identity["prompts"]["answer"] == "answer-v2"


def test_snapshot_date_is_not_invented_for_existing_python_callers():
    client = Chat([answer()])
    result = RagEngine(client, FixedRetriever([chunk("one")]), Words()).answer("Question")
    assert result.snapshot_id == "snapshot" and result.snapshot_date is None
    assert "Corpus acquisition date: unknown" in client.prompts[0][0]


def test_no_rag_without_retriever_preserves_explicit_snapshot_metadata():
    client = Chat([AnswerPayload(answer="Knowledge answer", status="uncertain")])
    result = RagEngine(client, None, Words(), snapshot_id="snapshot", snapshot_date="2026-10-10").answer("Question", profile="no_rag")
    assert result.snapshot_id == "snapshot" and result.snapshot_date == "2026-10-10"
    assert "2026-10-10" not in client.prompts[0][0] and not result.candidates


def test_multiquery_original_plus_two_distinct_queries_is_cached(tmp_path):
    client = Chat([QueryVariants(queries=["Variant one", "Variant two"]), answer(), answer()])
    retriever = FixedRetriever([chunk("one")])
    engine = RagEngine(client, retriever, Words(), cache_dir=tmp_path)
    first = engine.answer("Original", profile="multiquery")
    second = engine.answer("Original", profile="multiquery")
    assert first.status == second.status == "answered"
    assert first.queries == ["Original", "Variant one", "Variant two"]
    assert second.retrieval_trace["rewrite_cache_hit"] is True
    assert retriever.calls == first.queries * 2
    assert sum(schema is QueryVariants for _, schema in client.prompts) == 1
    assert len(second.retrieval_trace["query_rankings"]) == 3


def test_invalid_multiquery_variants_use_visible_original_only_policy():
    client = Chat([QueryVariants(queries=["Original", "Original"]), answer()])
    result = RagEngine(client, FixedRetriever([chunk("one")]), Words()).answer("Original", profile="multiquery")
    assert result.status == "answered"
    assert result.queries == ["Original"]
    assert result.retrieval_trace["rewrite_fallback"] == "original_only"


def test_rerank_scores_exact_twenty_candidate_pool_and_uses_same_context_budget():
    records = [chunk(str(i)) for i in range(25)]
    scored = RerankPayload(scores=[{"chunk_id": str(i), "score": 2 if i == 19 else 0} for i in range(20)])
    client = Chat([scored, answer("Evidence from 19.")])
    result = RagEngine(client, FixedRetriever(records), Words(), EngineConfig(context_units=1)).answer("Question", profile="rerank")
    assert result.status == "answered"
    assert len(result.retrieval_trace["rerank_input"]) == 20
    assert result.context.evidence[0].chunk.chunk_id == "19"
    assert result.context.token_count <= 4096


def test_rerank_missing_or_fabricated_id_is_an_exposed_failure():
    client = Chat([RerankPayload(scores=[{"chunk_id": "fabricated", "score": 2}])])
    result = RagEngine(client, FixedRetriever([chunk("one")]), Words()).answer("Question", profile="rerank")
    assert result.status == "error"
    assert "exact supplied candidate set" in result.errors[0]


def test_graph_adds_only_two_resolved_articles_into_actual_context_and_oracle_verifies_source(tmp_path):
    records = [chunk(str(i), status="unknown" if i == 6 else "no_cessation_evidence") for i in range(7)]
    units = [Unit(unit_id=f"law#art:{i}", document_id="law", kind="article", label=str(i), text=c.text) for i, c in enumerate(records)]
    relations = [Relation(relation_id=f"reference-{i}", source_id="law#art:0", target_id=f"law#art:{i}", kind="references", resolution="confirmed") for i in (5, 6)]
    relations.append(Relation(relation_id="unresolved", source_id="law#art:0", target_reference="missing#art:1", kind="references"))
    for name, rows in (("chunks", records), ("units", units), ("relations", relations)):
        write_jsonl(tmp_path / (name + ".jsonl"), rows)

    class Embedder:
        identity = {"model": "fixture", "dimension": 2, "idf": False}

        def embed_hybrid(self, texts, query=False):
            return [[1., 0.] for _ in texts], [models.SparseVector(indices=[1], values=[1]) for _ in texts]

    client = QdrantClient(":memory:")
    client.create_collection("chunks", vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)}, sparse_vectors_config={"sparse": models.SparseVectorParams()})
    for i, c in enumerate(records):
        client.upsert("chunks", [models.PointStruct(id=str(uuid5(NAMESPACE_URL, c.chunk_id)),
            vector={"dense": [1., i / 2], "sparse": models.SparseVector(indices=[1], values=[7. - i])}, payload=c.to_json_record())])
    try:
        embedder = Embedder()
        manifest = {"ready_for_retrieval": True, "snapshot_id": "snapshot", "embedding": embedder.identity, "sparse_embedding": embedder.identity,
                    "files": {name + ".jsonl": file_hash(tmp_path / (name + ".jsonl")) for name in ("chunks", "units", "relations")}}
        retriever = Retriever(client, embedder, corpus_dir=tmp_path, manifest=manifest)
        engine = RagEngine(Chat([answer("Evidence from 5.", "E4"), answer("Evidence from 5.")]), retriever, Words(), EngineConfig(candidate_limit=5, context_units=5))
        result = engine.answer("Question", profile="graph")
        assert result.status == "answered"
        assert result.retrieval_trace["graph_additions"] == ["5", "6"]
        assert [e.chunk.chunk_id for e in result.context.evidence] == ["0", "1", "2", "5", "6"]
        oracle = engine.answer("Question", profile="oracle", oracle_chunks=[records[5]])
        assert oracle.status == "answered"
        fabricated = records[5].model_copy(update={"text": "Fabricated gold answer"})
        rejected = engine.answer("Question", profile="oracle", oracle_chunks=[fabricated])
        assert rejected.status == "error" and "exact source record" in rejected.errors[0]
    finally:
        client.close()


def test_graph_adjacencies_are_reused_with_identical_order_and_dynamic_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace

    records = [chunk("seed", unit="law#art:1"), chunk("a", unit="law#art:4", status="ceased_confirmed"),
               chunk("b", unit="law#art:4", status="unknown"), chunk("c", unit="law#art:5"), chunk("d", unit="law#art:6")]
    units = [Unit(unit_id=f"law#art:{label}", document_id="law", kind="article", label=str(label), text="Article text") for label in (1, 4, 5, 6)]
    relations = [Relation(relation_id="r20", source_id="law#art:1", target_id="law#art:5", kind="references", resolution="confirmed"),
                 Relation(relation_id="r10", source_id="law", target_id="law#art:4", kind="amends", resolution="confirmed"),
                 Relation(relation_id="r05", source_id="unrelated", target_id="law#art:6", kind="references", resolution="confirmed"),
                 Relation(relation_id="r08", source_id="law#art:1", target_id="law#art:6", kind="references", resolution="unresolved"),
                 Relation(relation_id="r09", source_id="law#art:1", target_id="law#art:6", kind="contains", resolution="confirmed"),
                 Relation(relation_id="r40", source_id="law#art:4", target_id="law#art:6", kind="references", resolution="confirmed")]
    for name, rows in (("chunks", records), ("units", units), ("relations", relations)):
        write_jsonl(tmp_path / (name + ".jsonl"), rows)
    embedder = SimpleNamespace(identity={"model": "fixture"})
    manifest = {"ready_for_retrieval": True, "snapshot_id": "snapshot", "embedding": embedder.identity, "sparse_embedding": None,
                "files": {name + ".jsonl": file_hash(tmp_path / (name + ".jsonl")) for name in ("chunks", "units", "relations")}}
    retriever = Retriever(None, embedder, corpus_dir=tmp_path, manifest=manifest)
    added, used = retriever.expand_graph(candidates(records[:1]), config=EngineConfig())
    assert [item.chunk.chunk_id for item in added] == ["b", "c"] and used == ["r10", "r20"]
    adjacency, chunks_by_unit = retriever._graph
    assert [relation.relation_id for relation in adjacency["law#art:1"]] == ["r20"]
    assert [record.chunk_id for record in chunks_by_unit["law#art:4"]] == ["a", "b"]
    frozen = retriever._graph
    monkeypatch.setattr("legal_rag.retrieval.retriever.read_jsonl", lambda *args: pytest.fail("Graph artifacts must be loaded once"))
    monkeypatch.setattr("legal_rag.retrieval.retriever.file_hash", lambda *args: pytest.fail("Frozen adjacency must be reused"))
    repeated, repeated_used = retriever.expand_graph(candidates(records[:1]), config=EngineConfig())
    assert repeated == added and repeated_used == used and retriever._graph is frozen
    historical, historical_used = retriever.expand_graph(candidates(records[:1]), config=EngineConfig(exclude_ceased=False))
    assert [item.chunk.chunk_id for item in historical] == ["a", "b"] and historical_used == ["r10", "r10"]


def test_failed_quote_preserves_raw_output_for_inspection():
    result = RagEngine(Chat([answer("fabricated quote")]), FixedRetriever([chunk("one")]), Words()).answer("Question")
    assert result.status == "error" and result.raw_payload["citations"][0]["quote"] == "fabricated quote"
    assert not result.citations


def test_optional_enrichment_only_proposes_verified_spans_and_never_changes_status():
    document = Document(document_id="law", source_id="source", snapshot_id="snapshot", source_url="https://example.org/law",
                        source_path="law.html", identity="Law 1", title="Law", availability="body")
    block = Block(block_id="b1", document_id="law", position=0, tag="p", text="La legge è abrogata dal 1 gennaio 2020.")
    payload = ProposalPayload(proposals=[
        {"kind": "cessation_evidence", "block_id": "b1", "quote": "è abrogata", "description": "Explicit wording"},
        {"kind": "date", "block_id": "b1", "quote": "invented date", "description": "Unsupported"}])
    result = propose_enrichment(Chat([payload]), document, [block])
    assert len(result.proposals) == 1 and len(result.rejected) == 1
    assert result.proposals[0].review_state == "proposed"
    span = result.proposals[0].span
    assert block.text[span.start:span.end] == result.proposals[0].evidence
    assert document.status == "no_cessation_evidence"


def test_oracle_respects_operative_policy_and_abstains_without_eligible_evidence():
    ceased = chunk("gone", status="ceased_confirmed")

    class OracleRetriever(FixedRetriever):
        def verify_oracle(self, records):
            assert records == [ceased]

    client = Chat([])
    engine = RagEngine(client, OracleRetriever([]), Words())
    result = engine.answer("Current question", profile="oracle", oracle_chunks=[ceased])
    assert result.status == "insufficient_evidence" and not client.prompts
    assert result.retrieval_trace["oracle_policy_omissions"] == ["gone"]
    inclusive = RagEngine(Chat([answer("Evidence from gone.")]), OracleRetriever([]), Words(), EngineConfig(exclude_ceased=False))
    historical = inclusive.answer("Historical question", profile="oracle", oracle_chunks=[ceased])
    assert historical.status == "answered" and historical.retrieval_trace["oracle_policy_omissions"] == []


def test_invalid_model_json_is_preserved_for_inspection_without_provider_body():
    from legal_rag.contracts import ProviderConfig
    from legal_rag.providers import UtopiaClient
    from types import SimpleNamespace

    session = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=200,
        json=lambda: {"model": "SLURM.gpt-oss:120b", "message": {"content": "invalid model JSON"}}))
    client = UtopiaClient(ProviderConfig(), api_key="secret", session=session)
    result = RagEngine(client, None, Words()).answer("Question", profile="no_rag")
    assert result.status == "error" and result.raw_payload == {"model_text": "invalid model JSON"}
    assert "secret" not in str(result.to_json_record())

import json
from types import SimpleNamespace
from datetime import date, timedelta

import pytest

from legal_rag.contracts import ProviderConfig, Record
from legal_rag.providers import OpenAIClient, ProviderError, UtopiaClient
from legal_rag.providers.budget import BudgetLedger
from legal_rag.providers.embeddings import BgeEmbedder, OpenAIEmbedder, UtopiaEmbedder


class Output(Record):
    text: str


def priced(**kwargs):
    values = dict(provider="openai", model="test", price_date=date.today().isoformat(), price_source_url="https://openai.com/api/pricing/",
                  fx_date=date.today().isoformat(), fx_source_url="https://www.ecb.europa.eu/", input_usd_per_million=0.1, output_usd_per_million=0.5)
    return ProviderConfig(**(values | kwargs))


def response_sdk(responses):
    def parse(**kwargs):
        response = responses.parse(**kwargs)
        counters = getattr(response, "usage", None)
        text = getattr(response, "output_text", None)
        data = {"status": response.status, "model": getattr(response, "model", None),
                "usage": {"input_tokens": getattr(counters, "input_tokens", None),
                          "output_tokens": getattr(counters, "output_tokens", None)},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
                          if isinstance(text, str) else []}
        return SimpleNamespace(status_code=200, http_response=SimpleNamespace(json=lambda: data), parse=lambda: response)

    return SimpleNamespace(responses=SimpleNamespace(with_raw_response=SimpleNamespace(parse=parse)))


def http_response_payload(text, *, status="completed", refusal=False, include_usage=True):
    content = {"type": "refusal", "refusal": text} if refusal else {"type": "output_text", "text": text, "annotations": []}
    return {"id": "resp-test", "object": "response", "created_at": 0, "status": status,
            "model": "resolved-model", "metadata": {"private": "private-body-field"},
            "output": [{"id": "msg-test", "type": "message", "role": "assistant", "status": "completed", "content": [content]}],
            "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}} if include_usage else None}


@pytest.fixture
def sdk_http_response():
    import httpx
    from openai import OpenAI

    clients = []

    def build(payload, *, status_code=200, raw_body=None):
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(status_code, content=raw_body if raw_body is not None else json.dumps(payload),
                                  headers={"content-type": "application/json", "x-private": "private-header"})

        sdk = OpenAI(api_key="runtime-private-key", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        clients.append(sdk)
        return sdk, calls

    yield build
    for client in clients:
        client.close()


@pytest.mark.parametrize("include_usage", [True, False])
@pytest.mark.parametrize("original,status,refusal,error", [
    (' \n { "text" : "answer" } \t\n', "completed", False, None),
    (' { "text" : "unfinished"', "completed", False, "invalid_output"),
    (' { "unexpected" : "answer" }\n', "completed", False, "invalid_output"),
    (' { "text" : "partial"', "incomplete", False, "truncated_output"),
    ("I cannot provide this output.\n", "completed", True, "refusal"),
])
def test_sdk_http200_keeps_output_and_usage_before_validation(tmp_path, sdk_http_response, original, status, refusal, error, include_usage):
    sdk, calls = sdk_http_response(http_response_payload(original, status=status, refusal=refusal, include_usage=include_usage))
    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=sdk)
    if error:
        with pytest.raises(ProviderError) as caught:
            client.complete("Synthetic contract question", Output)
        assert caught.value.code == error and len(caught.value.usage) == 1
        assert caught.value.raw_payload == {"model_text": original}
        usage = caught.value.usage[0]
        assert "runtime-private-key" not in str(caught.value)
    else:
        parsed, usage = client.complete("Synthetic contract question", Output)
        assert parsed.text == "answer"
    assert len(calls) == 1
    assert calls[0]["store"] is False and calls[0]["truncation"] == "disabled"
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True
    assert calls[0]["text"]["format"]["schema"]["additionalProperties"] is False
    assert usage.returned_model == "resolved-model" and usage.raw_output == original
    assert usage.input_tokens == (50 if include_usage else None)
    assert usage.output_tokens == (20 if include_usage else None)
    assert usage.cost_basis == ("conservative_estimate" if include_usage else "conservative_reservation")
    assert client.ledger.snapshot()["spent_or_reserved_eur"] == pytest.approx(usage.cost_eur)
    entry = json.loads(client.ledger.path.read_text())["entries"][0]
    assert entry["state"] == ("settled" if include_usage else "uncertain")
    if include_usage:
        assert usage.cost_eur == pytest.approx(client.ledger.cost(50, 20))
    else:
        assert usage.cost_eur == entry["reserved_eur"]
    for private in ("runtime-private-key", "private-header", "private-body-field"):
        assert private not in str(usage.to_json_record())


@pytest.mark.parametrize("status_code,payload,raw_body,error,retained", [
    (200, None, "private-unreadable-envelope", "invalid_output", True),
    (429, {"error": {"message": "private-http-error", "type": "insufficient_quota", "code": "insufficient_quota"}}, None, "http_429", False),
])
def test_sdk_invalid_envelope_and_quota_keep_only_safe_accounting(tmp_path, sdk_http_response, status_code, payload, raw_body, error, retained):
    sdk, calls = sdk_http_response(payload, status_code=status_code, raw_body=raw_body)
    client = OpenAIClient(priced(retries=0), ledger_path=tmp_path / "ledger.json", sdk=sdk)
    with pytest.raises(ProviderError) as caught:
        client.complete("Synthetic contract question", Output)
    assert caught.value.code == error and len(calls) == 1 and len(caught.value.usage) == 1
    usage = caught.value.usage[0]
    assert usage.returned_model == "unknown" and usage.raw_output is None and caught.value.raw_payload is None
    assert usage.input_tokens is None and usage.output_tokens is None
    assert (usage.cost_eur > 0) == retained
    assert client.ledger.snapshot()["spent_or_reserved_eur"] == usage.cost_eur
    for private in ("runtime-private-key", "private-header", "private-http-error", "private-unreadable-envelope"):
        assert private not in str(caught.value) and private not in str(usage.to_json_record())


def test_sdk_budget_exhaustion_prevents_http_dispatch(tmp_path, sdk_http_response):
    sdk, calls = sdk_http_response(http_response_payload('{"text":"answer"}'))
    client = OpenAIClient(priced(budget_eur=.000001, budget_reserve_eur=0), ledger_path=tmp_path / "ledger.json", sdk=sdk)
    with pytest.raises(ProviderError, match="budget_exceeded"):
        client.complete("Synthetic contract question", Output)
    assert not calls and client.ledger.snapshot()["entries"] == 0


def test_sdk_envelope_validation_failure_preserves_captured_usage(tmp_path):
    import httpx
    from openai import APIResponseValidationError

    original = '{"text":"answer"}'

    def parse():
        response = httpx.Response(200, request=httpx.Request("POST", "https://example.test/responses"))
        raise APIResponseValidationError(response=response, body={"private": "private-parser-body"})

    raw_response = SimpleNamespace(status_code=200,
        http_response=SimpleNamespace(json=lambda: http_response_payload(original)), parse=parse)
    sdk = SimpleNamespace(responses=SimpleNamespace(with_raw_response=SimpleNamespace(parse=lambda **kwargs: raw_response)))
    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=sdk)
    with pytest.raises(ProviderError) as caught:
        client.complete("Synthetic contract question", Output)
    assert caught.value.code == "invalid_output"
    usage = caught.value.usage[0]
    assert usage.returned_model == "resolved-model" and usage.raw_output == original
    assert usage.input_tokens == 50 and usage.output_tokens == 20
    assert usage.cost_eur == pytest.approx(client.ledger.snapshot()["spent_or_reserved_eur"])
    assert caught.value.raw_payload == {"model_text": original}
    assert "private-parser-body" not in str(caught.value) and "private-parser-body" not in str(usage.to_json_record())


def test_persistent_budget_separates_experiment_and_validation(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(path, priced())
    reservation = ledger.reserve(15, model="test", request_hash="hash")
    ledger.settle(reservation, None)
    reopened = BudgetLedger(path, priced())
    with pytest.raises(ProviderError, match="budget_exceeded"):
        reopened.reserve(2, model="test", request_hash="another")
    validation = BudgetLedger(path, priced(), purpose="validation")
    validation.reserve(4, model="test", request_hash="validation")
    with pytest.raises(ProviderError, match="budget_exceeded"):
        validation.reserve(0.1, model="test", request_hash="too_much")
    assert validation.snapshot()["spent_or_reserved_eur"] == 19


def test_settlement_releases_unused_reservation(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json", priced())
    request = ledger.reserve(10, model="test", request_hash="hash")
    ledger.settle(request, 0.25)
    assert ledger.snapshot()["spent_or_reserved_eur"] == 0.25


def test_utopia_403_is_terminal_and_secret_free():
    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(status_code=403, text="private-token")

    session = Session()
    client = UtopiaClient(ProviderConfig(), api_key="private-token", session=session)
    with pytest.raises(ProviderError) as failure:
        client.complete("question", Output)
    assert failure.value.code == "http_403"
    assert session.calls == 1
    assert "private-token" not in str(failure.value)
    assert "private-token" not in str(client.identity)


def test_utopia_retries_transient_failures_and_validates_schema(monkeypatch):
    monkeypatch.setattr("legal_rag.providers.chat.time.sleep", lambda seconds: None)

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls < 3:
                return SimpleNamespace(status_code=503)
            return SimpleNamespace(status_code=200, json=lambda: {"model": kwargs["json"]["model"], "message": {"content": '{"text":"answer"}'}, "prompt_eval_count": 12, "eval_count": 3})

    client = UtopiaClient(ProviderConfig(), api_key="test", session=Session())
    result, usage = client.complete("question", Output)
    assert result.text == "answer"
    assert usage.returned_model == client.config.model


def test_openai_uses_responses_and_accounts_actual_usage(tmp_path):
    class Responses:
        def parse(self, **kwargs):
            self.arguments = kwargs
            return SimpleNamespace(status="completed", model="resolved-model", usage=SimpleNamespace(input_tokens=50, output_tokens=20), output_parsed=Output(text="answer"))

    responses = Responses()
    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=response_sdk(responses))
    output, usage = client.complete("question", Output)
    assert output.text == "answer"
    assert responses.arguments["store"] is False
    assert responses.arguments["truncation"] == "disabled"
    assert usage.returned_model == "resolved-model"
    assert usage.cost_eur == pytest.approx(0.000015)
    assert usage.ledger_entry_id and usage.cost_basis == "conservative_estimate"
    assert client.ledger.snapshot()["spent_or_reserved_eur"] == usage.cost_eur


def test_openai_timeout_keeps_reservation_for_each_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("legal_rag.providers.chat.time.sleep", lambda seconds: None)

    class APITimeoutError(Exception):
        pass

    class Responses:
        def parse(self, **kwargs):
            raise APITimeoutError("private-key")

    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=response_sdk(Responses()))
    with pytest.raises(ProviderError, match="provider_error") as failure:
        client.complete("question", Output)
    summary = client.ledger.snapshot()
    assert summary["entries"] == 3
    assert summary["spent_or_reserved_eur"] > 0
    assert len(failure.value.usage) == 3
    assert sum(usage.cost_eur for usage in failure.value.usage) == pytest.approx(summary["spent_or_reserved_eur"])
    assert all(usage.ledger_entry_id for usage in failure.value.usage)


def test_openai_refusal_is_exposed_and_charged(tmp_path):
    response = SimpleNamespace(status="completed", model="test", usage=SimpleNamespace(input_tokens=10, output_tokens=5), output_parsed=None)
    sdk = response_sdk(SimpleNamespace(parse=lambda **kwargs: response))
    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=sdk)
    with pytest.raises(ProviderError, match="refusal") as failure:
        client.complete("question", Output)
    assert len(failure.value.usage) == 1
    assert client.ledger.snapshot()["spent_or_reserved_eur"] > 0


def test_openai_requires_explicit_prices_before_calling(tmp_path):
    with pytest.raises(ProviderError, match="prices_missing"):
        OpenAIClient(ProviderConfig(provider="openai"), ledger_path=tmp_path / "ledger.json", sdk=object())


def test_stale_prices_and_future_or_old_fx_prevent_paid_calls(tmp_path):
    for config in (priced(price_date=(date.today() - timedelta(days=1)).isoformat()),
                   priced(fx_date=(date.today() - timedelta(days=5)).isoformat()),
                   priced(fx_date=(date.today() + timedelta(days=1)).isoformat())):
        with pytest.raises(ProviderError, match="prices_stale"):
            BudgetLedger(tmp_path / "ledger.json", config)
    assert not (tmp_path / "ledger.json").exists()


def test_cost_bound_violation_blocks_later_runs_persistently(tmp_path):
    response = SimpleNamespace(status="completed", model="test", usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=10), output_parsed=Output(text="answer"))
    path = tmp_path / "ledger.json"
    client = OpenAIClient(priced(), ledger_path=path, sdk=response_sdk(SimpleNamespace(parse=lambda **kwargs: response)))
    with pytest.raises(ProviderError, match="cost_bound_exceeded"):
        client.complete("question", Output)
    reopened = BudgetLedger(path, priced())
    with pytest.raises(ProviderError, match="budget_review_required"):
        reopened.preflight()
    with pytest.raises(ProviderError, match="budget_review_required"):
        reopened.reserve(.01, model="test", request_hash="later")


def test_full_prompt_and_schema_limit_is_checked_before_dispatch(tmp_path):
    class Responses:
        def parse(self, **kwargs):
            pytest.fail("An oversized prompt must never reach the provider")

    client = OpenAIClient(priced(max_input_tokens=3000), ledger_path=tmp_path / "ledger.json", sdk=response_sdk(Responses()))
    with pytest.raises(ProviderError, match="input_limit"):
        client.complete("question " * 500, Output)
    assert client.ledger.snapshot()["entries"] == 0


def test_missing_returned_model_is_not_assumed_to_match_the_request(tmp_path):
    response = SimpleNamespace(status="completed", usage=SimpleNamespace(input_tokens=10, output_tokens=5), output_parsed=Output(text="answer"))
    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=response_sdk(SimpleNamespace(parse=lambda **kwargs: response)))
    _, usage = client.complete("question", Output)
    assert usage.returned_model == "unknown"
    session = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"message": {"content": '{"text":"answer"}'}}))
    _, utopia_usage = UtopiaClient(ProviderConfig(), api_key="test", session=session).complete("question", Output)
    assert utopia_usage.returned_model == "unknown"


def test_bge_limit_includes_special_tokens_and_never_truncates():
    from qdrant_client import models

    class Model:
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(range(len(text.split()) + (2 if add_special_tokens else 0))))

        def encode_corpus(self, texts, **kwargs):
            return {"dense_vecs": [[1, 0] for _ in texts], "lexical_weights": [{"1": 1} for _ in texts]}

    embedder = BgeEmbedder()
    embedder._model = Model()
    assert embedder.embed_hybrid(["word " * 510])[1] == [models.SparseVector(indices=[1], values=[1])]
    with pytest.raises(ValueError, match="implicit truncation"):
        embedder.embed_hybrid(["word " * 511])
    with pytest.raises(ValueError, match="MPS"):
        BgeEmbedder(dtype="float16")
    assert BgeEmbedder(device="mps", dtype="float16").identity["dtype"] == "float16"


def test_embedding_identity_reads_optional_package_metadata_without_loading(monkeypatch):
    import builtins
    from importlib import metadata
    from legal_rag.providers.embeddings import BM25Embedder

    def installed_version(name):
        if name == "torch":
            raise metadata.PackageNotFoundError(name)
        return "fixture-v1"

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "FlagEmbedding", "transformers", "fastembed"}:
            raise AssertionError("Identity inspection must not import optional encoding libraries")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("legal_rag.providers.embeddings.metadata.version", installed_version)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    dense, sparse = BgeEmbedder(), BM25Embedder()
    before = dense.preflight()
    assert before["libraries"] == {"FlagEmbedding": "fixture-v1", "transformers": "fixture-v1", "torch": "unavailable"}
    assert before["adapter_schema_version"] == "bge-adapter-v1"
    assert sparse.preflight()["libraries"] == {"fastembed": "fixture-v1"}
    monkeypatch.setattr("legal_rag.providers.embeddings.metadata.version", lambda name: "fixture-v2")
    assert dense.identity != before
    assert sparse.identity["adapter_schema_version"] == "bm25-adapter-v1"
    assert dense._model is None and sparse._model is None


@pytest.mark.parametrize("failure", [None, RuntimeError("device failure"), IndexError("masked adjustment failure")])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("query", [False, True])
def test_local_inference_stops_without_fallback_and_cleans_mps_buffers(monkeypatch, device, failure, query):
    import sys

    cleared, calls = [], []
    torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                            mps=SimpleNamespace(empty_cache=lambda: cleared.append(True)))
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Model:
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [1, 2, 3])

        def encode_corpus(self, texts, **kwargs):
            calls.append(kwargs)
            if failure:
                raise failure
            return {"dense_vecs": [[1, 0]], "lexical_weights": [{"1": 1}]}

        encode_queries = encode_corpus

    embedder = BgeEmbedder(device=device)
    embedder._model = Model()
    if failure:
        with pytest.raises(ProviderError) as caught:
            embedder.embed_hybrid(["source text"], query=query)
        assert caught.value.code == "local_inference_error"
    else:
        assert embedder.embed_hybrid(["source text"], query=query)[0] == [[1, 0]]
    expected = {"batch_size": 2, "return_dense": True, "return_sparse": True, "return_colbert_vecs": False}
    if device == "mps":
        expected["pad_to_multiple_of"] = 32
    assert calls == [expected]
    assert embedder.identity["padding_multiple"] == (32 if device == "mps" else None)
    assert cleared == ([True] if device == "mps" else [])


def test_utopia_embedding_403_is_terminal_without_fallback():
    session = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=403))
    with pytest.raises(ProviderError, match="http_403"):
        UtopiaEmbedder("test", api_key="secret", session=session).embed_documents(["text"])


@pytest.mark.parametrize("texts,overrides,code", [
    ([], {}, "embedding_input_invalid"),
    ([""], {}, "embedding_input_invalid"),
    ([" \n\t"], {}, "embedding_input_invalid"),
    (["x"] * 2049, {}, "embedding_batch_limit"),
    (["x" * 6145], {}, "input_limit"),
    (["é" * 3073], {}, "input_limit"),
    (["x" * 6000] * 50, {}, "embedding_batch_limit"),
    (["abc"], {"max_input_tokens": 2050}, "input_limit"),
])
def test_openai_embedding_limits_precede_ledger_and_api(tmp_path, monkeypatch, texts, overrides, code):
    path = tmp_path / "ledger.json"
    embedder = OpenAIEmbedder(priced(**overrides), ledger_path=path, sdk=object())
    monkeypatch.setattr(embedder.ledger, "preflight", lambda: pytest.fail("Invalid input must not touch the ledger"))
    with pytest.raises(ProviderError) as caught:
        embedder.embed_documents(texts)
    assert caught.value.code == code
    assert not path.exists() and not path.with_suffix(".json.lock").exists()


@pytest.mark.parametrize("texts", [
    ["é" * 3072],
    ["x"] * 2048,
    ["x" * 6144] * 48 + ["x" * 3040],
])
def test_openai_embedding_exact_conservative_boundaries_are_accepted(tmp_path, texts):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1., 0.]) for i in range(len(kwargs["input"]))],
                               usage=SimpleNamespace(total_tokens=1))

    sdk = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    embedder = OpenAIEmbedder(priced(), ledger_path=tmp_path / "ledger.json", sdk=sdk)
    assert len(embedder.embed_documents(texts)) == len(texts)
    assert len(calls) == 1 and embedder.ledger.snapshot()["entries"] == 1


@pytest.mark.parametrize("provider", ["utopia", "openai"])
@pytest.mark.parametrize("completed", [True, False])
def test_chat_preserves_original_model_json_without_runtime_key(tmp_path, provider, completed):
    original = ' \n { "text" : "answer" } \t\n'
    if provider == "utopia":
        session = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {
            "model": "SLURM.gpt-oss:120b", "message": {"content": original}, "done": completed}))
        client = UtopiaClient(ProviderConfig(), api_key="runtime-private-key", session=session)
    else:
        response = SimpleNamespace(status="completed" if completed else "incomplete", model="test", output_text=original,
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=5), output_parsed=Output(text="answer"))
        client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", api_key="runtime-private-key",
                              sdk=response_sdk(SimpleNamespace(parse=lambda **kwargs: response)))
    if completed:
        parsed, usage = client.complete("question", Output)
        assert parsed.text == "answer"
    else:
        with pytest.raises(ProviderError) as caught:
            client.complete("question", Output)
        assert caught.value.code == "truncated_output"
        usage = caught.value.usage[0]
        assert caught.value.raw_payload == {"model_text": original}
    assert usage.raw_output == original
    assert "runtime-private-key" not in str(usage.to_json_record())
    assert "authorization" not in usage.to_json_record()


def test_utopia_rejects_wrong_returned_model_without_retry_and_preserves_output():
    original = ' { "text" : "answer" }\n'
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: {"model": "DIFFERENT-MODEL", "message": {"content": original},
                                                               "prompt_eval_count": 10, "eval_count": 5})

    client = UtopiaClient(ProviderConfig(), api_key="runtime-private-key", session=SimpleNamespace(post=post))
    with pytest.raises(ProviderError) as caught:
        client.complete("question", Output)
    assert caught.value.code == "model_mismatch" and len(calls) == 1
    assert caught.value.usage[0].returned_model == "DIFFERENT-MODEL"
    assert caught.value.usage[0].raw_output == original
    assert caught.value.raw_payload == {"model_text": original}
    assert "runtime-private-key" not in str(caught.value.usage[0].to_json_record())


def test_utopia_requires_explicit_context_and_rejects_oversize_prompt_before_dispatch():
    session = SimpleNamespace(post=lambda *args, **kwargs: pytest.fail("Invalid context configuration must not be dispatched"))
    for config, code in ((ProviderConfig(context_window_tokens=None), "context_unverified"),
                         (ProviderConfig(context_window_tokens=2048, max_output_tokens=1024), "context_limit")):
        with pytest.raises(ProviderError) as caught:
            UtopiaClient(config, api_key="test", session=session).complete("question", Output)
        assert caught.value.code == code and not caught.value.usage


def test_model_inspection_exports_only_safe_explicit_metadata():
    session = SimpleNamespace(
        post=lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"model_info": {"llama.context_length": 32768}, "capabilities": ["completion"], "template": "private-template"}),
        get=lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"models": [{"model": "SLURM.gpt-oss:120b", "digest": "sha256:verified"}]}))
    inspected = UtopiaClient(ProviderConfig(), api_key="secret", session=session).inspect_model()
    assert inspected.context_window_tokens == 32768 and inspected.digest == "sha256:verified"
    assert inspected.returned_model == "SLURM.gpt-oss:120b"
    assert "private-template" not in str(inspected.to_json_record()) and "secret" not in str(inspected.to_json_record())


def test_total_context_window_is_checked_in_addition_to_input_limit(tmp_path):
    client = OpenAIClient(priced(context_window_tokens=6000, max_output_tokens=4096), ledger_path=tmp_path / "ledger.json", sdk=object())
    with pytest.raises(ProviderError, match="context_limit"):
        client.complete("question", Output)
    assert client.ledger.snapshot()["entries"] == 0


def test_model_inspection_rejects_an_exposed_different_identity():
    session = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=200,
                              json=lambda: {"model": "unrequested-model", "capabilities": ["completion"]}))
    with pytest.raises(ProviderError, match="model_mismatch"):
        UtopiaClient(ProviderConfig(), api_key="secret", session=session).inspect_model()


def test_openai_identity_stays_identical_before_and_after_preflight(tmp_path):
    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=SimpleNamespace(base_url="ignored-test-url"))
    before = client.identity
    client.preflight()
    assert before == client.identity
    assert before["base_url"] == "https://api.openai.com/v1"


def test_scientific_identity_is_independent_of_updated_quotes_and_ledger_path(tmp_path):
    left = OpenAIClient(priced(), ledger_path=tmp_path / "one.json", sdk=object())
    right = OpenAIClient(priced(input_usd_per_million=.125), ledger_path=tmp_path / "two.json", sdk=object())
    assert left.identity == right.identity
    assert left.accounting != right.accounting
    assert "price_date" not in left.identity and "ledger_path" not in left.identity


def test_utopia_sends_explicit_thinking_and_configured_context_window():
    class Session:
        def post(self, *args, **kwargs):
            self.payload = kwargs["json"]
            return SimpleNamespace(status_code=200, json=lambda: {"model": kwargs["json"]["model"], "message": {"content": '{"text":"answer"}'}})

    session = Session()
    config = ProviderConfig(context_window_tokens=4096, max_output_tokens=1024, reasoning_effort="low")
    UtopiaClient(config, api_key="test", session=session).complete("question", Output)
    assert session.payload["think"] == "low" and session.payload["options"]["num_ctx"] == 4096
    UtopiaClient(ProviderConfig(reasoning_effort="none"), api_key="test", session=session).complete("question", Output)
    assert session.payload["think"] is False and session.payload["options"]["num_ctx"] == 32768
    UtopiaClient(ProviderConfig(reasoning_effort=None), api_key="test", session=session).complete("question", Output)
    assert "think" not in session.payload


def test_openai_explicit_none_and_provider_default_reasoning_are_distinct(tmp_path):
    class Responses:
        def parse(self, **kwargs):
            self.arguments = kwargs
            return SimpleNamespace(status="completed", model="test", usage=SimpleNamespace(input_tokens=10, output_tokens=5), output_parsed=Output(text="answer"))

    responses = Responses()
    OpenAIClient(priced(reasoning_effort="none"), ledger_path=tmp_path / "one.json", sdk=response_sdk(responses)).complete("question", Output)
    assert responses.arguments["reasoning"] == {"effort": "none"}
    OpenAIClient(priced(reasoning_effort=None), ledger_path=tmp_path / "two.json", sdk=response_sdk(responses)).complete("question", Output)
    assert "reasoning" not in responses.arguments


def test_success_after_uncertain_retry_includes_retained_reservation_in_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("legal_rag.providers.chat.time.sleep", lambda seconds: None)

    class APITimeoutError(Exception):
        pass

    class Responses:
        calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise APITimeoutError()
            return SimpleNamespace(status="completed", model="test", usage=SimpleNamespace(input_tokens=10, output_tokens=5), output_parsed=Output(text="answer"))

    client = OpenAIClient(priced(), ledger_path=tmp_path / "ledger.json", sdk=response_sdk(Responses()))
    _, usage = client.complete("question", Output)
    assert usage.cost_eur == pytest.approx(client.ledger.snapshot()["spent_or_reserved_eur"])
    assert usage.cost_basis == "conservative_estimate_including_retry_reservations"

from __future__ import annotations

from pathlib import Path

from legal_indexing.sparse import SparseEncoder


def test_sparse_encoder_fit_transform_and_roundtrip(tmp_path: Path) -> None:
    texts = [
        "La procedura amministrativa regionale",
        "Sanzione per violazione dell'articolo",
        "Articolo 3 disciplina la procedura",
    ]
    enc = SparseEncoder(min_token_len=2, stopwords_lang="it")
    enc.fit(texts)

    assert enc.is_fitted
    assert enc.vocab_size > 0

    v1 = enc.transform("procedura articolo")
    v2 = enc.transform("procedura articolo")
    assert v1.indices == v2.indices
    assert v1.values == v2.values
    assert len(v1.indices) > 0

    artifact = tmp_path / "sparse_encoder.json"
    enc.save_json(artifact)
    loaded = SparseEncoder.load_json(artifact)
    v3 = loaded.transform("procedura articolo")
    assert v3.indices == v1.indices
    assert v3.values == v1.values

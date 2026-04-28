from __future__ import annotations

from legal_indexing.chunk_refinement import passage_order_key, refine_chunks_with_diagnostics
from legal_indexing.io import PassageRecord
from legal_indexing.settings import make_chunking_profile


def _passage(
    *,
    passage_id: str,
    article_id: str,
    law_id: str,
    passage_label: str,
    text: str,
    source_chunk_ids: tuple[str, ...],
) -> PassageRecord:
    return PassageRecord(
        passage_id=passage_id,
        article_id=article_id,
        law_id=law_id,
        passage_label=passage_label,
        text=text,
        source_chunk_ids=source_chunk_ids,
        law_date="2025-01-01",
        law_number=1,
        law_title="Legge test",
        law_status="current",
        article_is_abrogated=False,
        index_views=("historical", "current"),
        related_law_ids=("law:x",),
        relation_types=("REFERS_TO",),
        inbound_law_ids=(),
        outbound_law_ids=("law:x",),
        status_confidence=0.7,
        status_evidence=tuple(),
    )


def test_passage_order_key_is_deterministic() -> None:
    labels = ["c2", "intro", "c1", "c1.lit_b", "c1.lit_a", "lit_a", "c1bis"]
    ordered = sorted(labels, key=passage_order_key)
    assert ordered == ["intro", "c1", "c1.lit_a", "c1.lit_b", "c1bis", "c2", "lit_a"]


def test_refine_chunks_merges_splits_and_links_neighbors() -> None:
    article_id = "law:test#art:1"
    law_id = "law:test"
    long_text = " ".join([f"w{i}" for i in range(80)])

    passages = [
        _passage(
            passage_id=f"{article_id}#p:intro",
            article_id=article_id,
            law_id=law_id,
            passage_label="intro",
            text="Titolo",
            source_chunk_ids=("s0",),
        ),
        _passage(
            passage_id=f"{article_id}#p:c1",
            article_id=article_id,
            law_id=law_id,
            passage_label="c1",
            text="uno due tre quattro cinque sei sette otto nove",
            source_chunk_ids=("s1",),
        ),
        _passage(
            passage_id=f"{article_id}#p:c2",
            article_id=article_id,
            law_id=law_id,
            passage_label="c2",
            text=long_text,
            source_chunk_ids=("s2",),
        ),
    ]

    profile = make_chunking_profile(
        "balanced", min_words_merge=20, max_words_split=30, overlap_words_split=5
    )

    chunks_a, diag = refine_chunks_with_diagnostics(passages, {article_id: 0}, profile)
    chunks_b, _ = refine_chunks_with_diagnostics(passages, {article_id: 0}, profile)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert diag.merged_units >= 1
    assert diag.split_units >= 1

    merged_chunk = next(c for c in chunks_a if c.source_passage_labels == ("intro", "c1"))
    assert merged_chunk.article_order_in_law == 0
    assert merged_chunk.article_chunk_order == 0

    # c2 should be split in more than one chunk with neighbor links.
    split_chunks = [c for c in chunks_a if c.source_passage_labels == ("c2",)]
    assert len(split_chunks) >= 2
    assert split_chunks[0].next_chunk_id is not None
    assert split_chunks[-1].prev_chunk_id is not None

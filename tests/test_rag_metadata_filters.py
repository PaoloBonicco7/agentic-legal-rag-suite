from __future__ import annotations

from legal_indexing.rag_runtime.config import (
    AdvancedMetadataFilteringConfig,
    QdrantPayloadFieldMap,
)
from legal_indexing.law_references import LawCatalog, resolve_law_references
from legal_indexing.rag_runtime.metadata_filters import (
    build_metadata_filter,
    resolve_metadata_filter_decision,
)


def test_hybrid_metadata_filter_infers_view_relations_and_year() -> None:
    cfg = AdvancedMetadataFilteringConfig(mode="hybrid", enable_heuristics=True)
    decision = resolve_metadata_filter_decision(
        "Quale legge abrogata nel 2010 ha modificato l'articolo law:test#art:1?",
        config=cfg,
        default_view="none",
    )
    assert decision.view == "historical"
    assert decision.year_from == 2010
    assert decision.year_to == 2010
    assert "ABROGATED_BY" in set(decision.relation_types)
    assert "law:test#art:1" in set(decision.article_ids)
    assert "law:test" in set(decision.law_ids)
    assert decision.applied_heuristics

    qfilter = build_metadata_filter(QdrantPayloadFieldMap(), decision)
    assert qfilter is not None
    assert qfilter.must is not None
    assert len(qfilter.must) >= 1


def test_explicit_metadata_overrides_heuristics() -> None:
    cfg = AdvancedMetadataFilteringConfig(
        mode="explicit_only",
        explicit_view="current",
        explicit_law_status="current",
        explicit_law_ids=("law:explicit",),
        explicit_relation_types=("AMENDS",),
        explicit_year_from=2020,
        explicit_year_to=2021,
    )
    decision = resolve_metadata_filter_decision(
        "domanda storica abrogata nel 1990",
        config=cfg,
        default_view="none",
    )
    assert decision.view == "current"
    assert decision.law_status == "current"
    assert decision.law_ids == ("law:explicit",)
    assert decision.relation_types == ("AMENDS",)
    assert decision.year_from == 2020
    assert decision.year_to == 2021
    assert decision.applied_heuristics == tuple()


def test_hybrid_metadata_filter_uses_resolved_references_when_available() -> None:
    cfg = AdvancedMetadataFilteringConfig(mode="hybrid", enable_heuristics=True)
    catalog = LawCatalog(
        law_ids=("vda:lr:2000-01-25:5",),
        by_year_number={(2000, 5): ("vda:lr:2000-01-25:5",)},
        by_date_number={("2000-01-25", 5): "vda:lr:2000-01-25:5"},
        article_ids_by_law={"vda:lr:2000-01-25:5": ("vda:lr:2000-01-25:5#art:12",)},
        article_labels_by_law={"vda:lr:2000-01-25:5": ("12",)},
    )
    resolved = resolve_law_references(
        "Legge regionale 25 gennaio 2000, n. 5 - Art. 12",
        catalog=catalog,
    )

    decision = resolve_metadata_filter_decision(
        "Legge regionale 25 gennaio 2000, n. 5 - Art. 12",
        config=cfg,
        default_view="none",
        resolved_references=resolved,
    )

    assert decision.law_ids == ("vda:lr:2000-01-25:5",)
    assert decision.article_ids == ("vda:lr:2000-01-25:5#art:12",)
    assert "law_reference_resolved" in set(decision.applied_heuristics)

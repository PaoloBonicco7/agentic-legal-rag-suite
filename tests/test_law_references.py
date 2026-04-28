from __future__ import annotations

import json
from pathlib import Path

from legal_indexing.law_references import (
    LawCatalog,
    build_law_catalog,
    compute_eval_reference_coverage,
    resolve_law_references,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "dataset"
    laws = [
        {"law_id": "vda:lr:2000-01-25:5"},
        {"law_id": "vda:lr:2019-10-08:16"},
    ]
    articles = [
        {
            "law_id": "vda:lr:2000-01-25:5",
            "article_id": "vda:lr:2000-01-25:5#art:12",
            "article_label_norm": "12",
        },
        {
            "law_id": "vda:lr:2019-10-08:16",
            "article_id": "vda:lr:2019-10-08:16#art:4bis",
            "article_label_norm": "4bis",
        },
    ]
    _write_jsonl(ds / "laws.jsonl", laws)
    _write_jsonl(ds / "articles.jsonl", articles)
    return ds


def test_resolve_law_references_from_human_readable_citation(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    catalog: LawCatalog = build_law_catalog(dataset_dir)

    out = resolve_law_references(
        "Legge regionale 25 gennaio 2000, n. 5 - Art. 12",
        catalog=catalog,
    )
    assert out.law_ids == ("vda:lr:2000-01-25:5",)
    assert out.article_ids == ("vda:lr:2000-01-25:5#art:12",)
    assert out.unresolved_mentions == tuple()


def test_resolve_law_references_from_num_year_variant(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    catalog: LawCatalog = build_law_catalog(dataset_dir)

    out = resolve_law_references("Ai sensi della L.R. n. 16/2019, art. 4 bis", catalog=catalog)
    assert out.law_ids == ("vda:lr:2019-10-08:16",)
    assert out.article_ids == ("vda:lr:2019-10-08:16#art:4bis",)


def test_eval_reference_coverage_report(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    catalog: LawCatalog = build_law_catalog(dataset_dir)

    report = compute_eval_reference_coverage(
        catalog=catalog,
        references=[
            "Legge regionale 25 gennaio 2000, n. 5 - Art. 12",
            "Legge regionale 8 ottobre 2019, n. 16 - Art. 4 bis",
            "Legge regionale 7 ottobre 2024, n. 19 - Art. 2",
        ],
    )

    assert report.references_total == 3
    assert report.references_with_any_law == 3
    assert report.references_resolved == 2
    assert report.coverage is not None
    assert round(float(report.coverage), 4) == round(2 / 3, 4)
    assert len(report.missing_references_sample) == 1

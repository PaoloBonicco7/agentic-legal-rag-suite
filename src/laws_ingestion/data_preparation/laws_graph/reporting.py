from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


_ABROGATED_BY_EVIDENCE_RE = re.compile(r"\babrogat[oa]\b.*\bdall[ao]\b|\blegge\s+abrogata\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_duplicates(records: list[dict], id_field: str) -> int:
    seen: set[str] = set()
    dupes = 0
    for rec in records:
        rid = str(rec.get(id_field) or "")
        if not rid:
            continue
        if rid in seen:
            dupes += 1
        else:
            seen.add(rid)
    return dupes


def relation_coverage_abrogated_by(edges_raw: list[dict]) -> tuple[int, int, float]:
    total = 0
    hits = 0
    for e in edges_raw:
        ev = str(e.get("evidence") or e.get("evidence_text") or "")
        if not _ABROGATED_BY_EVIDENCE_RE.search(ev):
            continue
        total += 1
        if str(e.get("relation_type") or "") == "ABROGATED_BY":
            hits += 1
    cov = (hits / total) if total else 1.0
    return total, hits, cov


def build_quality_metrics(
    *,
    laws: list[dict],
    articles: list[dict],
    passages: list[dict],
    notes: list[dict],
    edges_raw: list[dict],
    edges_clean: list[dict],
    events: list[dict],
    chunks: list[dict],
    unresolved_refs_new: int,
    unresolved_refs_baseline: int,
) -> dict:
    total_abrog_cases, abrog_hits, abrog_cov = relation_coverage_abrogated_by(edges_raw)

    d_article = count_duplicates(articles, "article_id")
    d_passage = count_duplicates(passages, "passage_id")
    d_chunk = count_duplicates(chunks, "chunk_id")
    self_loops_clean = sum(1 for e in edges_clean if bool(e.get("is_self_loop")))

    gates = {
        "duplicate_article_id_passage_id_chunk_id_zero": (d_article == 0 and d_passage == 0 and d_chunk == 0),
        "self_loops_edges_zero_in_clean": (self_loops_clean == 0),
        "unresolved_refs_not_worse": (int(unresolved_refs_new) <= int(unresolved_refs_baseline)),
        "abrogated_by_coverage_ge_99": (abrog_cov >= 0.99),
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "counts": {
            "laws": len(laws),
            "articles": len(articles),
            "passages": len(passages),
            "notes": len(notes),
            "edges_raw": len(edges_raw),
            "edges_clean": len(edges_clean),
            "events": len(events),
            "chunks": len(chunks),
        },
        "duplicates": {
            "article_id": d_article,
            "passage_id": d_passage,
            "chunk_id": d_chunk,
        },
        "self_loops": {
            "raw": sum(1 for e in edges_raw if bool(e.get("is_self_loop"))),
            "clean": self_loops_clean,
        },
        "unresolved_refs": {
            "baseline": int(unresolved_refs_baseline),
            "new": int(unresolved_refs_new),
            "delta": int(unresolved_refs_new) - int(unresolved_refs_baseline),
        },
        "abrogated_by_coverage": {
            "phrase_cases": total_abrog_cases,
            "hits": abrog_hits,
            "coverage": abrog_cov,
        },
        "relation_type_distribution_clean": dict(Counter(str(e.get("relation_type") or "") for e in edges_clean)),
        "status_distribution": dict(Counter(str(l.get("status") or "") for l in laws)),
        "gates": gates,
        "ready_to_embedding": all(gates.values()),
    }


def write_quality_markdown(path: Path, metrics: dict) -> None:
    lines = [
        "# Step 08 - Quality Report",
        "",
        f"- Ready to embedding: **{metrics.get('ready_to_embedding')}**",
        f"- Timestamp (UTC): `{metrics.get('created_at')}`",
        "",
        "## Gate Results",
    ]
    for gate, ok in (metrics.get("gates") or {}).items():
        lines.append(f"- `{gate}`: **{ok}**")

    lines.extend(
        [
            "",
            "## Counts",
            f"- laws: {metrics['counts']['laws']}",
            f"- articles: {metrics['counts']['articles']}",
            f"- passages: {metrics['counts']['passages']}",
            f"- notes: {metrics['counts']['notes']}",
            f"- edges_raw: {metrics['counts']['edges_raw']}",
            f"- edges_clean: {metrics['counts']['edges_clean']}",
            f"- events: {metrics['counts']['events']}",
            f"- chunks: {metrics['counts']['chunks']}",
            "",
            "## Key Metrics",
            f"- duplicate article_id: {metrics['duplicates']['article_id']}",
            f"- duplicate passage_id: {metrics['duplicates']['passage_id']}",
            f"- duplicate chunk_id: {metrics['duplicates']['chunk_id']}",
            f"- self loops raw: {metrics['self_loops']['raw']}",
            f"- self loops clean: {metrics['self_loops']['clean']}",
            f"- unresolved refs baseline/new: {metrics['unresolved_refs']['baseline']} / {metrics['unresolved_refs']['new']}",
            f"- ABROGATED_BY coverage: {metrics['abrogated_by_coverage']['coverage']:.4f}",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

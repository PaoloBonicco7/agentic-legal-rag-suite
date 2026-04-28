from __future__ import annotations

from collections import defaultdict


def deduplicate_records(records: list[dict], id_field: str) -> tuple[list[dict], int]:
    out: list[dict] = []
    seen: dict[str, int] = {}
    collisions = 0

    for rec in records:
        rid = str(rec.get(id_field) or "")
        if rid not in seen:
            seen[rid] = 1
            out.append(rec)
            continue

        collisions += 1
        seen[rid] += 1
        new_id = f"{rid}~dup{seen[rid]}"
        rec2 = dict(rec)
        rec2[id_field] = new_id

        if id_field == "article_id":
            if rec2.get("passage_id") and isinstance(rec2.get("passage_id"), str):
                rec2["passage_id"] = rec2["passage_id"].replace(rid, new_id, 1)
            if rec2.get("chunk_id") and isinstance(rec2.get("chunk_id"), str):
                rec2["chunk_id"] = rec2["chunk_id"].replace(rid, new_id, 1)

        if id_field == "chunk_id":
            rec2["chunk_id"] = new_id

        out.append(rec2)

    return out, collisions


def enrich_chunks_with_views(chunks: list[dict], statuses: dict[str, dict], edges_clean: list[dict]) -> list[dict]:
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)
    relation_types: dict[str, set[str]] = defaultdict(set)

    for e in edges_clean:
        src = str(e.get("src_law_id") or "")
        dst = str(e.get("dst_law_id") or "")
        if not src or not dst:
            continue
        outbound[src].add(dst)
        inbound[dst].add(src)
        relation_types[src].add(str(e.get("relation_type") or ""))

    out: list[dict] = []
    for c in chunks:
        law_id = str(c.get("law_id") or "")
        status_info = statuses.get(law_id) or {}
        law_status = status_info.get("status") or c.get("law_status") or "unknown"
        article_is_abrogated = bool(c.get("article_is_abrogated") or False)

        index_views = ["historical"]
        if law_status == "current" and not article_is_abrogated:
            index_views.append("current")

        rec = dict(c)
        rec["law_status"] = law_status
        rec["status_confidence"] = float(status_info.get("status_confidence") or 0.0)
        rec["status_evidence"] = status_info.get("status_evidence") or []
        rec["inbound_law_ids"] = sorted(inbound.get(law_id, set()))
        rec["outbound_law_ids"] = sorted(outbound.get(law_id, set()))
        rec["relation_types"] = sorted(set(rec.get("relation_types") or []) | relation_types.get(law_id, set()))
        rec["index_views"] = index_views
        out.append(rec)

    out.sort(key=lambda x: str(x.get("chunk_id") or ""))
    return out

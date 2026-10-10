"""Build immutable, validated corpus artifacts from one source snapshot."""

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import Field

from ..artifacts import code_identity, file_hash, new_output, read_json, read_jsonl, utc_now, write_json, write_jsonl
from ..contracts import SCHEMA_VERSION, Block, Chunk, Document, Event, Record, Relation, SourceRecord, Unit
from ..tokens import TokenCounter
from .acquisition import SnapshotManifest
from .chunking import CHUNKING_VERSION, build_chunks
from .graph import GRAPH_VERSION, extract_relations, graph_nodes, reference_registry
from .parser import PARSER_VERSION, body_text, parse_source, span_text, substantive
from .status import STATUS_VERSION, assess_status


class CorpusQuality(Record):
    schema_version: str = SCHEMA_VERSION
    gates: dict[str, bool]
    distributions: dict[str, dict[str, int]]
    counts: dict[str, int]
    warnings: list[dict[str, str]] = Field(default_factory=list)
    ready_for_indexing: bool


class CorpusManifest(Record):
    schema_version: str = SCHEMA_VERSION
    snapshot_id: str
    snapshot_date: str | None
    created_at: str
    source_manifest_sha256: str
    source_catalog_sha256: str
    code: dict
    versions: dict[str, str]
    tokenizer: dict[str, str]
    config: dict[str, int]
    counts: dict[str, int]
    files: dict[str, str]
    ready_for_indexing: bool
    quality: dict[str, bool]
    limitations: list[str]


def _quality(documents: list[Document], blocks: list[Block], units: list[Unit], events: list[Event], relations: list[Relation], chunks: list[Chunk], counter: TokenCounter, max_tokens: int) -> CorpusQuality:
    by_block = {b.block_id: b for b in blocks}
    unit_by_id = {u.unit_id: u for u in units}
    doc_by_id = {d.document_id: d for d in documents}
    nodes = set(unit_by_id) | set(doc_by_id)
    covered = {span.block_id for unit in units for span in unit.spans}
    valid_spans = all(span.block_id in by_block and span.start <= span.end <= len(by_block[span.block_id].text) for row in units + chunks + events for span in ([row.span] if isinstance(row, Event) else row.spans))
    valid_ids = all(len({getattr(row, attribute) for row in rows}) == len(rows) for rows, attribute in [(documents, "document_id"), (blocks, "block_id"), (units, "unit_id"), (events, "event_id"), (relations, "relation_id"), (chunks, "chunk_id")])
    evidence_matches = valid_spans and all(by_block[event.span.block_id].text[event.span.start:event.span.end] == event.evidence for event in events)
    target_scopes = all(event.resolution != "confirmed" or event.target_id in nodes and ((event.scope == "law" and event.target_id in doc_by_id) or (event.scope in {"article", "paragraph"} and unit_by_id.get(event.target_id) is not None and unit_by_id[event.target_id].kind == event.scope)) for event in events)
    mixed_status = any(chunk.status != "ceased_confirmed" and any(unit_by_id[uid].status == "ceased_confirmed" for uid in chunk.unit_ids) for chunk in chunks)
    gates = {
        "documents_present": bool(documents),
        "chunks_present": bool(chunks),
        "unique_ids": valid_ids,
        "all_blocks_classified": all(b.region != "unassigned" for b in blocks),
        "significant_body_preserved": all(b.block_id in covered or not substantive(body_text(b)) for b in blocks if b.region in {"body", "body_quote"}),
        "safe_spans": valid_spans,
        "exact_unit_spans": valid_spans and all(span_text(u.spans, by_block) == u.text for u in units),
        "exact_chunk_spans": valid_spans and all(span_text(c.spans, by_block) == c.text for c in chunks),
        "event_evidence_matches": evidence_matches,
        "confirmed_event_scope": target_scopes,
        "parents_resolve": all(u.parent_id is None or u.parent_id in unit_by_id for u in units),
        "relations_resolve": all(r.source_id in nodes and (r.target_id is None or r.target_id in nodes) and r.source_id != r.target_id for r in relations),
        "chunk_units_resolve": all(c.document_id in doc_by_id and all(uid in unit_by_id for uid in c.unit_ids) for c in chunks),
        "substantive_chunks": all(substantive(c.text) for c in chunks),
        "no_eligible_ceased_mixture": not mixed_status,
        "complete_embedding_budget": all(counter.count(c.embedding_text) <= max_tokens for c in chunks),
        "no_new_encoding_anomalies": all(d.representation == "legacy" or "invalid_text_encoding" not in d.warnings for d in documents),
    }
    distributions = {"document_status": dict(Counter(d.status for d in documents)), "availability": dict(Counter(d.availability for d in documents)), "unit_status": dict(Counter(u.status for u in units)), "unit_kind": dict(Counter(u.kind for u in units)), "block_region": dict(Counter(b.region for b in blocks)), "chunk_status": dict(Counter(c.status for c in chunks)), "relation_kind": dict(Counter(r.kind for r in relations))}
    distributions["block_exclusion_reason"] = dict(Counter(b.exclusion_reason for b in blocks if b.exclusion_reason))
    counts = {"documents": len(documents), "blocks": len(blocks), "units": len(units), "events": len(events), "relations": len(relations), "chunks": len(chunks), "unresolved_events": sum(e.resolution == "unresolved" for e in events), "unresolved_relations": sum(r.resolution == "unresolved" for r in relations), "encoding_anomalies": sum("invalid_text_encoding" in d.warnings for d in documents), "omitted_annexes": sum("omitted_annex" in d.warnings for d in documents)}
    counts["excluded_blocks"] = sum(b.exclusion_reason is not None for b in blocks)
    warnings = [{"document_id": d.document_id, "warning": warning} for d in documents for warning in d.warnings]
    return CorpusQuality(gates=gates, distributions=distributions, counts=counts, warnings=warnings, ready_for_indexing=all(gates.values()))


def build_corpus(snapshot_dir: Path, output_dir: Path, counter: TokenCounter, target_tokens: int = 384, max_tokens: int = 512) -> dict:
    snapshot_dir, output_dir = Path(snapshot_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Artifact directory already exists: {output_dir}")
    if not 0 < target_tokens <= max_tokens:
        raise ValueError("Expected 0 < target_tokens <= max_tokens")
    source_manifest = SnapshotManifest.model_validate(read_json(snapshot_dir / "manifest.json")).to_json_record()
    if source_manifest["complete"] is not True:
        raise ValueError("Cannot build a corpus from an incomplete acquisition snapshot")
    snapshot_id = source_manifest["snapshot_id"]
    run_code = code_identity()
    snapshot_date = source_manifest.get("acquisition_finished_at")
    catalog_path = snapshot_dir / "catalog.jsonl"
    if file_hash(catalog_path) != source_manifest["catalog_sha256"]:
        raise ValueError("Snapshot catalog hash mismatch")
    catalog = read_jsonl(catalog_path, SourceRecord)
    if len(catalog) != source_manifest["source_count"]:
        raise ValueError("Snapshot catalog count differs from the source manifest")
    if not catalog or len({source.source_id for source in catalog}) != len(catalog):
        raise ValueError("Source catalog must be nonempty with unique source IDs")
    if len({source.path for source in catalog}) != len(catalog):
        raise ValueError("Source catalog paths must be unique")
    root = snapshot_dir.resolve()
    source_paths = {}
    for source in catalog:
        path = Path(source.path)
        if path.is_absolute() or not (snapshot_dir / path).resolve().is_relative_to(root):
            raise ValueError(f"Unsafe source path: {source.path}")
        path = snapshot_dir / path
        if file_hash(path) != source.sha256:
            raise ValueError(f"Source hash mismatch: {source.source_id}")
        source_paths[source.source_id] = path
    documents, blocks, units, events = [], [], [], []
    per_document: dict[str, tuple[list[Block], list[Unit]]] = {}
    for source in sorted(catalog, key=lambda row: row.source_id):
        path = source_paths[source.source_id]
        doc, doc_blocks, doc_units = parse_source(source, snapshot_id, path.read_bytes().decode(source.encoding))
        if doc.document_id in per_document:
            raise ValueError(f"Duplicate legal identity: {doc.document_id}")
        documents.append(doc)
        blocks.extend(doc_blocks)
        units.extend(doc_units)
        events.extend(assess_status(doc, doc_blocks, doc_units, snapshot_date))
        per_document[doc.document_id] = (doc_blocks, doc_units)
    by_identity = {doc.document_id: doc for doc in documents}
    registry = reference_registry(documents, units)
    relations, chunks = [], []
    for doc in documents:
        doc_blocks, doc_units = per_document[doc.document_id]
        relations.extend(extract_relations(doc, doc_blocks, doc_units, by_identity, units, registry=registry))
        chunks.extend(build_chunks(doc, doc_units, counter, target_tokens, max_tokens))
    references_by_span: dict[tuple[str, int], list[Relation]] = defaultdict(list)
    for relation in relations:
        if relation.kind == "references" and relation.span:
            references_by_span[(relation.span.block_id, relation.span.start)].append(relation)
    for event in events:
        refs = references_by_span.get((event.span.block_id, event.span.start), [])
        identities = list(dict.fromkeys(r.target_reference.split("#")[0] for r in refs if r.target_reference.split("#")[0] in by_identity))
        if len(identities) == 1:
            event.related_document_id = identities[0]
    quality = _quality(documents, blocks, units, events, relations, chunks, counter, max_tokens)
    nodes = graph_nodes(documents, units)
    quality.counts["nodes"] = len(nodes)
    new_output(output_dir)
    for name, records in {"documents": documents, "blocks": blocks, "units": units, "events": events, "relations": relations, "chunks": chunks, "nodes": nodes}.items():
        write_jsonl(output_dir / f"{name}.jsonl", records)
    write_json(output_dir / "quality.json", quality)
    files = {path.name: file_hash(path) for path in sorted(output_dir.iterdir())}
    manifest = CorpusManifest(snapshot_id=snapshot_id, snapshot_date=snapshot_date, created_at=utc_now(), source_manifest_sha256=file_hash(snapshot_dir / "manifest.json"), source_catalog_sha256=file_hash(snapshot_dir / "catalog.jsonl"), code=run_code, versions={"parser": PARSER_VERSION, "status": STATUS_VERSION, "graph": GRAPH_VERSION, "chunking": CHUNKING_VERSION}, tokenizer={"name": counter.name, "revision": counter.revision}, config={"target_tokens": target_tokens, "max_tokens": max_tokens}, counts=quality.counts, files=files, ready_for_indexing=quality.ready_for_indexing, quality=quality.gates, limitations=["Source representation and absence of cessation evidence do not certify applicability.", "Historical as-of reconstruction is not implemented."] + (["Legacy capture date is unknown; this snapshot does not establish current freshness."] if snapshot_date is None else []))
    write_json(output_dir / "manifest.json", manifest)
    return manifest.to_json_record()

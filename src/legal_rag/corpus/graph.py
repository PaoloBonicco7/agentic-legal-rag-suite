"""Evidence-backed containment and clause-local legal references."""

import re
from urllib.parse import parse_qs, urlparse

from ..artifacts import digest
from ..contracts import Block, Document, Relation, SourceSpan, Unit
from .parser import LAW_RE, law_identity, normalize_label
from .status import clauses, note_bindings

GRAPH_VERSION = "explicit-graph-2"


def reference_registry(documents: list[Document], units: list[Unit]) -> dict:
    articles: dict[tuple[str, str], list[Unit]] = {}
    for unit in units:
        if unit.kind == "article":
            articles.setdefault((unit.document_id, unit.label), []).append(unit)
    return {"by_number": {(int(doc.document_id.split(":")[2][:4]), int(doc.document_id.split(":")[-1])): doc for doc in documents}, "by_source": {doc.source_id: doc for doc in documents}, "article_targets": articles}


def extract_relations(document: Document, blocks: list[Block], units: list[Unit], documents_by_identity: dict[str, Document], all_units: list[Unit] | None = None, *, registry: dict | None = None) -> list[Relation]:
    all_units = all_units if all_units is not None else units
    by_identity = documents_by_identity
    registry = registry or reference_registry(list(by_identity.values()), all_units)
    by_number, by_source, article_targets = registry["by_number"], registry["by_source"], registry["article_targets"]
    by_block: dict[str, list[Unit]] = {}
    for unit in units:
        for span in unit.spans:
            by_block.setdefault(span.block_id, []).append(unit)
    bindings, duplicates = note_bindings(blocks, units)
    relations, seen = [], set()

    def emit(source_id: str, target_id: str | None, reference: str, kind: str, evidence: str = "", span: SourceSpan | None = None) -> None:
        if target_id == source_id:
            return
        key = (source_id, target_id, reference, kind, span.block_id if span else "", span.start if span else 0)
        if key in seen:
            return
        seen.add(key)
        relations.append(Relation(relation_id="rel:" + digest(str(key))[:24], source_id=source_id, target_id=target_id, target_reference=reference, kind=kind, evidence=evidence, span=span, resolution="confirmed" if target_id else "unresolved"))

    for unit in units:
        emit(unit.parent_id or document.document_id, unit.unit_id, unit.unit_id, "contains")
    current_anchor = None
    for block in blocks:
        if block.anchor and block.anchor.startswith("nota"):
            current_anchor = block.anchor
        if block.region in {"index", "metadata", "heading", "separator", "empty", "omitted"}:
            continue
        candidates = [u for u in by_block.get(block.block_id, []) if u.kind != "article"]
        source_id = candidates[-1].unit_id if candidates else document.document_id
        if block.region == "note" and current_anchor not in duplicates and len(bindings.get(current_anchor, [])) == 1:
            note_units = bindings[current_anchor][0][1]
            if len(note_units) == 1:
                source_id = note_units[0].unit_id
        for clause in clauses(block):
            citations: list[tuple[int, int, str, Document | None]] = []
            for link in block.links:
                if "start" not in link or not (clause.start <= int(link["start"]) < clause.end):
                    continue
                href, label = link.get("href", ""), link.get("text", "")
                if href.startswith("#"):
                    continue
                query = parse_qs(urlparse(href).query)
                target = by_identity.get(law_identity(label))
                number = query.get("numero_legge", [""])[0]
                if target is None and re.fullmatch(r"\d+/\d{2,4}", number):
                    num, year = map(int, number.split("/"))
                    choices = [doc for (yyyy, n), doc in by_number.items() if n == num and (yyyy == year or yyyy % 100 == year)]
                    target = choices[0] if len(choices) == 1 else None
                if target is None and query.get("pk_lr"):
                    pk = query["pk_lr"][0]
                    target = by_source.get(pk) or next((doc for doc in by_identity.values() if doc.source_id.split(":")[-1] == pk), None)
                if number or query.get("pk_lr") or law_identity(label):
                    reference = law_identity(label) or (f"vda:lr:{number}" if number else href)
                    citations.append((int(link["start"]), int(link["end"]), reference, target))
            for match in LAW_RE.finditer(clause.text):
                start, end = clause.start + match.start(), clause.start + match.end()
                if not any(a <= start < b or start <= a < end for a, b, _, _ in citations):
                    identity = law_identity(match[0])
                    citations.append((start, end, identity, by_identity.get(identity)))
            for start, end, reference, target_doc in sorted(citations):
                preceding = block.text[clause.start:start]
                target_id = target_doc.document_id if target_doc else None
                article = list(re.finditer(r"(?:articolo|art\.)\s+(\d+\s*(?:bis|ter|quater|quinquies)?)\b", preceding, re.I))
                governing_article = article and re.search(r"\b(?:della|alla|del|dell')\s*$", preceding, re.I)
                if governing_article:
                    label = normalize_label(article[-1][1])
                    reference = (target_doc.document_id if target_doc else reference) + f"#art:{label}"
                    targets = article_targets.get((target_doc.document_id, label), []) if target_doc else []
                    target_id = targets[0].unit_id if len(targets) == 1 else None
                evidence = clause.text
                span = SourceSpan(block_id=block.block_id, start=clause.start, end=clause.end)
                emit(source_id, target_id, reference, "references", evidence, span)
                if block.region in {"historical", "body_quote"} or re.search(r"\bnon\s+abrogat|eccetto|eccezion|illegittim|costituzional", preceding, re.I):
                    continue
                passive_matches = list(re.finditer(r"\b(?:abrogat[oaie]|modificat[oaie]|sostituit[oaie]|inserit[oaie])\s+da(?:l|lla)", preceding, re.I))
                active_matches = list(re.finditer(r"\b(abroga|abrogano|modifica|modificano|sostituisce|sostituiscono|aggiunge|aggiungeva|inserisce|inseriscono|sostituiva)\b", preceding, re.I))
                passive = passive_matches[-1] if passive_matches else None
                active = active_matches[-1] if active_matches else None
                if passive and active and active.start() > passive.start():
                    passive = None
                if passive and target_doc:
                    kind = "repeals" if "abrogat" in passive[0].lower() else "amends"
                    emit(target_id or target_doc.document_id, source_id, source_id, kind, evidence, span)
                elif active:
                    kind = "repeals" if active[1].lower().startswith("abrog") else "amends"
                    emit(source_id, target_id, reference, kind, evidence, span)
    return relations


def graph_nodes(documents: list[Document], units: list[Unit]) -> list[dict]:
    return [{"node_id": doc.document_id, "kind": "document", "document_id": doc.document_id, "label": doc.title, "status": doc.status} for doc in documents] + [{"node_id": unit.unit_id, "kind": unit.kind, "document_id": unit.document_id, "label": unit.label, "status": unit.status} for unit in units]

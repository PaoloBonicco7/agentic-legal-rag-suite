"""Conservative, scoped cessation evidence for a source snapshot."""

import re
from collections import defaultdict
from dataclasses import dataclass

from ..artifacts import digest
from ..contracts import Block, Document, Event, SourceSpan, Unit
from .parser import HISTORICAL_RE, article_labels, parse_date

STATUS_VERSION = "conservative-status-1"
REPEAL_RE = re.compile(r"\babrogat[oaie]\b|\bcessat[oa]\s+(?:di\s+)?(?:efficacia|vigore)\b", re.I)
COMPLEX_RE = re.compile(r"\b(?:non\s+abrogat\w*|eccetto|eccezion\w*|salv\w*|parzial\w*|limitatamente|condizion\w*|purché|qualora|sostituit\w*|illegittim\w*|costituzional\w*)\b", re.I)
EFFECT_RE = re.compile(r"\b(?:a\s+decorrere\s+da(?:l|lla)?|con\s+effetto\s+da(?:l|lla)?|dal)\s+(\d{1,2}[°º]?\s+\w+\s+\d{4}|\d{4})\b", re.I)
RELATIVE_EFFECT_RE = re.compile(r"\b(?:entrata\s+in\s+vigore|dalla\s+data|a\s+decorrere|con\s+effetto|dal\s+\d)", re.I)


@dataclass(frozen=True)
class Clause:
    start: int
    end: int
    text: str


def clauses(block: Block) -> list[Clause]:
    boundaries = [0] + [match.end() for match in re.finditer(r";\s*|(?<=[.!?])\s+(?=[A-ZÀ-Ú\"(])", block.text)] + [len(block.text)]
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        part = block.text[start:end]
        left = start + len(part) - len(part.lstrip())
        right = end - len(part) + len(part.rstrip())
        if left < right:
            result.append(Clause(left, right, block.text[left:right]))
    return result


def note_bindings(blocks: list[Block], units: list[Unit]) -> tuple[dict[str, list[tuple[Block, list[Unit]]]], set[str]]:
    by_block: dict[str, list[Unit]] = defaultdict(list)
    articles: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        if unit.kind == "article":
            articles[unit.label].append(unit)
        for span in unit.spans:
            if unit not in by_block[span.block_id]:
                by_block[span.block_id].append(unit)
    bindings: dict[str, list[tuple[Block, list[Unit]]]] = defaultdict(list)
    definitions: dict[str, int] = defaultdict(int)
    current_articles: list[Unit] = []
    for block in blocks:
        if block.anchor and block.anchor.startswith("nota"):
            definitions[block.anchor] += 1
        if block.region == "heading" and block.exclusion_reason == "article_heading":
            current_articles = [u for label in article_labels(block.text) for u in articles[label]]
        candidates = [u for u in by_block[block.block_id] if u.kind in {"letter", "paragraph", "unstructured", "annex"}]
        if not candidates and block.region == "heading":
            candidates = current_articles
        for anchor in dict.fromkeys(link["href"][1:] for link in block.links if link.get("href", "").startswith("#nota")):
            bindings[anchor].append((block, candidates))
    return dict(bindings), {anchor for anchor, count in definitions.items() if count > 1}


def compatible_targets(anchor: str | None, scope: str, bindings: dict, duplicates: set[str]) -> list[Unit]:
    backlinks = bindings.get(anchor, [])
    if not anchor or anchor in duplicates or len(backlinks) != 1:
        return []
    block, candidates = backlinks[0]
    expected = {"article": "article", "paragraph": "paragraph"}.get(scope)
    targets = [unit for unit in candidates if unit.kind == expected and (scope != "paragraph" or not unit.label.startswith("intro:"))]
    if len(targets) == 1:
        return targets
    if scope == "article" and block.exclusion_reason == "article_heading" and len(article_labels(block.text)) == len(targets):
        return targets
    return []


def assess_status(document: Document, blocks: list[Block], units: list[Unit], snapshot_date: str | None = None) -> list[Event]:
    bindings, duplicates = note_bindings(blocks, units)
    for anchor in sorted(duplicates):
        document.warnings.append(f"duplicate_note_anchor:{anchor}")
    events: list[Event] = []
    current_anchor: str | None = None
    editorial = " ".join(b.text for b in blocks if b.region == "editorial")
    conflict = bool(COMPLEX_RE.search(editorial))
    unit_by_id = {u.unit_id: u for u in units}

    def uncertain(targets: list[Unit]) -> None:
        for target in targets:
            if target.status != "ceased_confirmed":
                target.status = "unknown"

    for block in blocks:
        if block.anchor and block.anchor.startswith("nota"):
            current_anchor = block.anchor
        if block.region not in {"editorial", "note"}:
            continue
        for clause in clauses(block):
            clean = re.sub(r"^\(\s*\d+[a-z]*\s*\)\s*", "", clause.text, flags=re.I).strip().strip("()").strip()
            if HISTORICAL_RE.search(clean):
                continue
            constitutional = bool(re.search(r"illegittim\w*|corte\s+costituzionale", clean, re.I))
            if not REPEAL_RE.search(clean) and not constitutional:
                continue
            law_scope = block.region == "editorial" and bool(re.match(r"^(?:legge\s+)?(?:non\s+)?abrogata\b", clean, re.I))
            if law_scope:
                scope = "law"
            elif re.match(r"^articol[oi]\s+(?:\d+\s+)?(?:è\s+|sono\s+)?abrogat[oi]\b", clean, re.I):
                scope = "article"
            elif re.match(r"^comm[ai]\s+(?:\d+\s+)?(?:è\s+|sono\s+)?abrogat[oi]\b", clean, re.I):
                scope = "paragraph"
            else:
                scope = "unknown"
            targets = compatible_targets(current_anchor, scope, bindings, duplicates) if scope != "law" else []
            effect_match = EFFECT_RE.search(clean)
            effective_at = parse_date(effect_match[1]) if effect_match else None
            unsafe_date = bool(effect_match and not effective_at) or bool(RELATIVE_EFFECT_RE.search(clean) and not effective_at)
            future = effective_at is not None and (snapshot_date is None or effective_at > snapshot_date[:10])
            formula = bool(re.match(r"^(?:(?:legge\s+)?abrogata|articol[oi]\s+(?:\d+\s+)?(?:è\s+|sono\s+)?abrogat[oi]|comm[ai]\s+(?:\d+\s+)?(?:è\s+|sono\s+)?abrogat[oi])\s+da(?:l|lla)", clean, re.I))
            adjacent_qualifier = bool(COMPLEX_RE.search(block.text)) or bool(RELATIVE_EFFECT_RE.search(block.text) and not effective_at)
            confirmed = formula and not COMPLEX_RE.search(clean) and not adjacent_qualifier and not unsafe_date and not future and (scope == "law" or bool(targets))
            if scope == "law" and conflict:
                confirmed = False
            ids = [document.document_id] if scope == "law" else [u.unit_id for u in targets] or [None]
            for target_id in ids:
                event = Event(event_id="event:" + digest(f"{document.snapshot_id}:{block.block_id}:{clause.start}:{target_id}")[:24], document_id=document.document_id, target_id=target_id, kind="constitutional_ruling" if constitutional else "repeal", scope=scope, evidence=clause.text, span=SourceSpan(block_id=block.block_id, start=clause.start, end=clause.end), effective_at=effective_at, resolution="confirmed" if confirmed else "unresolved")
                events.append(event)
            if confirmed and scope == "law":
                document.status = "ceased_confirmed"
            elif confirmed:
                for target in targets:
                    target.status = "ceased_confirmed"
            elif scope == "law" and document.status != "ceased_confirmed":
                document.status = "unknown"
            else:
                uncertain(targets or [u for _, candidates in bindings.get(current_anchor, []) for u in candidates])
    for unit in units:
        if unit.text.startswith("[") and unit.text.rstrip().endswith("]"):
            uncertain([unit])
        ancestor = unit_by_id.get(unit.parent_id)
        while ancestor:
            if ancestor.status == "ceased_confirmed":
                unit.status = "ceased_confirmed"
                break
            ancestor = unit_by_id.get(ancestor.parent_id)
        if document.status == "ceased_confirmed":
            unit.status = "ceased_confirmed"
    document.event_ids = [event.event_id for event in events]
    return events

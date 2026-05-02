"""HTML block extraction for the phase 1 laws preprocessing pipeline."""

from __future__ import annotations

from lxml import html as lxml_html

from .common import normalize_ws
from .models import Anchor, Block, Link

BLOCK_TAGS = {"p", "h1", "h2", "td", "th"}


def _element_text(element: lxml_html.HtmlElement) -> str:
    """Extract visible text while preserving line breaks from ``br`` tags."""
    def collect(node: lxml_html.HtmlElement) -> str:
        parts: list[str] = []
        if node.text:
            parts.append(node.text)
        for child in node:
            tag = (child.tag or "").lower() if isinstance(child.tag, str) else ""
            parts.append("\n" if tag == "br" else collect(child))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    return normalize_ws(collect(element))


def _anchors(element: lxml_html.HtmlElement) -> tuple[Anchor, ...]:
    """Collect named anchors that identify articles and notes."""
    anchors: list[Anchor] = []
    for anchor in element.xpath(".//a[@name]"):
        name = (anchor.get("name") or "").strip()
        if name:
            anchors.append(Anchor(name=name, text=_element_text(anchor)))
    return tuple(anchors)


def _links(element: lxml_html.HtmlElement) -> tuple[Link, ...]:
    """Collect hyperlinks so explicit legal references can be resolved later."""
    links: list[Link] = []
    for anchor in element.xpath(".//a[@href]"):
        href = (anchor.get("href") or "").strip()
        if href:
            links.append(Link(href=href, text=_element_text(anchor)))
    return tuple(links)


def parse_blocks_from_html(html: str) -> list[Block]:
    """Parse raw HTML into ordered text blocks with anchors and hyperlinks."""
    if not html:
        return []
    try:
        doc = lxml_html.fromstring(html)
    except (ValueError, lxml_html.ParserError):
        return []

    roots = doc.xpath("//article") or [doc]
    blocks: list[Block] = []
    for root in roots:
        for element in root.iter():
            tag = (element.tag or "").lower() if isinstance(element.tag, str) else ""
            if tag not in BLOCK_TAGS:
                continue
            text = _element_text(element)
            if not text:
                continue
            kind = "table_row" if tag in {"td", "th"} else tag
            blocks.append(Block(kind=kind, text=text, anchors=_anchors(element), links=_links(element)))
    return blocks

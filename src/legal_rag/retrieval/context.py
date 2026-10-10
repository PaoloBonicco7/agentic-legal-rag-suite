"""One complete-source context builder shared by every profile."""

from legal_rag.contracts import Candidate, Context, Evidence


def build_context(candidates: list[Candidate], token_counter, *, max_tokens=4096, max_units=12) -> Context:
    result = Context()
    seen = set()
    covered = {}
    for candidate in candidates:
        chunk = candidate.chunk
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        if chunk.spans and all(any(start <= span.start and end >= span.end for start, end in covered.get((chunk.document_id, span.block_id), [])) for span in chunk.spans):
            result.omissions.append({"chunk_id": chunk.chunk_id, "reason": "source_already_shown"})
            continue
        context_id = f"E{len(result.evidence) + 1}"
        recovered_boundary = any(warning == "source_quote_boundary_recovered" or warning.startswith("source_quote_boundary_recovered:") for warning in chunk.warnings)
        uncertainty = "Parsing uncertainty: source_quote_boundary_recovered\n" if recovered_boundary else ""
        rendered = f"[{context_id}] {chunk.identity or chunk.document_id}\n{chunk.title}\nArticle: {chunk.article_label or 'unstructured'} — {chunk.heading}\nSource: {chunk.source_url}\nStatus: {chunk.status}\n{uncertainty}{chunk.text}"
        combined = result.text + ("\n\n" if result.text else "") + rendered
        reason = "unit_limit" if len(result.evidence) >= max_units else ("token_limit" if token_counter.count(combined) > max_tokens else None)
        if reason:
            result.omissions.append({"chunk_id": chunk.chunk_id, "reason": reason})
            continue
        result.evidence.append(Evidence(context_id=context_id, chunk=chunk, rendered=rendered))
        result.text = combined
        for span in chunk.spans:
            key = (chunk.document_id, span.block_id)
            intervals = sorted([*covered.get(key, []), (span.start, span.end)])
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            covered[key] = merged
    result.token_count = token_counter.count(result.text)
    return result

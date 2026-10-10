"""Offline HTML reports from saved, inspectable artifacts."""

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field

from legal_rag.artifacts import code_identity, digest, file_hash, read_json, utc_now, write_json
from legal_rag.contracts import Record
from legal_rag.evaluation import compare_runs, load_diagnostics, load_results, summarize, summarize_diagnostics

from .comments import load_comments

REPORT_SCHEMA_VERSION = "reporting-v2"
CSS = """
:root{color-scheme:light;--ink:#192b35;--accent:#086b68;--muted:#5b6d75;--line:#d7e1e2}
*{box-sizing:border-box}body{margin:0;background:#f3f6f5;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
header{background:#133b44;color:white;padding:34px max(24px,calc((100vw - 1180px)/2));border-bottom:5px solid #d4b475}
header a{color:#d8f3ec}header h1{margin:8px 0;font-size:34px;line-height:1.15}header p{max-width:850px;color:#dae6e7}
main{max-width:1220px;margin:28px auto;padding:0 22px}section,.card{background:white;border:1px solid var(--line);border-radius:10px;padding:22px;margin:18px 0}
h2{margin:0 0 16px;font-size:22px}h3{font-size:18px}a{color:var(--accent)}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#f0f5f4;color:var(--muted)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f4;padding:16px;border-radius:6px;font:13px/1.6 ui-monospace,monospace}.muted{color:var(--muted)}.bad{color:#9c332b}.good{color:#086b68}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px}.card strong{display:block;font-size:30px}.badge{display:inline-block;padding:3px 9px;border-radius:14px;background:#e8f1ee;font-size:12px}.warning{border-left:4px solid #bc872f;background:#fff7e5;padding:16px}
details{margin:12px 0}summary{cursor:pointer;font-weight:600}input,select{padding:9px;border:1px solid var(--line);border-radius:6px;margin:0 8px 14px 0}.scroll{overflow-x:auto}blockquote{border-left:3px solid var(--accent);margin:12px 0;padding:8px 16px;background:#f6f9f8}.meta{font-size:13px;color:var(--muted)}footer{padding:30px 0;color:var(--muted);font-size:13px}
@media(max-width:600px){header{padding:25px}header h1{font-size:28px}main{padding:0 12px}section{padding:15px}}
"""
FILTER_SCRIPT = """document.querySelectorAll('[data-filter]').forEach(input=>input.addEventListener('input',()=>{
const table=document.getElementById(input.dataset.filter);const term=input.value.toLowerCase();
table.querySelectorAll('tbody tr').forEach(row=>{row.hidden=!row.textContent.toLowerCase().includes(term)});
}));"""


class ReportManifest(Record):
    schema_version: str = REPORT_SCHEMA_VERSION
    created_at: str
    corpus_dir: str | None
    run_dirs: list[str]
    diagnostic_dirs: list[str] = Field(default_factory=list)
    comments_path: str | None = None
    input_hashes: dict[str, str]
    code: dict = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def escape(value) -> str:
    return html.escape(str(value), quote=True)


def _redact(value):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if str(key).lower() in {"api_key", "authorization", "password", "secret", "access_token"} else _redact(nested)
                for key, nested in value.items()}
    if isinstance(value, list):
        return [_redact(nested) for nested in value]
    return value


def _pre(value) -> str:
    text = json.dumps(_redact(value), ensure_ascii=False, indent=2, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return f"<pre>{escape(text)}</pre>"


def _link(url: str, label: str) -> str:
    try:
        parsed = urlsplit(url)
        safe = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
    except ValueError:
        safe = False
    return f'<a href="{escape(url)}" rel="noopener noreferrer">{escape(label)}</a>' if safe else escape(label)


def _snapshot_info(manifest, corpus_manifest=None, corpus_sha=None, *, snapshot_id=None):
    inputs = manifest.get("inputs")
    if inputs is None:
        return {"id": manifest.get("snapshot_id"), "date": manifest.get("snapshot_date")}
    index = inputs.get("index", {})
    identities = [value for value in (inputs.get("identity", {}).get("corpus"), inputs.get("corpus"), index.get("corpus"))
                  if value is not None]
    identity = next((value for value in identities if isinstance(value, dict)), {})
    reconciled = bool(corpus_manifest and corpus_manifest.get("snapshot_id") and corpus_sha and identities and
        index.get("snapshot_id", corpus_manifest["snapshot_id"]) == corpus_manifest["snapshot_id"] and all(isinstance(value, dict) and
        value.get("snapshot_id") == corpus_manifest["snapshot_id"] and value.get("manifest_sha256") == corpus_sha for value in identities))
    return {"id": identity.get("snapshot_id") or index.get("snapshot_id") or snapshot_id,
            "date": corpus_manifest.get("snapshot_date") if reconciled else None}


def _snapshot_html(snapshot) -> str:
    if snapshot is None:
        return ""
    return f'<p class="snapshot"><b>Snapshot:</b> {escape(snapshot.get("id") or "ID non disponibile")} · <b>Data snapshot:</b> {escape(snapshot.get("date") or "data non disponibile")}</p>'


def _page(title: str, subtitle: str, content: str, *, prefix: str = "", snapshot=None) -> str:
    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · Legal RAG</title><link rel="stylesheet" href="{prefix}style.css"></head><body>
<header><a href="{prefix}index.html">LEGAL RAG · TESI</a><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>{_snapshot_html(snapshot)}</header>
<main>{content}<footer>Report statico offline · artefatti verificabili · nessuna chiamata a modelli</footer></main>
<script>{FILTER_SCRIPT}</script></body></html>"""


def _table(headers: list[str], rows: list[list[str]], *, table_id: str | None = None) -> str:
    identifier = f' id="{escape(table_id)}"' if table_id else ""
    search = f'<input aria-label="Filtra tabella" placeholder="Filtra per QID, livello, stato o testo" data-filter="{escape(table_id)}">' if table_id else ""
    return search + f'<div class="scroll"><table{identifier}><thead><tr>' + "".join(f"<th>{escape(header)}</th>" for header in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _percent(value) -> str:
    return "non misurato" if value is None else f"{100 * value:.1f}%"


def _filename_id(value: str) -> str:
    return value if len(value) <= 64 and re.fullmatch(r"[A-Za-z0-9_-]+", value) else digest(value)[:16]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _comments_section(comments) -> str:
    if not comments:
        return ""
    content = '<section><h2>Commenti interpretativi</h2><p class="muted">Note conservate separatamente: non modificano score, gold o artefatti sperimentali.</p>'
    for comment in comments:
        content += f'<article><h3>{escape(comment.author)} · {escape(comment.comment_id)}</h3>'
        content += _pre({"qid": comment.qid, "run_hash": comment.run_hash, "document_id": comment.document_id})
        content += f'<blockquote>{escape(comment.text)}</blockquote></article>'
    return content + '</section>'


def _qid_comments(comments, qid, run_hash):
    return [comment for comment in comments if comment.qid == qid and (comment.run_hash is None or comment.run_hash == run_hash)]


def _qid_page(row, run_name: str, comments=(), *, snapshot=None) -> str:
    answer = row.answer
    judging = row.judge.to_json_record() if row.judge else {"score": row.score, "judge_error": row.judge_error}
    content = f'<section><span class="badge">{escape(row.level)} · {escape(row.format)} · score {escape(row.score)}</span><h2>{escape(row.question)}</h2><h3>Gold ({escape(run_name)})</h3>{_pre(row.correct_answer)}<h3>Risposta</h3>{_pre(answer.answer)}<p>Stato: {escape(answer.status)} · opzione: {escape(answer.selected_option)}</p></section>'
    content += '<section><h2>Giudizio e annotazioni</h2>' + _pre(judging) + _pre({"flags": row.flags, "expected_references": [ref.to_json_record() for ref in row.expected_references], "errors": answer.errors,
        "control_judge": row.control_judge.to_json_record() if row.control_judge else None, "control_error": row.control_error}) + '</section>'
    content += '<section><h2>Contesto realmente inviato</h2>' + f'<p>{answer.context.token_count} token sperimentali · {len(answer.context.evidence)} evidenze</p>' + _pre(answer.context.text)
    for evidence in answer.context.evidence:
        chunk = evidence.chunk
        content += f'<details><summary>{escape(evidence.context_id)} · {escape(chunk.document_id)} · articolo {escape(chunk.article_label)}</summary><p>{_link(chunk.source_url, "Fonte ufficiale")} · stato {escape(chunk.status)}</p>{_pre(chunk.text)}{_pre({"chunk_id": chunk.chunk_id, "spans": [span.to_json_record() for span in chunk.spans], "warnings": chunk.warnings})}</details>'
    content += '</section><section><h2>Citazioni e supporto</h2>' + _pre(row.citation_checks) + '<p class="muted">Una citazione formalmente valida non dimostra che supporti tutti i claim della risposta.</p></section>'
    candidates = [[str(candidate.rank), escape(candidate.origin), escape(candidate.score), escape(candidate.chunk.document_id), escape(candidate.chunk.article_label),
                   f'<details><summary>{escape(candidate.chunk.chunk_id)}</summary>{_pre(candidate.chunk.text)}</details>'] for candidate in answer.candidates]
    content += '<section><h2>Retrieval e selezione</h2>' + _pre({"queries": answer.queries, "omissions": answer.context.omissions, "reference_proxies": row.retrieval_proxies}) + _table(["Rank", "Origine", "Score", "Legge", "Art.", "Testo"], candidates, table_id="candidates")
    content += '<details><summary>Trace delle fasi e dei budget</summary>' + _pre(getattr(answer, "retrieval_trace", {})) + '</details></section>'
    content += '<section><h2>Uso dei provider</h2>' + _pre([usage.to_json_record() for usage in (*answer.usage, *row.judge_usage, *row.control_usage)]) + '</section>'
    content += _comments_section(comments)
    return _page(row.qid, f"{run_name} · gold e contesto restano separati dal runtime di generazione", content, prefix="../", snapshot=snapshot)


def _experiment_page(manifest: dict, rows, metrics: dict, detail_links: dict, prefix: str = "../", *, snapshot=None) -> str:
    profile = manifest["inputs"]["profile"]
    table_rows = [[f'<a href="{detail_links[(row.qid, row.format)]}">{escape(row.qid)}</a>', escape(row.format), escape(row.level), escape(row.score),
                   escape(row.answer.status), escape(", ".join(row.flags)), escape(row.question)] for row in rows]
    observed = {(row.qid, row.format) for row in rows}
    questions = {question["qid"]: question for question in manifest.get("selected_questions", [])}
    for qid in manifest["inputs"]["qids"]:
        for format_name in manifest["inputs"]["formats"]:
            if (qid, format_name) not in observed:
                question = questions.get(qid, {})
                table_rows.append([f'<a href="{detail_links[(qid, format_name)]}">{escape(qid)}</a>', escape(format_name), escape(question.get("level", "")),
                    "mancante", "not_evaluated", "nessun risultato finale salvato", escape(question.get("question", ""))])
    content = '<section><h2>Stato e denominatori</h2>' + _pre({"complete": manifest["complete"], "status": manifest["status"], "selected_intents": manifest["selected_intents"], "view_intents": manifest["view_intents"], "original_intents": manifest["original_intents"], "pilot": manifest["pilot"], "view_provisional": manifest.get("view_provisional", False), "reviewed_intents": manifest.get("reviewed_intents", 0), "view": manifest["inputs"]["view"]}) + '</section>'
    content += '<section><h2>Metriche</h2>'
    for format_name, values in metrics.items():
        content += f'<h3>{escape(format_name)}</h3><div class="cards"><div class="card"><span>Fully correct / intenti</span><strong>{_percent(values["fully_correct_rate"])}</strong><span>{values["score_distribution"]["2"]} / {values["denominator"]}</span></div><div class="card"><span>Media 0–2</span><strong>{values["mean_score"]:.3f}</strong></div><div class="card"><span>Copertura giudizi</span><strong>{_percent(values["score_coverage"])}</strong></div></div>'
    content += '<details><summary>Metriche complete e denominatori</summary>' + _pre(metrics) + '</details></section><section><h2>Domande</h2>' + _table(["QID", "Formato", "Livello", "Score", "Stato", "Flag", "Domanda"], table_rows, table_id="questions") + '</section>'
    content += '<section><h2>Identità e revisioni</h2><details><summary>Configurazione e provenienza</summary>' + _pre(manifest) + '</details></section>'
    return _page(profile, "Esperimento ispezionabile; score mancanti non vengono nascosti", content, prefix=prefix, snapshot=snapshot)


def _pending_qid_page(qid, format_name, manifest, answer_checkpoint=None, comments=(), *, snapshot=None):
    question = next((question for question in manifest.get("selected_questions", []) if question["qid"] == qid), {"qid": qid})
    content = '<p class="warning">Caso selezionato senza risultato finale di valutazione. Lo score resta mancante.</p>'
    content += '<section><h2>Domanda e riferimenti del benchmark</h2>' + _pre(question) + '</section><section><h2>Stato del run</h2>' + _pre({"format": format_name,
        "stopped_reason": manifest.get("stopped_reason"), "status": manifest["status"]}) + '</section>'
    content += '<section><h2>Checkpoint di generazione</h2>' + (_pre(answer_checkpoint) if answer_checkpoint else '<p>Nessun checkpoint salvato; il caso non risulta elaborato.</p>') + '</section>'
    content += _comments_section(comments)
    return _page(qid, f"{manifest['inputs']['profile']} · valutazione mancante", content, prefix="../", snapshot=snapshot)


def _corpus_page(corpus_dir: Path, output_dir: Path, comments=()) -> tuple[str, dict[str, str]]:
    manifest = read_json(corpus_dir / "manifest.json")
    snapshot = _snapshot_info(manifest)
    documents = _jsonl(corpus_dir / "documents.jsonl")
    units = defaultdict(list)
    for unit in _jsonl(corpus_dir / "units.jsonl"):
        units[unit["document_id"]].append(unit)
    events, relations = defaultdict(list), defaultdict(list)
    for event in _jsonl(corpus_dir / "events.jsonl"):
        events[event["document_id"]].append(event)
    for relation in _jsonl(corpus_dir / "relations.jsonl"):
        relations[relation["source_id"].split("#", 1)[0]].append(relation)
    hashes = {str(corpus_dir / name): file_hash(corpus_dir / name) for name in ("manifest.json", "documents.jsonl", "units.jsonl", "events.jsonl", "relations.jsonl", "quality.json") if (corpus_dir / name).exists()}
    quality = read_json(corpus_dir / "quality.json") if (corpus_dir / "quality.json").exists() else {}
    table_rows = []
    detail_dir = output_dir / "corpus"
    detail_dir.mkdir()
    for document in documents:
        document_id = document["document_id"]
        filename = digest(document_id)[:16] + ".html"
        table_rows.append([f'<a href="corpus/{filename}">{escape(document_id)}</a>', escape(document.get("title", "")),
                           escape(document.get("status", "")), escape(document.get("availability", "")), str(len(units[document_id]))])
        content = '<section><h2>Provenienza e stato</h2>' + _pre(document) + f'<p>{_link(document.get("source_url", ""), "Fonte ufficiale")}</p></section><section><h2>Testo e struttura</h2>'
        for unit in units[document_id]:
            content += f'<details><summary>{escape(unit.get("kind", ""))} {escape(unit.get("label", ""))} · {escape(unit.get("status", ""))}</summary>{_pre(unit.get("text", ""))}{_pre({"unit_id": unit["unit_id"], "parent_id": unit.get("parent_id"), "spans": unit.get("spans", [])})}</details>'
        content += '</section><section><h2>Eventi e relazioni verificabili</h2>' + _pre({"events": events[document_id], "relations": relations[document_id]}) + '</section>'
        content += _comments_section([comment for comment in comments if comment.document_id == document_id and comment.run_hash is None])
        (detail_dir / filename).write_text(_page(document.get("title", document_id), document_id, content, prefix="../", snapshot=snapshot), encoding="utf-8")
    statuses = Counter(document.get("status", "unknown") for document in documents)
    cards = '<div class="cards">' + ''.join(f'<div class="card"><span>{escape(status)}</span><strong>{count}</strong></div>' for status, count in statuses.items()) + '</div>'
    content = '<p class="warning">no_cessation_evidence significa assenza di cessazione esplicita rilevata nello snapshot; non certifica la vigenza giuridica. unknown resta distinto.</p>' + cards
    content += '<section><h2>Documenti</h2>' + _table(["ID", "Titolo", "Stato", "Disponibilità", "Unità"], table_rows, table_id="documents") + '</section>'
    content += '<section><h2>Qualità e snapshot</h2>' + _pre(quality) + '<details><summary>Manifest</summary>' + _pre(manifest) + '</details></section>'
    return _page("Corpus normativo", "Snapshot, struttura ed evidenze delle annotazioni", content, snapshot=snapshot), hashes


def _diagnostic_case_page(row, name: str, comments=(), *, snapshot=None) -> str:
    answer = row.answer
    content = f'<p class="warning">Solo retrieval: nessuna risposta generata, nessun giudice e nessuno score di correttezza.</p><section><h2>{escape(answer.question)}</h2>'
    content += _pre({"qid": row.qid, "level": row.level, "profile": answer.profile, "flags": row.flags,
        "expected_references": [ref.to_json_record() for ref in row.expected_references], "retrieval_proxies": row.retrieval_proxies,
        "timings_seconds": row.timings, "errors": answer.errors}) + '</section>'
    content += '<section><h2>Contesto selezionato</h2>' + _pre(answer.context.to_json_record()) + '</section>'
    content += '<section><h2>Pool originale e selezione finale</h2>' + _pre(answer.retrieval_trace)
    for candidate in answer.candidates:
        content += f'<details><summary>{candidate.rank} · {escape(candidate.chunk.document_id)} · art. {escape(candidate.chunk.article_label)}</summary>{_pre(candidate.chunk.to_json_record())}</details>'
    content += '</section>'
    content += _comments_section(comments)
    return _page(row.qid, f"{name} · {answer.profile} · diagnosi delle fonti recuperate", content, prefix="../", snapshot=snapshot)


def _diagnostic_profile_page(manifest, profile: str, rows, metrics, links, *, snapshot=None) -> str:
    content = '<p class="warning">Article Success e MRR misurano corrispondenze con riferimenti annotati incompleti. Non misurano correttezza della risposta né sufficienza delle evidenze.</p>'
    content += '<section><h2>Proxy e copertura</h2>' + _pre(metrics) + '</section>'
    table = [[f'<a href="{links[row.qid]}">{escape(row.qid)}</a>', escape(row.level), escape(row.answer.status),
        escape(row.retrieval_proxies.get("context_article_success")), escape(", ".join(row.flags)), escape(row.answer.question)] for row in rows]
    content += '<section><h2>Casi</h2>' + _table(["QID", "Livello", "Stato retrieval", "Articolo in contesto", "Flag", "Domanda"], table, table_id="cases") + '</section>'
    content += '<section><h2>Identità e policy dei tempi</h2>' + _pre(manifest) + '</section>'
    return _page(f"Diagnosi · {profile}", "Retrieval senza generazione o giudice", content, prefix="../", snapshot=snapshot)


def _case_comparison_page(qid, entries, *, format_name=None, comparisons=(), diagnostic=False, comments=()):
    if diagnostic:
        notice = "Confronto descrittivo del retrieval: proxy di riferimenti incompleti, nessuna risposta o score generativo. Tempi e cache dipendono dal run."
        checks = [{"run": entry["name"], "run_hash": entry["manifest"]["run_hash"],
                   "inputs": entry["manifest"]["inputs"]} for entry in entries]
    else:
        participants = {entry["path"] for entry in entries if entry["selected"]}
        pairs = [pair for pair in comparisons if pair["left"] in participants and pair["right"] in participants]
        compatible = len(participants) > 1 and len(pairs) == len(participants) * (len(participants) - 1) // 2 and all(pair["comparable"] for pair in pairs)
        notice = ("Vista affiancata di run compatibili; i gate di promozione riguardano l'esperimento completo." if compatible else
                  "Confronto descrittivo: condizioni incompatibili, incomplete o insufficienti per un confronto controllato.")
        checks = [{"left": pair["left"], "right": pair["right"], "comparable": pair["comparable"], "issues": pair["issues"],
                   "format_gates": pair.get("formats", {}).get(format_name, {}).get("promotion_gates")} for pair in pairs]
    content = f'<p class="warning">{escape(notice)}</p><div class="cards">'
    for entry in entries:
        row, question, checkpoint = entry.get("row"), entry.get("question", {}), entry.get("checkpoint")
        answer = row.answer.to_json_record() if row else checkpoint.get("answer") if checkpoint else None
        status = answer["status"] if answer else "not_evaluated" if entry["selected"] else "not_selected"
        if checkpoint and not checkpoint.get("attempted", True):
            status = "not_attempted"
        content += f'<section class="card"><h2>{escape(entry["profile"])}</h2><p>{escape(entry["name"])} · {escape(status)}</p>'
        content += _snapshot_html(entry.get("snapshot"))
        content += _pre({"view": entry["manifest"]["inputs"]["view"], "run_hash": entry["manifest"]["run_hash"]})
        fallback = "Domanda non disponibile nel checkpoint" if entry["selected"] else "Domanda non selezionata in questo run"
        text = row.answer.question if diagnostic and row else row.question if row else question.get("question", fallback)
        content += f'<h3>Domanda</h3>{_pre(text)}'
        if diagnostic:
            content += '<h3>Proxy retrieval</h3>' + _pre({"reference_proxies": row.retrieval_proxies, "timings_seconds": row.timings,
                "flags": row.flags, "context_tokens": row.answer.context.token_count} if row else {"record": "mancante", "proxies": None})
        else:
            content += '<h3>Gold e revisioni</h3>' + _pre({"gold": row.correct_answer if row else question.get("correct_answer"),
                "reviews": [review for review in entry["manifest"].get("reviews", []) if review["qid"] == qid]})
            content += '<h3>Risposta</h3>' + (_pre(answer.get("answer", "")) if answer else '<p>Nessun checkpoint di generazione salvato.</p>')
            content += '<h3>Score e giudici</h3>' + _pre({"score": row.score if row else None,
                "judge": row.judge.to_json_record() if row and row.judge else None,
                "control_judge": row.control_judge.to_json_record() if row and row.control_judge else None,
                "judge_error": row.judge_error if row else None, "flags": row.flags if row else [],
                "generation_errors": answer.get("errors", []) if answer else []})
        if entry.get("detail"):
            content += f'<p><a href="{escape(entry["detail"])}">Apri contesto e dettagli salvati</a></p>'
        content += '</section>'
    content += '</div><section><h2>Compatibilità e provenienza</h2>' + _pre(checks) + '</section>'
    content += _comments_section(comments)
    return _page(qid, "Profili retrieval affiancati" if diagnostic else f"Risposte affiancate · {format_name}", content, prefix="../")


def build_report(corpus_dir: Path | None, run_dirs: list[Path], output_dir: Path, *, diagnostic_dirs=(), comments_path: Path | None = None) -> Path:
    output_dir = Path(output_dir)
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else None
    run_dirs = [Path(directory) for directory in run_dirs]
    diagnostic_dirs = [Path(directory) for directory in diagnostic_dirs]
    comments_path = Path(comments_path) if comments_path is not None else None
    comments = load_comments(comments_path) if comments_path is not None else []
    if not corpus_dir and not run_dirs and not diagnostic_dirs:
        raise ValueError("A report needs a corpus, evaluation run or retrieval diagnostics")
    if output_dir.exists():
        raise FileExistsError(f"Report directory already exists: {output_dir}")
    if any(output_dir.resolve().is_relative_to(path.resolve()) for path in ([corpus_dir] if corpus_dir else []) + run_dirs + diagnostic_dirs):
        raise ValueError("Report outputs must be separate from input artifact directories")
    input_hashes, warnings = {}, []
    if comments_path is not None:
        input_hashes[str(comments_path)] = file_hash(comments_path)
    loaded = []
    for run_dir in run_dirs:
        manifest = read_json(run_dir / "manifest.json")
        rows = load_results(run_dir)
        metrics = {format_name: summarize([row for row in rows if row.format == format_name], manifest["selected_intents"]) for format_name in manifest["inputs"]["formats"]}
        for filename in ("manifest.json", "answers.jsonl", "results.jsonl", "metrics.json"):
            if (run_dir / filename).exists():
                input_hashes[str(run_dir / filename)] = file_hash(run_dir / filename)
        loaded.append((run_dir, manifest, rows, metrics))
    comparisons = compare_runs(run_dirs) if run_dirs else {"schema_version": "comparison-v2", "comparisons": []}
    for saved_run in comparisons.get("runs", []):
        if saved_run["integrity_issues"]:
            warnings.append(f'Artifact integrity: {Path(saved_run["path"]).name}: {", ".join(saved_run["integrity_issues"])}')
    output_dir.mkdir(parents=True)
    (output_dir / "style.css").write_text(CSS, encoding="utf-8")
    (output_dir / "experiments").mkdir()
    (output_dir / "questions").mkdir()
    (output_dir / "diagnostics").mkdir()
    corpus_manifest, corpus_sha = None, None
    if corpus_dir:
        corpus_manifest = read_json(corpus_dir / "manifest.json")
        corpus_sha = file_hash(corpus_dir / "manifest.json")
        for filename, expected in corpus_manifest.get("files", {}).items():
            path = corpus_dir / filename
            if not path.is_file() or file_hash(path) != expected:
                warnings.append(f"Corpus artifact hash mismatch: {filename}")
        corpus_html, hashes = _corpus_page(corpus_dir, output_dir, comments)
        (output_dir / "corpus.html").write_text(corpus_html, encoding="utf-8")
        input_hashes.update(hashes)
    overview_rows, comparison_runs = [], []
    for position, (run_dir, manifest, rows, metrics) in enumerate(loaded, 1):
        stem = f"run-{position}"
        snapshot = _snapshot_info(manifest, corpus_manifest, corpus_sha,
                                  snapshot_id=next((row.answer.snapshot_id for row in rows if row.answer.snapshot_id), None))
        detail_links = {}
        checkpoints = {tuple(raw["key"]): raw for raw in _jsonl(run_dir / "answers.jsonl")}
        for row in rows:
            filename = f"{stem}-{_filename_id(row.qid)}-{row.format}.html"
            (output_dir / "questions" / filename).write_text(_qid_page(row, run_dir.name, _qid_comments(comments, row.qid, manifest["run_hash"]), snapshot=snapshot), encoding="utf-8")
            detail_links[(row.qid, row.format)] = f"../questions/{filename}"
        for qid in manifest["inputs"]["qids"]:
            for format_name in manifest["inputs"]["formats"]:
                if (qid, format_name) not in detail_links:
                    filename = f"{stem}-{_filename_id(qid)}-{_filename_id(format_name)}.html"
                    (output_dir / "questions" / filename).write_text(_pending_qid_page(qid, format_name, manifest, checkpoints.get((qid, format_name)),
                        _qid_comments(comments, qid, manifest["run_hash"]), snapshot=snapshot), encoding="utf-8")
                    detail_links[(qid, format_name)] = f"../questions/{filename}"
        (output_dir / "experiments" / f"{stem}.html").write_text(_experiment_page(manifest, rows, metrics, detail_links, snapshot=snapshot), encoding="utf-8")
        comparison_runs.append({"path": str(run_dir), "name": run_dir.name, "manifest": manifest, "profile": manifest["inputs"]["profile"],
            "rows": {(row.qid, row.format): row for row in rows}, "questions": {question["qid"]: question for question in manifest.get("selected_questions", [])},
            "checkpoints": checkpoints, "links": detail_links, "snapshot": snapshot})
        for format_name, values in metrics.items():
            overview_rows.append([f'<a href="experiments/{stem}.html">{escape(manifest["inputs"]["profile"])}</a>', _snapshot_html(snapshot), escape(manifest["inputs"]["view"]),
                escape(format_name), str(values["denominator"]), _percent(values["fully_correct_rate"]), f'{values["mean_score"]:.3f}',
                _percent(values["score_coverage"]), _percent(values["citation_validity"]), f'{values["usage"]["known_cost_eur"]:.4f} €', escape(manifest["status"])])
        if not manifest["complete"]:
            warnings.append(f"Incomplete evaluation: {run_dir.name}")
    content = '<p class="warning">Le 100 domande legacy erano già state esplorate. La vista legacy descrive il benchmark originale e non dimostra correttezza del diritto corrente. La vista current_approved contiene soltanto revisioni approvate per lo snapshot indicato. Nessuna vista dimostra prestazioni su domande nuove. MCQ e no-hint condividono gli stessi intenti.</p>'
    if corpus_dir:
        content += '<section><h2>Corpus fornito</h2>' + _snapshot_html(_snapshot_info(corpus_manifest)) + '<p><a href="corpus.html">Esplora snapshot, qualità del preprocessing e documenti</a></p></section>'
    if overview_rows:
        content += '<section><h2>Risultati</h2><p class="muted">Fully correct = score 2 sul denominatore fisso; se mancano score, Fully correct e media 0–2 sono limiti inferiori, non confronti completi. I costi LLM contabilizzati possono essere stime o riserve prudenziali; embedding e chiamate interrotte richiedono il ledger del provider. Basi e importi sconosciuti sono nei dettagli.</p>' + _table(["Metodo", "Snapshot e data", "Vista", "Formato", "Intenti", "Fully correct", "Media 0–2", "Copertura", "Citazioni valide", "Costo LLM contabilizzato", "Stato"], overview_rows) + '</section>'
    case_rows = []
    keys = sorted({key for entry in comparison_runs for key in entry["links"]})
    for qid, format_name in keys:
        entries = [{**entry, "row": entry["rows"].get((qid, format_name)), "question": entry["questions"].get(qid, {}),
            "checkpoint": entry["checkpoints"].get((qid, format_name)), "detail": entry["links"].get((qid, format_name)),
            "selected": (qid, format_name) in entry["links"]} for entry in comparison_runs]
        filename = f"compare-{_filename_id(qid)}-{_filename_id(format_name)}.html"
        scoped = [comment for comment in comments if comment.qid == qid and (comment.run_hash is None or comment.run_hash in {entry["manifest"]["run_hash"] for entry in entries if entry["selected"]})]
        (output_dir / "questions" / filename).write_text(_case_comparison_page(qid, entries, format_name=format_name,
            comparisons=comparisons["comparisons"], comments=scoped), encoding="utf-8")
        question = next((entry["question"] for entry in entries if entry["question"]), {})
        case_rows.append([f'<a href="questions/{filename}">{escape(qid)}</a>', escape(format_name), escape(question.get("level", "")),
            str(sum(entry["selected"] for entry in entries)), escape(question.get("question", ""))])
    if case_rows:
        content += '<section><h2>Risposte affiancate per domanda</h2>' + _table(["QID", "Formato", "Livello", "Run selezionati", "Domanda"], case_rows, table_id="answer-comparisons") + '</section>'
    diagnostic_overview, diagnostic_cases = [], defaultdict(list)
    for position, directory in enumerate(diagnostic_dirs, 1):
        manifest = read_json(directory / "manifest.json")
        rows = load_diagnostics(directory)
        snapshot = _snapshot_info(manifest, corpus_manifest, corpus_sha)
        expected_keys = {(qid, profile) for qid in manifest["inputs"]["qids"] for profile in manifest["inputs"]["profiles"]}
        if {(row.qid, row.answer.profile) for row in rows} != expected_keys:
            warnings.append(f"Diagnostic selection mismatch: {directory.name}")
        for filename in ("manifest.json", "records.jsonl", "metrics.json"):
            input_hashes[str(directory / filename)] = file_hash(directory / filename)
            expected = manifest.get("output_hashes", {}).get(filename)
            if expected and input_hashes[str(directory / filename)] != expected:
                warnings.append(f"Diagnostic artifact hash mismatch: {directory.name}/{filename}")
        for profile in manifest["inputs"]["profiles"]:
            profile_rows = [row for row in rows if row.answer.profile == profile]
            values = summarize_diagnostics(profile_rows, manifest["selected_intents"])
            stem, links = f"diagnostic-{position}-{_filename_id(profile)}", {}
            for row in profile_rows:
                filename = f"{stem}-{_filename_id(row.qid)}.html"
                (output_dir / "questions" / filename).write_text(_diagnostic_case_page(row, directory.name, _qid_comments(comments, row.qid, manifest["run_hash"]), snapshot=snapshot), encoding="utf-8")
                links[row.qid] = f"../questions/{filename}"
            (output_dir / "diagnostics" / f"{stem}.html").write_text(_diagnostic_profile_page(manifest, profile, profile_rows, values, links, snapshot=snapshot), encoding="utf-8")
            by_qid = {row.qid: row for row in profile_rows}
            for qid in manifest["inputs"]["qids"]:
                diagnostic_cases[qid].append({"name": directory.name, "profile": profile, "manifest": manifest,
                    "row": by_qid.get(qid), "detail": links.get(qid), "selected": True, "snapshot": snapshot})
            diagnostic_overview.append([f'<a href="diagnostics/{stem}.html">{escape(profile)}</a>', escape(directory.name), _snapshot_html(snapshot), str(values["denominator"]),
                _percent(values["candidate_pool_article_success"]["100"]["fixed_denominator_value"]),
                _percent(values["article_success"]["10"]["fixed_denominator_value"]),
                _percent(values["context_article_success"]["fixed_denominator_value"]), str(values["retrieval_errors"]),
                f'{values["timings_seconds"].get("total", {}).get("total", 0):.2f} s'])
    if diagnostic_overview:
        content += '<section><h2>Diagnosi del retrieval</h2><p class="muted">Solo proxy di riferimenti: nessuna risposta generata e nessuno score di correttezza. Valori sul denominatore fisso; copertura e tempi con cache nei dettagli.</p>' + _table(["Profilo", "Run", "Snapshot e data", "Intenti", "Pool Article@100", "Selezione Article@10", "Articolo in contesto", "Errori retrieval", "Tempo totale"], diagnostic_overview) + '</section>'
    diagnostic_case_rows = []
    for qid, entries in sorted(diagnostic_cases.items()):
        filename = f"compare-retrieval-{_filename_id(qid)}.html"
        scoped = [comment for comment in comments if comment.qid == qid and (comment.run_hash is None or comment.run_hash in {entry["manifest"]["run_hash"] for entry in entries})]
        (output_dir / "questions" / filename).write_text(_case_comparison_page(qid, entries, diagnostic=True, comments=scoped), encoding="utf-8")
        row = next((entry["row"] for entry in entries if entry["row"]), None)
        diagnostic_case_rows.append([f'<a href="questions/{filename}">{escape(qid)}</a>', escape(row.level if row else ""),
            escape(", ".join(entry["profile"] for entry in entries)), escape(row.answer.question if row else "record mancante")])
    if diagnostic_case_rows:
        content += '<section><h2>Retrieval affiancato per domanda</h2>' + _table(["QID", "Livello", "Profili", "Domanda"], diagnostic_case_rows, table_id="retrieval-comparisons") + '</section>'
    content += '<section><h2>Confronti appaiati e gate</h2>'
    for comparison in comparisons["comparisons"]:
        content += f'<h3>{escape(Path(comparison["right"]).name)} − {escape(Path(comparison["left"]).name)}</h3>'
        if not comparison["comparable"]:
            content += '<p class="bad">Confronto non compatibile: ' + escape(", ".join(comparison["issues"])) + '</p>'
        else:
            for format_name, values in comparison["formats"].items():
                interval = values["fully_correct"]
                content += f'<p>{escape(format_name)}: delta {100 * interval["delta"]:+.1f} pp; IC95% [{100 * interval["low"]:+.1f}, {100 * interval["high"]:+.1f}] pp · promozione operativa: {escape(values["promoted"])}</p>' + _pre(values)
    content += '</section>'
    content += _comments_section(comments)
    if warnings:
        content += '<section><h2>Limiti operativi</h2>' + _pre(warnings) + '</section>'
    (output_dir / "index.html").write_text(_page("Evidenze e risultati", "Legal RAG per le leggi della Valle d’Aosta · dal corpus alla risposta", content), encoding="utf-8")
    write_json(output_dir / "comparisons.json", comparisons)
    manifest = ReportManifest(created_at=utc_now(), corpus_dir=str(corpus_dir) if corpus_dir else None,
        run_dirs=[str(path) for path in run_dirs], diagnostic_dirs=[str(path) for path in diagnostic_dirs], comments_path=str(comments_path) if comments_path else None,
        input_hashes=input_hashes, code=code_identity(), warnings=warnings,
        output_hashes={str(path.relative_to(output_dir)): file_hash(path) for path in sorted(output_dir.rglob("*")) if path.is_file()})
    write_json(output_dir / "manifest.json", manifest)
    return output_dir / "index.html"

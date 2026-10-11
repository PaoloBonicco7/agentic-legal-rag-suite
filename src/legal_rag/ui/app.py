"""Streamlit calls the same engine as the CLI; no conversation state."""

import argparse
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st

from legal_rag.artifacts import read_json
from legal_rag.contracts import EngineConfig
from legal_rag.runtime import create_engine, provider_config


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index')
    parser.add_argument('--corpus')
    parser.add_argument('--provider', default='utopia')
    parser.add_argument('--model')
    parser.add_argument('--profile', default='hybrid')
    parser.add_argument('--pricing', default='configs/openai-pricing.json')
    parser.add_argument('--ledger', default='data/budget/openai.json')
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--budget-eur', type=float, default=20)
    parser.add_argument('--budget-reserve-eur', type=float, default=4)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--context-window-tokens', type=int)
    parser.add_argument('--context-tokens', type=int, default=4096)
    parser.add_argument('--context-units', type=int, default=12)
    parser.add_argument('--include-ceased', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


@st.cache_resource
def load_engine(values):
    config = provider_config(values['provider'], values['model'], pricing_path=values['pricing'],
                             max_output_tokens=values['max_output_tokens'], timeout_seconds=values['timeout'],
                             budget_eur=values['budget_eur'], budget_reserve_eur=values['budget_reserve_eur'],
                             retries=values['retries'],
                             context_window_tokens=values.get('context_window_tokens'))
    retrieval = EngineConfig(profile=values['profile'], context_tokens=values['context_tokens'],
                             context_units=values['context_units'], exclude_ceased=not values['include_ceased'])
    return create_engine(config, retrieval, index_dir=values['index'], corpus_dir=values['corpus'],
                         pricing_path=values['pricing'], ledger_path=values['ledger'], local_files_only=values['local_files_only'])


def source_context(values):
    """Read provenance without opening an index or loading a model."""
    values = dict(values)
    index = read_json(Path(values['index']) / 'manifest.json') if values['index'] else None
    if not values['corpus'] and index:
        values['corpus'] = index['corpus_dir']
    corpus = read_json(Path(values['corpus']) / 'manifest.json') if values['corpus'] else None
    if corpus and not corpus.get('snapshot_id'):
        raise ValueError('Corpus snapshot identity is missing')
    if index and (corpus is None or index['snapshot_id'] != corpus['snapshot_id']):
        raise ValueError('Corpus and index snapshots differ')
    return values, corpus


def main():
    args = arguments()
    st.set_page_config(page_title="Leggi della Valle d’Aosta", page_icon='📚', layout='wide')
    st.title("Leggi della Valle d’Aosta")
    st.write('Formula una domanda. Ogni risposta viene elaborata indipendentemente dalle precedenti.')
    try:
        values, manifest = source_context(vars(args))
    except Exception:
        st.error('Impossibile leggere i dati del corpus. Verifica i percorsi e la compatibilità dell’indice.')
        return
    if manifest:
        st.caption('Snapshot: ' + manifest['snapshot_id'])
        observed = manifest.get('snapshot_date')
        if observed:
            st.caption(f"Fonti acquisite fino al {observed}. La data di acquisizione non certifica l’applicabilità di ogni disposizione.")
        else:
            st.warning('Corpus storico: la data di acquisizione originale è sconosciuta.')
    else:
        st.caption('Snapshot: non specificato. Data di acquisizione: non disponibile.')
    if args.include_ceased:
        st.info('Questa sessione include anche disposizioni con cessazione confermata.')
    if args.profile == 'no_rag':
        st.info('Risposta basata sulla conoscenza del modello, senza consultazione delle fonti.')
    with st.form('question'):
        question = st.text_area('La tua domanda', placeholder='Quali sono le competenze del Consiglio regionale?', height=110)
        submitted = st.form_submit_button('Cerca la risposta', type='primary')
    if not submitted:
        return
    if not question.strip():
        st.warning('Scrivi una domanda prima di avviare la ricerca.')
        return
    try:
        with st.spinner('Consultazione delle fonti e preparazione della risposta…'):
            engine, _ = load_engine(values)
            result = engine.answer(question.strip())
    except Exception as error:
        st.error(f'Impossibile completare la ricerca ({type(error).__name__}). Verifica la configurazione e riprova.')
        return
    if result.status == 'error':
        st.error('La risposta non è stata completata. Verifica la configurazione e riprova.')
    else:
        if result.status == 'insufficient_evidence':
            st.warning('Le evidenze disponibili non bastano a rispondere.')
        elif result.status == 'uncertain':
            st.warning('La risposta contiene elementi incerti.')
        st.write(result.answer)
    by_id = {item.context_id: item for item in result.context.evidence}
    if result.citations:
        st.subheader('Fonti citate')
    for citation in result.citations:
        evidence = by_id.get(citation.context_id)
        if evidence is None:
            st.warning('Una citazione non corrisponde agli estratti disponibili.')
            continue
        with st.expander(f'{citation.context_id} — {evidence.chunk.title} · articolo {evidence.chunk.article_label or "non strutturato"}'):
            st.write(citation.quote)
            try:
                url = urlsplit(citation.source_url)
                safe_link = url.scheme in {'http', 'https'} and bool(url.netloc)
            except ValueError:
                safe_link = False
            if safe_link:
                st.link_button('Apri la fonte ufficiale', citation.source_url)
            else:
                st.warning('Il collegamento della fonte non è disponibile.')
    with st.expander('Dettaglio della ricerca'):
        st.write(f'{len(result.candidates)} candidati; {len(result.context.evidence)} estratti forniti al modello.')
        for item in result.context.evidence:
            st.text(item.rendered)
        if result.context.omissions:
            st.write('Estratti esclusi e motivazioni')
            st.dataframe(result.context.omissions, hide_index=True)


if __name__ == '__main__':
    main()

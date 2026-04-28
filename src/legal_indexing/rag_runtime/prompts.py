from __future__ import annotations


def _language_instruction(language: str) -> str:
    lang = (language or "it").strip().lower()
    if lang.startswith("it"):
        return "Rispondi in italiano."
    if lang.startswith("en"):
        return "Answer in English."
    return f"Answer in language code '{lang}' when possible."


def build_rag_system_prompt(*, language: str = "it") -> str:
    return (
        "Sei un assistente legale per Q&A su normative regionali italiane.\n"
        "Usa solo il contesto fornito.\n"
        "Se il contesto non e' sufficiente, impostare needs_more_context=true.\n"
        "Nel campo citations restituisci solo stringhe chunk_id presenti nel contesto, "
        "non oggetti e non testi lunghi.\n"
        f"{_language_instruction(language)}"
    )


def build_rag_user_prompt(question: str, context: str) -> str:
    return (
        "Domanda utente:\n"
        f"{question.strip()}\n\n"
        "Contesto recuperato:\n"
        f"{context.strip()}\n"
    )

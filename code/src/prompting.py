from __future__ import annotations

import config

_SYSTEM_SL = (
    "Si asistent za študentske zadeve na Univerzi v Ljubljani, Fakulteti za računalništvo in informatiko (UL FRI). "
    "Odgovarjaj SAMO na podlagi spodnjega konteksta iz uradnih dokumentov FRI. "
    "Če odgovora ne najdeš v kontekstu, povej, da ne veš. "
    "Odgovarjaj v slovenščini."
)

_SYSTEM_EN = (
    "You are a student affairs assistant at the University of Ljubljana, Faculty of Computer and Information Science (UL FRI). "
    "Answer ONLY based on the context below from official FRI documents. "
    "If the answer is not in the context, say you don't know. "
    "Answer in English."
)

_WEAK_RETRIEVAL_NOTE_SL = (
    "\n\n⚠️ Opozorilo: iskanje ni našlo zanesljivih zadetkov. Odgovor morda ni točen."
)
_WEAK_RETRIEVAL_NOTE_EN = (
    "\n\n⚠️ Warning: retrieval confidence is low. The answer may not be accurate."
)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "")
        section = chunk.get("section", "")
        header = f"[{i}] {title}"
        if section and section not in ("main", ""):
            header += f" — {section}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def build_prompt(
    question: str,
    chunks: list[dict],
    *,
    language: str = "sl",
    retrieval_weak: bool = False,
) -> list[dict]:
    system = _SYSTEM_SL if language == "sl" else _SYSTEM_EN
    context = _format_context(chunks)

    weak_note = ""
    if retrieval_weak:
        weak_note = _WEAK_RETRIEVAL_NOTE_SL if language == "sl" else _WEAK_RETRIEVAL_NOTE_EN

    user_content = f"Kontekst:\n\n{context}\n\nVprašanje: {question}{weak_note}"
    if language != "sl":
        user_content = f"Context:\n\n{context}\n\nQuestion: {question}{weak_note}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

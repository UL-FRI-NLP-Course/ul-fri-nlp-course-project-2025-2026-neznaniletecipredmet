from __future__ import annotations

_SYSTEM_SL = (
    "Si asistent za študentske zadeve na Univerzi v Ljubljani, Fakulteti za računalništvo in informatiko (UL FRI). "
    "Odgovarjaj SAMO na podlagi spodnjega konteksta iz uradnih dokumentov FRI. "
    "Prioritiziraj informacije z uradne spletne strani FRI (https://fri.uni-lj.si) in iz uradnih dokumentov FRI, ki so del konteksta. "
    "Če odgovora ne najdeš v kontekstu, povej, da ne veš. "
    "Odgovarjaj v slovenščini."
)

_SYSTEM_EN = (
    "You are a student affairs assistant at the University of Ljubljana, Faculty of Computer and Information Science (UL FRI). "
    "Answer ONLY based on the context below from official FRI documents. "
    "Prioritize information from the official FRI website (https://fri.uni-lj.si) and from official FRI documents included in the context. "
    "If the answer is not in the context, say you don't know. "
    "Answer in English."
)

_RESPONSE_RULES_SL = (
    "Oblika odgovora: najprej podaj kratek neposreden odgovor (1-2 stavka). "
    "Nato dodaj razdelek 'Dokazi iz konteksta:' s kratkimi alinejami. "
    "Vsako dejstveno trditev podpri s citatom na enega ali vec kontekstnih odlomkov v obliki [1], [2]. "
    "Ce se odlomki v kontekstu med seboj razlikujejo ali si nasprotujejo, to izrecno povej in prikazi obe trditvi s citati. "
    "Ne ugibaj in ne dodajaj zunanjih informacij."
)

_RESPONSE_RULES_EN = (
    "Answer format: first give a short direct answer (1-2 sentences). "
    "Then add a section 'Evidence from context:' with short bullet points. "
    "Support every factual claim with citations to one or more context chunks in the form [1], [2]. "
    "If context chunks disagree or conflict, say that explicitly and present both claims with citations. "
    "Do not guess or add outside information."
)

_WEAK_RETRIEVAL_SYSTEM_SL = (
    "Zanesljivost iskanja je nizka. Bodi zelo previden: ce nimas jasnih dokazov v kontekstu, reci da ne ves."
)

_WEAK_RETRIEVAL_SYSTEM_EN = (
    "Retrieval confidence is low. Be conservative: if evidence in the context is not clear, say you don't know."
)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "")
        section = chunk.get("section", "")
        url = chunk.get("url", "")
        header = f"[{i}] {title}"
        if section and section not in ("main", ""):
            header += f" — {section}"
        body = chunk["text"]
        if url:
            body = f"Source: {url}\n{body}"
        parts.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(parts)


def build_prompt(
    question: str,
    chunks: list[dict],
    *,
    language: str = "sl",
    retrieval_weak: bool = False,
) -> list[dict]:
    is_sl = language == "sl"
    system = (_SYSTEM_SL + " " + _RESPONSE_RULES_SL) if is_sl else (_SYSTEM_EN + " " + _RESPONSE_RULES_EN)
    if retrieval_weak:
        system += " " + (_WEAK_RETRIEVAL_SYSTEM_SL if is_sl else _WEAK_RETRIEVAL_SYSTEM_EN)

    context = _format_context(chunks)

    user_content = f"Kontekst:\n\n{context}\n\nVprašanje: {question}"
    if not is_sl:
        user_content = f"Context:\n\n{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

"""Bootstrap evaluation questions and labels.

Appends additional evaluation questions to the evaluation set and 
heuristically populates `relevant_chunk_ids` and `relevant_doc_ids` 
for unlabeled questions using exact-question substring and keyword overlap.

Usage:
  python scripts/bootstrap_eval_questions.py --run Test --hybrid --rerank --rerank-candidate-k 20
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap eval questions + auto-label chunk ids")
    p.add_argument("--run", default=None, help="Run name under config.RUNS_DIR")
    p.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (BM25 + dense)")
    p.add_argument("--rerank", action="store_true", help="Rerank candidates with cross-encoder")
    p.add_argument("--rerank-model", default=None, help="Override config.RERANK_MODEL")
    p.add_argument("--rerank-candidate-k", type=int, default=None, help="Override config.RERANK_CANDIDATE_K")
    p.add_argument("--retrieve-k", type=int, default=12, help="How many chunks to retrieve per question")
    p.add_argument("--label-k", type=int, default=2, help="How many chunk_ids to auto-label per question")
    p.add_argument(
        "--append-questions",
        action="store_true",
        help="Append a batch of additional questions (default: on)",
    )
    p.add_argument(
        "--no-append-questions",
        action="store_true",
        help="Do not append additional questions",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing relevant_* labels (default: only fill missing)",
    )
    return p.parse_args()


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _question_in_text(question: str, text: str) -> bool:
    q = _norm(question)
    t = _norm(text)
    if not q or not t:
        return False
    # Require at least a few words to reduce false positives.
    if len(_WORD_RE.findall(q)) < 4:
        return False
    return q in t


def _keyword_hits(keywords: list[str], text: str) -> int:
    if not keywords:
        return 0
    t = _norm(text)
    hits = 0
    for kw in keywords:
        kw = _norm(kw)
        if kw and kw in t:
            hits += 1
    return hits


def _choose_chunk_ids(
    chunks: list[dict],
    question: str,
    keywords: list[str],
    *,
    label_k: int,
) -> list[str]:
    scored = []
    for rank, c in enumerate(chunks):
        cid = c.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            continue
        text = c.get("text", "")
        in_text = _question_in_text(question, text)
        kw_hits = _keyword_hits(keywords, text)
        score = c.get("score")
        try:
            score_f = float(score) if score is not None else 0.0
        except Exception:
            score_f = 0.0
        scored.append((1 if in_text else 0, kw_hits, score_f, -rank, cid))

    scored.sort(reverse=True)
    out: list[str] = []
    for _, __, ___, ____, cid in scored:
        if cid not in out:
            out.append(cid)
        if len(out) >= int(label_k):
            break
    return out


def _doc_ids_from_chunks(chunks: list[dict], chunk_ids: list[str]) -> list[str]:
    doc_ids: list[str] = []
    chunk_id_set = set(chunk_ids)
    for c in chunks:
        cid = c.get("chunk_id")
        did = c.get("doc_id")
        if cid in chunk_id_set and isinstance(did, str) and did and did not in doc_ids:
            doc_ids.append(did)
    return doc_ids


def _default_questions() -> list[dict]:
    # Keep these short and close to the existing topics.
    in_corpus = [
        {
            "question": "Kako se odjavim od izpita?",
            "language": "sl",
            "expected_keywords": ["odjav", "izpit"],
        },
        {
            "question": "Do kdaj je možna prijava na izpit preko STUDIS?",
            "language": "sl",
            "expected_keywords": ["STUDIS", "prijav", "rok"],
        },
        {
            "question": "Ali lahko grem na izpit, če nisem opravil vaj?",
            "language": "sl",
            "expected_keywords": ["vaje", "izpit"],
        },
        {
            "question": "Kako dobim potrdilo o vpisu (Certificate of Registration)?",
            "language": "sl",
            "expected_keywords": ["potrdilo", "vpis"],
        },
        {
            "question": "Kje najdem vpisna navodila za višji letnik?",
            "language": "sl",
            "expected_keywords": ["vpis", "navodila", "letnik"],
        },
        {
            "question": "Koliko kreditnih točk potrebujem za vpis v 2. letnik?",
            "language": "sl",
            "expected_keywords": ["kredit", "točk", "vpis", "2."],
        },
        {
            "question": "Kje najdem informacije o magistrskem delu in zagovorih?",
            "language": "sl",
            "expected_keywords": ["magistr", "zagovor"],
        },
        {
            "question": "Kakšni so pogoji za ponavljanje 1. letnika?",
            "language": "sl",
            "expected_keywords": ["ponavlj", "letnik"],
        },
        {
            "question": "Kje je glavni vhod na UL FRI in kaj je v pritličju?",
            "language": "sl",
            "expected_keywords": ["glavni vhod", "pritlič"],
        },
        {
            "question": "What is STUDIS and where do I print a Certificate of Registration?",
            "language": "en",
            "expected_keywords": ["STUDIS", "Certificate of Registration"],
        },
        {
            "question": "Where can I find information about thesis (diploma) submission and defense?",
            "language": "en",
            "expected_keywords": ["thesis", "defense", "diploma"],
        },
        {
            "question": "Kako se prijavim na predmet vnaprej (opravljanje predmetov vnaprej)?",
            "language": "sl",
            "expected_keywords": ["predmet", "vnaprej", "prošnja"],
        },
        {
            "question": "Koliko časa veljajo laboratorijske vaje pri predmetu?",
            "language": "sl",
            "expected_keywords": ["vaje", "veljavnost"],
        },
        {
            "question": "Kje najdem študijski koledar UL FRI?",
            "language": "sl",
            "expected_keywords": ["študijski koledar"],
        },
        {
            "question": "Kako lahko podaljšam status študenta iz upravičenih razlogov?",
            "language": "sl",
            "expected_keywords": ["podaljš", "status", "upravičen"],
        },
        {
            "question": "Kaj pomeni dodatno leto in kdaj ga lahko koristim?",
            "language": "sl",
            "expected_keywords": ["dodatno leto", "status"],
        },
        {
            "question": "Kje lahko najdem pravilnike, vloge in cenike UL FRI?",
            "language": "sl",
            "expected_keywords": ["pravilniki", "vloge", "ceniki"],
        },
        {
            "question": "Kje so objavljeni zagovori diplom in magisterijev?",
            "language": "sl",
            "expected_keywords": ["zagovor", "objave"],
        },
        {
            "question": "Kakšne so uradne ure študentskega referata?",
            "language": "sl",
            "expected_keywords": ["študentski referat", "uradne ure"],
        },
        {
            "question": "Kako stopim v stik s študentskim referatom (email/telefon)?",
            "language": "sl",
            "expected_keywords": ["referat", "telefon", "email"],
        },
        {
            "question": "Kje se nahaja glavni vhod na fakulteto in kaj je v avli?",
            "language": "sl",
            "expected_keywords": ["glavni vhod", "avla"],
        },
        {
            "question": "Kje je recepcija in kje je fotokopirnica na UL FRI?",
            "language": "sl",
            "expected_keywords": ["recepcija", "fotokopirnica"],
        },
        {
            "question": "Kje je knjižnica na UL FRI/FKKT in kakšne so uradne ure?",
            "language": "sl",
            "expected_keywords": ["knjižnica", "uradne ure"],
        },
        {
            "question": "Kako poteka Erasmus+ izmenjava na UL FRI?",
            "language": "sl",
            "expected_keywords": ["Erasmus"],
        },
        {
            "question": "Katere dokumente potrebujem za Erasmus izmenjavo?",
            "language": "sl",
            "expected_keywords": ["Erasmus", "prijava"],
        },
        {
            "question": "Kakšni so pogoji za prijavo na mednarodno izmenjavo?",
            "language": "sl",
            "expected_keywords": ["izmenjava", "pogoji"],
        },
        {
            "question": "Kakšni so roki za vpis v 1. letnik dodiplomskega študija?",
            "language": "sl",
            "expected_keywords": ["vpis", "1.", "letnik"],
        },
        {
            "question": "Kdaj poteka vpis v višji letnik oziroma dodatno leto?",
            "language": "sl",
            "expected_keywords": ["vpis", "višji letnik", "dodatno leto"],
        },
        {
            "question": "Kdaj poteka vpis za ponavljalce?",
            "language": "sl",
            "expected_keywords": ["vpis", "ponavlj"],
        },
        {
            "question": "Kako oddam prošnjo za vpis v izjemnih primerih?",
            "language": "sl",
            "expected_keywords": ["prošnja", "izjemnih"],
        },
        {
            "question": "What are the office hours of the student office (Študentski referat)?",
            "language": "en",
            "expected_keywords": ["office hours", "student"],
        },
        {
            "question": "Where can I find rules/regulations and forms (pravilniki, vloge, ceniki) at UL FRI?",
            "language": "en",
            "expected_keywords": ["rules", "forms"],
        },
        {
            "question": "Where can I find information about enrollment (vpis) for 2025/2026?",
            "language": "en",
            "expected_keywords": ["enrollment", "vpis"],
        },
    ]

    negatives = [
        {
            "question": "Kakšna bo cena bencina v Sloveniji naslednji teden?",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Recept za potico z orehi",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Who won the NBA finals in 2020?",
            "language": "en",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Kako popravim Windows, če se ne zažene?",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Kdaj je naslednja tekma Olimpije in kakšen bo rezultat?",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Najboljši recept za pico napoletano",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Kako investirati v delnice (osnove)?",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "What is the current Bitcoin price?",
            "language": "en",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
        {
            "question": "Prevedi ta stavek v japonščino",
            "language": "sl",
            "expected_keywords": [],
            "is_negative": True,
            "non_in_corpus": True,
        },
    ]

    out: list[dict] = []
    for q in in_corpus + negatives:
        q = dict(q)
        q.setdefault("expected_keywords", [])
        q.setdefault("relevant_doc_ids", [])
        q.setdefault("relevant_chunk_ids", [])
        out.append(q)
    return out


def main() -> None:
    args = parse_args()
    append_questions = bool(args.append_questions) or not bool(args.no_append_questions)

    config.apply_run(getattr(args, "run", None))

    if not config.EVAL_QUESTIONS_FILE.exists():
        print(f"Eval file not found: {config.EVAL_QUESTIONS_FILE}")
        raise SystemExit(1)

    questions = read_jsonl(config.EVAL_QUESTIONS_FILE)
    existing_q = {_norm(q.get("question", "")) for q in questions}

    if append_questions:
        added = 0
        for q in _default_questions():
            q_text = _norm(q.get("question", ""))
            if q_text and q_text not in existing_q:
                questions.append(q)
                existing_q.add(q_text)
                added += 1
        print(f"Appended {added} new questions")

    # Lazy import: retrieval pulls heavier deps.
    try:
        from src.retrieval import load, retrieve
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}.", file=sys.stderr)
        print("Install dependencies, e.g.: pip install -r code/requirements.txt", file=sys.stderr)
        raise

    print("Loading index...")
    load()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    updated = 0
    for q in questions:
        if bool(q.get("needs_review", False)):
            # User-marked: do not auto-label.
            continue
        if bool(q.get("is_negative", False)) or bool(q.get("non_in_corpus", False)):
            # Ensure negatives never have relevance labels.
            if args.overwrite:
                q["relevant_doc_ids"] = []
                q["relevant_chunk_ids"] = []
            continue

        question = (q.get("question") or "").strip()
        if not question:
            continue

        relevant_doc_ids = q.get("relevant_doc_ids") or []
        relevant_chunk_ids = q.get("relevant_chunk_ids") or []

        needs_doc = args.overwrite or not relevant_doc_ids
        needs_chunk = args.overwrite or not relevant_chunk_ids
        if not (needs_doc or needs_chunk):
            continue

        keywords = q.get("expected_keywords") or []

        r = retrieve(
            question,
            top_k=int(args.retrieve_k),
            use_hybrid=bool(args.hybrid),
            use_rerank=bool(args.rerank),
            rerank_model=args.rerank_model,
            rerank_candidate_k=args.rerank_candidate_k,
        )
        chunks = r.get("chunks", [])
        if not chunks:
            continue

        if needs_chunk:
            if relevant_doc_ids:
                # If doc_ids were already chosen manually, map them to chunk_ids.
                chunk_ids: list[str] = []
                wanted = [d for d in relevant_doc_ids if isinstance(d, str) and d]
                for did in wanted:
                    for c in chunks:
                        if c.get("doc_id") == did:
                            cid = c.get("chunk_id")
                            if isinstance(cid, str) and cid and cid not in chunk_ids:
                                chunk_ids.append(cid)
                                break
                if not chunk_ids:
                    chunk_ids = _choose_chunk_ids(chunks, question, keywords, label_k=int(args.label_k))
            else:
                chunk_ids = _choose_chunk_ids(chunks, question, keywords, label_k=int(args.label_k))

            q["relevant_chunk_ids"] = chunk_ids
            relevant_chunk_ids = chunk_ids

        if needs_doc:
            if relevant_chunk_ids:
                q["relevant_doc_ids"] = _doc_ids_from_chunks(chunks, relevant_chunk_ids)
            else:
                # Fall back: choose doc_ids from top chunks.
                doc_ids: list[str] = []
                for c in chunks[: int(args.label_k)]:
                    did = c.get("doc_id")
                    if isinstance(did, str) and did and did not in doc_ids:
                        doc_ids.append(did)
                q["relevant_doc_ids"] = doc_ids

        q["auto_labeled"] = True
        q["auto_labeled_at"] = now
        updated += 1

    if updated:
        write_jsonl(questions, config.EVAL_QUESTIONS_FILE)
        print(f"Updated {updated} questions -> {config.EVAL_QUESTIONS_FILE}")
    else:
        print("No questions updated.")


if __name__ == "__main__":
    main()

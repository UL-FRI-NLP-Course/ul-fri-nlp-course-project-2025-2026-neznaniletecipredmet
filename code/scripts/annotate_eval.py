"""Interactive annotation script for retrieval evaluation.

Assists with adding `relevant_doc_ids` and `relevant_chunk_ids` to
<run>/eval/questions.jsonl by running the embedding model and 
displaying the top retrieved chunks for each question.

Example:
  python scripts/annotate_eval.py --run Test --top-k 10 --hybrid
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate retrieval evaluation questions")
    p.add_argument("--run", default=None, help="Run name under config.RUNS_DIR")
    p.add_argument("--top-k", type=int, default=13, help="How many retrieved chunks to show")
    p.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (BM25 + dense)")
    p.add_argument("--rerank", action="store_true", help="Rerank candidates with cross-encoder")
    p.add_argument("--rerank-model", default=None, help="Override config.RERANK_MODEL")
    p.add_argument("--rerank-candidate-k", type=int, default=None, help="Override config.RERANK_CANDIDATE_K")
    p.add_argument(
        "--doc-only",
        action="store_true",
        help="Only store relevant_doc_ids (do not store relevant_chunk_ids)",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Only annotate questions that have empty relevant_doc_ids",
    )
    p.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="How many characters of each chunk to show (ignored with --full-text)",
    )
    p.add_argument(
        "--full-text",
        action="store_true",
        help="Print full chunk text (can be very long; consider --top-k 3)",
    )
    return p.parse_args()


def _format_chunk_text(text: str, *, preview_chars: int, full_text: bool) -> str:
    raw = (text or "").strip("\n")
    if full_text:
        return raw

    t = raw.replace("\n", " ").strip()
    if preview_chars <= 0:
        preview_chars = 1
    if len(t) <= preview_chars:
        return t
    return t[:preview_chars] + "… (truncated; use --full-text)"


def _parse_selection(s: str, max_idx: int) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    if s.lower() in {"n", "no", "none", "skip"}:
        return []
    out: list[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
                for i in range(start, end + 1):
                    if 1 <= i <= max_idx:
                        out.append(i)
        elif part.isdigit():
            i = int(part)
            if 1 <= i <= max_idx:
                out.append(i)
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def main() -> None:
    args = parse_args()

    # Lazy imports so `--help` works even if heavy deps aren't installed yet.
    try:
        from src.retrieval import load, retrieve
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}.", file=sys.stderr)
        print("Install dependencies, e.g.: pip install -r code/requirements.txt", file=sys.stderr)
        raise

    config.apply_run(getattr(args, "run", None))

    if not config.EVAL_QUESTIONS_FILE.exists():
        print(f"Eval file not found: {config.EVAL_QUESTIONS_FILE}")
        raise SystemExit(1)

    questions = read_jsonl(config.EVAL_QUESTIONS_FILE)
    if not questions:
        print("No questions to annotate.")
        return

    print(f"Loaded {len(questions)} questions from {config.EVAL_QUESTIONS_FILE}")

    print("Loading index...")
    load()

    updated = 0

    for idx, q in enumerate(questions, 1):
        if bool(q.get("is_negative", False)):
            continue

        existing = q.get("relevant_doc_ids", [])
        if args.only_missing and existing:
            continue

        question = q.get("question", "").strip()
        if not question:
            continue

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(questions)}] {question}")

        r = retrieve(
            question,
            top_k=int(args.top_k),
            use_hybrid=bool(args.hybrid),
            use_rerank=bool(args.rerank),
            rerank_model=args.rerank_model,
            rerank_candidate_k=args.rerank_candidate_k,
        )

        chunks = r.get("chunks", [])
        if not chunks:
            print("No chunks retrieved.")
            continue

        for i, c in enumerate(chunks, 1):
            title = (c.get("title") or "").strip()
            section = (c.get("section") or "").strip()
            url = (c.get("url") or "").strip()
            doc_id = c.get("doc_id")
            chunk_id = c.get("chunk_id")
            score = c.get("score")
            pre = c.get("pre_rerank_score")

            src = title
            if section and section not in ("main", ""):
                src = f"{title} — {section}" if title else section

            score_str = f"score={float(score):.3f}" if score is not None else "score=?"
            if pre is not None:
                score_str += f" pre={float(pre):.3f}"

            print(f"\n({i}) {src}")
            print(f"    doc_id={doc_id} chunk_id={chunk_id} {score_str}")
            if url:
                print(f"    {url}")
            text = c.get("text", "")
            formatted = _format_chunk_text(text, preview_chars=int(args.preview_chars), full_text=bool(args.full_text))
            if args.full_text:
                # Keep newlines when showing full text; indent consistently.
                print(textwrap.indent(formatted, "    "))
            else:
                print(f"    {formatted}")

        max_idx = len(chunks)
        prompt = "Select relevant chunk numbers (e.g. 1,3 or 2-4). Enter to skip: "
        sel = _parse_selection(input(prompt), max_idx=max_idx)
        if not sel:
            continue

        selected_chunks = [chunks[i - 1] for i in sel]
        doc_ids = []
        chunk_ids = []
        for c in selected_chunks:
            did = c.get("doc_id")
            cid = c.get("chunk_id")
            if isinstance(did, str) and did and did not in doc_ids:
                doc_ids.append(did)
            if isinstance(cid, str) and cid and cid not in chunk_ids:
                chunk_ids.append(cid)

        q["relevant_doc_ids"] = doc_ids
        if not args.doc_only:
            q["relevant_chunk_ids"] = chunk_ids

        updated += 1
        msg = f"Saved labels: relevant_doc_ids={doc_ids}"
        if not args.doc_only:
            msg += f" | relevant_chunk_ids={chunk_ids}"
        print(msg)

    if updated:
        write_jsonl(questions, config.EVAL_QUESTIONS_FILE)
        print(f"\nUpdated {updated} questions -> {config.EVAL_QUESTIONS_FILE}")
    else:
        print("\nNo questions updated.")


if __name__ == "__main__":
    main()

"""Audit reference answers in an evaluation question file against a corpus.

For each question with a non-empty reference_answer, retrieves a broad pool
(default top-30 hybrid) of candidate chunks and asks Claude whether the
reference is fully supported by those chunks. Emits a verdict per question
plus an optional grounded rewrite that the apply_reference_audit.py script
can merge into a cleaned eval set.

This addresses failure mode B identified during judge calibration: bootstrap
references that contain claims (deadlines, prices, locations, names) which
do not appear anywhere in the crawled corpus. The audit is the gating step
before any embedder/chunk/retrieval comparison, because every downstream
metric depends on the gold-label quality.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/audit_references.py \
      --run 2026-05-20__full__fri4_ul2__v1 \
      --questions ../code/questions_full.json \
      --top-k 30 \
      --hybrid \
      --out ../code/questions_audited.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude-based reference-answer audit (B.5)")
    p.add_argument("--run", default=None, help="Run/dataset name (uses its FAISS index)")
    p.add_argument(
        "--questions",
        required=True,
        help="Path to JSON array of question objects (e.g. questions_full.json)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <questions stem>_audited.json next to input)",
    )
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Audit model (default: claude-sonnet-4-6 - this is the quality bottleneck)")
    p.add_argument("--top-k", type=int, default=30,
                   help="Broad retrieval pool size for the audit (default: 30)")
    p.add_argument("--hybrid", action="store_true", default=True,
                   help="Use hybrid retrieval (BM25 + dense). Default: on for the audit.")
    p.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                   help="Disable hybrid retrieval (use dense only)")
    p.add_argument("--rerank", action="store_true",
                   help="Cross-encoder rerank the candidate pool")
    p.add_argument("--max-chunk-chars", type=int, default=900,
                   help="Truncate each chunk to N chars before sending to the auditor")
    p.add_argument("--max-tokens", type=int, default=1500,
                   help="Max output tokens per audit call")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 for reproducibility)")
    p.add_argument("--limit", type=int, default=None,
                   help="Audit only the first N questions (for smoke testing)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Sleep N seconds between API calls (default: 0)")
    p.add_argument("--skip-negatives", action="store_true", default=True,
                   help="Skip is_negative questions (they have no factual reference). Default: on.")
    return p.parse_args()


def _resolve_questions_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    for cand in (
        config.BASE_DIR / name_or_path,
        Path(__file__).resolve().parents[2] / name_or_path,
    ):
        if cand.exists():
            return cand
    return p


def _format_chunks(chunks: list[dict], *, max_chars: int) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        title = (c.get("title") or "").strip()
        section = (c.get("section") or "").strip()
        url = (c.get("url") or "").strip()
        text = (c.get("text") or c.get("preview") or "").strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        header = f"[{i}] {title}".strip()
        if section and section not in ("main", ""):
            header += f" — {section}"
        if url:
            header += f"  ({url})"
        parts.append(header + "\n" + text)
    return "\n\n".join(parts) if parts else "(no chunks)"


def _system_prompt() -> str:
    return (
        "You are a strict fact-checker for a Slovenian/English Retrieval-Augmented "
        "Generation system for UL FRI students. You are given a QUESTION, a "
        "REFERENCE_ANSWER (claimed gold), and a list of CANDIDATE_CHUNKS retrieved "
        "broadly from the corpus. Your job is to determine whether the reference "
        "answer is fully supported by the chunks. "
        "Use a strict reading: every concrete claim (deadlines, prices, names, "
        "locations, room numbers, contact details, URLs, numeric thresholds) must "
        "appear in the chunks. General phrasing is okay if its substance is in the "
        "chunks. If a claim is not in any chunk, mark it unsupported. "
        "Output strictly valid JSON, no prose outside the JSON."
    )


def _user_prompt(question: str, reference: str, chunks_text: str) -> str:
    schema = (
        '{\n'
        '  "verdict": "supported" | "partial" | "unsupported",\n'
        '  "supported_claims": [<short strings copied from reference>],\n'
        '  "unsupported_claims": [<short strings copied from reference>],\n'
        '  "suggested_rewrite": <string rewriting the reference using ONLY chunk facts, '
        'or null if no answerable content exists in the chunks>,\n'
        '  "reasoning": <1-3 sentences>\n'
        '}'
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE_ANSWER:\n{reference}\n\n"
        f"CANDIDATE_CHUNKS:\n{chunks_text}\n\n"
        f"Decide whether REFERENCE_ANSWER is fully supported by CANDIDATE_CHUNKS, "
        f"using this exact JSON shape:\n{schema}\n\n"
        "Rules:\n"
        "- 'supported' = every concrete claim in the reference appears in some chunk.\n"
        "- 'partial' = some claims are supported, others (specific numbers/names/locations) are not.\n"
        "- 'unsupported' = the chunks contain no information addressing the question, "
        "OR all specific claims are absent.\n"
        "- 'suggested_rewrite' MUST contain only facts present in the chunks; if the chunks "
        "do not address the question, set suggested_rewrite to null.\n"
        "- Quote unsupported_claims verbatim (or near-verbatim) from the reference.\n"
        "- Be strict about specific numbers, deadlines, prices, names, room numbers, dates."
    )


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in audit output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _make_client():
    try:
        import anthropic
    except ImportError:
        print("Missing dependency 'anthropic'. Install: pip install anthropic tenacity", file=sys.stderr)
        raise
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        raise SystemExit(2)
    return anthropic.Anthropic()


def _call_auditor(client, *, model: str, system: str, user: str,
                  max_tokens: int, temperature: float) -> str:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _do_call() -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        parts = []
        for block in msg.content:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        return "\n".join(parts)

    return _do_call()


def _summarize(audited: list[dict]) -> dict:
    n = len(audited)
    if n == 0:
        return {"n": 0}
    by_verdict: dict[str, int] = {}
    n_with_rewrite = 0
    for a in audited:
        v = (a.get("audit") or {}).get("verdict") or "unknown"
        by_verdict[v] = by_verdict.get(v, 0) + 1
        if (a.get("audit") or {}).get("suggested_rewrite"):
            n_with_rewrite += 1
    return {
        "n": n,
        "verdict_counts": by_verdict,
        "with_suggested_rewrite": n_with_rewrite,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.apply_run(args.run)

    questions_path = _resolve_questions_path(args.questions)
    if not questions_path.exists():
        print(f"Questions file not found: {questions_path}", file=sys.stderr)
        raise SystemExit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"{questions_path}: expected a JSON array of question objects", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loaded {len(data)} questions from {questions_path}")

    try:
        from src.retrieval import load as load_index, retrieve
    except ModuleNotFoundError as e:
        print(f"Cannot import retrieval (missing dependency): {e}", file=sys.stderr)
        raise SystemExit(1)

    print("Loading index for audit retrieval...")
    load_index()

    client = _make_client()
    model = args.model
    system = _system_prompt()

    audited: list[dict] = []
    failures: list[dict] = []

    rows_to_audit = data[: int(args.limit)] if args.limit else data
    print(f"Auditing {len(rows_to_audit)} questions with model={model}, top-k={args.top_k}, hybrid={args.hybrid}, rerank={args.rerank}")

    for i, q in enumerate(rows_to_audit, 1):
        if not isinstance(q, dict):
            audited.append(q)
            continue

        out_record = dict(q)
        question = (q.get("question") or "").strip()
        ref = (q.get("reference_answer") or "").strip()
        is_negative = bool(q.get("is_negative", False))

        if is_negative and args.skip_negatives:
            out_record["audit"] = {
                "verdict": "skipped_negative",
                "reasoning": "Negative (out-of-scope) question; reference is not factual.",
            }
            audited.append(out_record)
            if i % 5 == 0 or i == len(rows_to_audit):
                print(f"  [{i}/{len(rows_to_audit)}] negative skipped")
            continue

        if not question or not ref:
            out_record["audit"] = {
                "verdict": "skipped_no_reference",
                "reasoning": "Missing question or reference_answer.",
            }
            audited.append(out_record)
            if i % 5 == 0 or i == len(rows_to_audit):
                print(f"  [{i}/{len(rows_to_audit)}] skipped (no reference)")
            continue

        try:
            r = retrieve(
                question,
                top_k=int(args.top_k),
                use_hybrid=bool(args.hybrid),
                use_rerank=bool(args.rerank),
            )
            chunks = r.get("chunks", []) or []
        except Exception as e:
            log.warning("[%d] retrieval failed for %r: %s", i, question[:60], e)
            failures.append({"i": i, "question": question, "error": f"retrieval: {e}"})
            out_record["audit"] = {"verdict": "error", "reasoning": f"retrieval failed: {e}"}
            audited.append(out_record)
            continue

        chunks_text = _format_chunks(chunks, max_chars=int(args.max_chunk_chars))

        try:
            raw = _call_auditor(
                client,
                model=model,
                system=system,
                user=_user_prompt(question, ref, chunks_text),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
            )
            verdict_obj = _extract_json(raw)
        except Exception as e:
            log.warning("[%d] audit call failed for %r: %s", i, question[:60], e)
            failures.append({"i": i, "question": question, "error": f"audit: {e}"})
            out_record["audit"] = {"verdict": "error", "reasoning": f"audit call failed: {e}"}
            audited.append(out_record)
            continue

        out_record["audit"] = {
            "verdict": verdict_obj.get("verdict"),
            "supported_claims": verdict_obj.get("supported_claims") or [],
            "unsupported_claims": verdict_obj.get("unsupported_claims") or [],
            "suggested_rewrite": verdict_obj.get("suggested_rewrite"),
            "reasoning": verdict_obj.get("reasoning"),
            "model": model,
            "top_k": int(args.top_k),
            "hybrid": bool(args.hybrid),
            "rerank": bool(args.rerank),
        }
        audited.append(out_record)

        if i % 5 == 0 or i == len(rows_to_audit):
            print(f"  [{i}/{len(rows_to_audit)}] verdict={verdict_obj.get('verdict')}")

        if args.sleep:
            time.sleep(float(args.sleep))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute() and not out_path.parent.exists():
            out_path = questions_path.parent / out_path
    else:
        out_path = questions_path.with_name(questions_path.stem + "_audited.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audited, f, ensure_ascii=False, indent=2)
    print(f"Wrote -> {out_path}")

    summary = _summarize(audited)
    print("=" * 50)
    print(f"Audit model: {model}")
    print(f"Source questions: {questions_path}")
    print(f"Audited: {summary['n']} (failed: {len(failures)})")
    print(f"Verdict counts: {summary.get('verdict_counts', {})}")
    print(f"Suggested rewrites available: {summary.get('with_suggested_rewrite', 0)}")
    print("=" * 50)

    if failures:
        fail_path = out_path.with_name(out_path.stem + "_failures.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"Failures -> {fail_path}")


if __name__ == "__main__":
    main()

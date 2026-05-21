"""LLM-as-judge using Claude on RAG eval results.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/judge_answers.py --run <run> --results results_<model>.jsonl

Reads <run>/eval/<results>, sends each (question, retrieved_chunks, answer) to Claude
with a RAGAS-style rubric, writes <run>/eval/judged_<results>.jsonl plus aggregate metrics.
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
from src.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude judge for RAG answers")
    p.add_argument("--run", default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", help="Path or filename inside <run>/eval/ (existing model output)")
    src.add_argument(
        "--from-questions",
        help="JSON array of {question, reference_answer, ...} (e.g. questions_full.json). "
             "Runs retrieval on each question and judges the reference_answer as the model answer.",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=None, help=f"Judge model (default: {config.JUDGE_MODEL})")
    p.add_argument("--reference-file", default=None, help="JSON list of {question, reference_answer} (separate from --from-questions)")
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k-context", type=int, default=4)
    p.add_argument("--max-chunk-chars", type=int, default=900)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument(
        "--chunks-file",
        default=None,
        help="Optional path to chunks.jsonl (default: <run>/processed/chunks.jsonl). "
             "Used to look up chunk text by chunk_id when results lack it.",
    )
    p.add_argument("--retrieval-top-k", type=int, default=config.TOP_K, help="(only with --from-questions) top_k for retrieval")
    p.add_argument("--hybrid", action="store_true", help="(only with --from-questions) use hybrid retrieval")
    p.add_argument("--rerank", action="store_true", help="(only with --from-questions) cross-encoder rerank")
    p.add_argument("--rerank-model", default=None, help="(only with --from-questions) override config.RERANK_MODEL")
    p.add_argument("--rerank-candidate-k", type=int, default=None, help="(only with --from-questions) override config.RERANK_CANDIDATE_K")
    p.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature (default 0.0 for reproducibility)")
    return p.parse_args()


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _resolve_results_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    cand = config.EVAL_DIR / name_or_path
    if cand.exists():
        return cand
    return p


def _resolve_questions_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    for cand in (
        config.BASE_DIR / name_or_path,
        Path(__file__).resolve().parents[2] / name_or_path,
        config.EVAL_DIR / name_or_path,
    ):
        if cand.exists():
            return cand
    return p


def _rows_from_questions_file(
    path: Path,
    *,
    top_k: int,
    use_hybrid: bool,
    use_rerank: bool,
    rerank_model: str | None,
    rerank_candidate_k: int | None,
) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of question objects")

    try:
        from src.retrieval import load as load_index, retrieve
    except ModuleNotFoundError as e:
        raise SystemExit(f"Cannot import retrieval (missing dependency): {e}")

    print("Loading index for on-the-fly retrieval...")
    load_index()

    rows: list[dict] = []
    for i, q in enumerate(data, 1):
        if not isinstance(q, dict):
            continue
        question = (q.get("question") or "").strip()
        ref = (q.get("reference_answer") or "").strip()
        if not question or not ref:
            continue
        r = retrieve(
            question,
            top_k=int(top_k),
            use_hybrid=bool(use_hybrid),
            use_rerank=bool(use_rerank),
            rerank_model=rerank_model,
            rerank_candidate_k=rerank_candidate_k,
        )
        rows.append({
            "question": question,
            "language": q.get("language", "sl"),
            "answer": ref,
            "reference_answer": ref,
            "retrieved": r.get("chunks", []),
            "retrieval_weak": bool(r.get("retrieval_weak", False)),
            "is_negative": bool(q.get("is_negative", False)),
            "_judging_reference": True,
        })
        if i % 5 == 0 or i == len(data):
            print(f"  retrieved {i}/{len(data)} questions")
    return rows


def _load_references(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        cand = Path(__file__).resolve().parents[2] / path
        if cand.exists():
            p = cand
    if not p.exists():
        cand = config.BASE_DIR / path if hasattr(config, "BASE_DIR") else None
        if cand and cand.exists():
            p = cand
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out: dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            r = (item.get("reference_answer") or "").strip()
            if q and r:
                out[_norm(q)] = r
    return out


def _system_prompt() -> str:
    return (
        "You are an impartial evaluator for a Slovenian/English Retrieval-Augmented "
        "Generation assistant for UL FRI students. You receive a question, the chunks "
        "retrieved from the corpus, and the model's answer. Score the answer using a "
        "RAGAS-style rubric and check for hallucinations. "
        "Output strictly valid JSON. Do not include any prose outside the JSON."
    )


def _user_prompt(question: str, chunks_text: str, answer: str, reference: str | None) -> str:
    ref_block = ""
    if reference:
        ref_block = (
            f"REFERENCE_ANSWER (human-curated; helpful but not absolute):\n{reference}\n\n"
        )
    rubric = (
        '{\n'
        '  "faithfulness": <float in [0,1]>,\n'
        '  "answer_relevance": <float in [0,1]>,\n'
        '  "context_relevance": <float in [0,1]>,\n'
        '  "hallucination": <true|false>,\n'
        '  "hallucinated_spans": [<short quoted phrases from answer>],\n'
        '  "refusal": <true|false>,\n'
        '  "overall_score": <integer 1..5>,\n'
        '  "reasoning": <1-3 sentences>\n'
        '}'
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_text}\n\n"
        f"MODEL ANSWER:\n{answer}\n\n"
        f"{ref_block}"
        f"Score the answer using this exact JSON shape:\n{rubric}\n\n"
        "Rules:\n"
        "- 'faithfulness' is HIGH only if every claim in the answer is supported by the chunks.\n"
        "- If chunks did not contain the answer and the model correctly said so, set 'refusal'=true and 'faithfulness' high.\n"
        "- If the model invented administrative facts (deadlines, names, contacts) not in the chunks, set 'hallucination'=true.\n"
        "- 'overall_score': 5=excellent, 4=good with minor issues, 3=partially correct, 2=mostly wrong/unsupported, 1=hallucinated or wrongly refused."
    )


def _load_chunk_text_map(chunks_file: Path | None) -> dict[str, str]:
    if not chunks_file or not chunks_file.exists():
        return {}
    out: dict[str, str] = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = d.get("chunk_id")
            text = d.get("text")
            if isinstance(cid, str) and isinstance(text, str):
                out[cid] = text
    return out


def _format_chunks(
    retrieved: list[dict],
    *,
    top_k: int,
    max_chars: int,
    text_map: dict[str, str],
) -> str:
    parts: list[str] = []
    for i, c in enumerate(retrieved[: int(top_k)], 1):
        title = c.get("title", "") or ""
        section = c.get("section", "") or ""
        url = c.get("url", "") or ""
        text = (c.get("text") or c.get("preview") or "").strip()
        if not text:
            cid = c.get("chunk_id")
            if isinstance(cid, str):
                text = (text_map.get(cid) or "").strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        header = f"[{i}] {title}".strip()
        if section and section not in ("main", ""):
            header += f" — {section}"
        if url:
            header += f"  ({url})"
        parts.append(header + "\n" + text)
    return "\n\n".join(parts) if parts else "(no chunks)"


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
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


def _call_judge(client, *, model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
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


def _aggregate(judged: list[dict]) -> dict:
    n = len(judged)
    if n == 0:
        return {"n": 0}

    def _avg(key: str) -> float | None:
        nums = [float(r[key]) for r in judged if isinstance(r.get(key), (int, float))]
        return (sum(nums) / len(nums)) if nums else None

    return {
        "n": n,
        "faithfulness_mean": _avg("faithfulness"),
        "answer_relevance_mean": _avg("answer_relevance"),
        "context_relevance_mean": _avg("context_relevance"),
        "overall_score_mean": _avg("overall_score"),
        "hallucination_rate": sum(1 for r in judged if r.get("hallucination") is True) / n,
        "refusal_rate": sum(1 for r in judged if r.get("refusal") is True) / n,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.apply_run(args.run)

    if args.from_questions:
        src_path = _resolve_questions_path(args.from_questions)
        if not src_path.exists():
            print(f"Questions file not found: {src_path}", file=sys.stderr)
            raise SystemExit(1)
        rows = _rows_from_questions_file(
            src_path,
            top_k=int(args.retrieval_top_k),
            use_hybrid=bool(args.hybrid),
            use_rerank=bool(args.rerank),
            rerank_model=args.rerank_model,
            rerank_candidate_k=args.rerank_candidate_k,
        )
        res_path = src_path
    else:
        res_path = _resolve_results_path(args.results)
        if not res_path.exists():
            print(f"Results file not found: {res_path}", file=sys.stderr)
            raise SystemExit(1)
        rows = read_jsonl(res_path)

    if args.limit:
        rows = rows[: int(args.limit)]
    print(f"Loaded {len(rows)} rows from {res_path}")

    references = _load_references(args.reference_file) if args.reference_file else {}
    if references:
        print(f"Loaded {len(references)} reference answers")

    chunks_file = Path(args.chunks_file) if args.chunks_file else config.CHUNKS_JSONL
    text_map = _load_chunk_text_map(chunks_file)
    if text_map:
        print(f"Loaded {len(text_map)} chunk texts from {chunks_file}")
    else:
        print(f"WARNING: no chunk texts available (looked at {chunks_file}); judge will only see metadata.")

    model = args.model or config.JUDGE_MODEL
    client = _make_client()

    judged: list[dict] = []
    failed = 0

    for i, row in enumerate(rows, 1):
        question = row.get("question", "") or ""
        answer = row.get("answer", "") or ""
        retrieved = row.get("retrieved", []) or []
        if not question:
            continue
        if not answer:
            print(f"[{i}/{len(rows)}] no 'answer' field; skipping (retrieval-only run?)")
            continue

        chunks_text = _format_chunks(
            retrieved,
            top_k=int(args.top_k_context),
            max_chars=int(args.max_chunk_chars),
            text_map=text_map,
        )
        ref = references.get(_norm(question))

        try:
            raw = _call_judge(
                client,
                model=model,
                system=_system_prompt(),
                user=_user_prompt(question, chunks_text, answer, ref),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
            )
            data = _extract_json(raw)
        except Exception as e:
            failed += 1
            log.warning("[%d/%d] judge failed: %s", i, len(rows), e)
            judged.append({"question": question, "answer": answer, "judge_error": str(e)})
            if args.sleep:
                time.sleep(float(args.sleep))
            continue

        retrieved_top = []
        for c in retrieved[: int(args.top_k_context)]:
            c_out = dict(c)
            if not c_out.get("text"):
                cid = c_out.get("chunk_id")
                if isinstance(cid, str):
                    txt = text_map.get(cid)
                    if txt:
                        c_out["text"] = txt
            retrieved_top.append(c_out)

        record = {
            "question": question,
            "language": row.get("language"),
            "answer": answer,
            "reference_answer": ref or row.get("reference_answer"),
            "judging_reference": bool(row.get("_judging_reference", False)),
            "faithfulness": data.get("faithfulness"),
            "answer_relevance": data.get("answer_relevance"),
            "context_relevance": data.get("context_relevance"),
            "hallucination": bool(data.get("hallucination", False)),
            "hallucinated_spans": data.get("hallucinated_spans") or [],
            "refusal": bool(data.get("refusal", False)),
            "overall_score": data.get("overall_score"),
            "reasoning": data.get("reasoning"),
            "is_negative": bool(row.get("is_negative", False)),
            "retrieval_weak": bool(row.get("retrieval_weak", False)),
            "retrieved": retrieved_top,
        }
        judged.append(record)
        score = "?" if record["overall_score"] is None else str(record["overall_score"])
        faith = record["faithfulness"]
        faith_str = "?" if faith is None else f"{float(faith):.2f}"
        print(f"[{i}/{len(rows)}] score={score} hall={int(record['hallucination'])} faith={faith_str}")

        if args.sleep:
            time.sleep(float(args.sleep))

    if args.out:
        out_name = args.out
    elif args.from_questions:
        out_name = f"judged_reference_{res_path.stem}.jsonl"
    else:
        out_name = f"judged_{res_path.stem}.jsonl"
    out_path = config.EVAL_DIR / out_name
    write_jsonl(judged, out_path)

    summary = _aggregate([j for j in judged if "judge_error" not in j])
    print()
    print("=" * 50)
    print(f"Judge model: {model}")
    print(f"Source results: {res_path}")
    print(f"Judged rows: {summary['n']} (failed: {failed})")
    if summary["n"]:
        for k in ("faithfulness_mean", "answer_relevance_mean", "context_relevance_mean", "overall_score_mean"):
            v = summary.get(k)
            if v is not None:
                print(f"{k}: {float(v):.3f}")
        print(f"hallucination_rate: {summary['hallucination_rate']:.2%}")
        print(f"refusal_rate: {summary['refusal_rate']:.2%}")
    print("=" * 50)
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()

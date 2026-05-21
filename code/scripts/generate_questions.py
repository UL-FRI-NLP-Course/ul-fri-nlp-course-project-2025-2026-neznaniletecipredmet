"""Generate eval questions from corpus chunks using Claude.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/generate_questions.py --run <run> --num-chunks 80 --questions-per-chunk 2 --negatives 20

Reads <run>/processed/chunks.jsonl, samples chunks stratified by source domain,
asks Claude for student-style questions, writes <run>/eval/questions_generated.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate eval questions from corpus chunks via Claude")
    p.add_argument("--run", default=None)
    p.add_argument("--num-chunks", type=int, default=80)
    p.add_argument("--questions-per-chunk", type=int, default=2)
    p.add_argument("--negatives", type=int, default=20)
    p.add_argument("--out-name", default="questions_generated.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=None, help=f"Claude model (default: {config.EVAL_QUESTION_GEN_MODEL})")
    p.add_argument("--max-output-tokens", type=int, default=1500)
    p.add_argument("--min-chunk-chars", type=int, default=400)
    p.add_argument("--limit-chunks-per-domain", type=int, default=None)
    p.add_argument("--language-mix", default="sl:0.7,en:0.3")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _domain_of(url: str, source_path: str) -> str:
    if url:
        try:
            host = urlparse(url).netloc.lower()
            if host:
                return host
        except Exception:
            pass
    if source_path:
        return Path(source_path).suffix.lstrip(".") or "local"
    return "unknown"


def _stratified_sample(
    chunks: list[dict],
    n: int,
    *,
    seed: int,
    min_chars: int,
    per_domain_cap: int | None,
) -> list[dict]:
    rng = random.Random(seed)
    eligible = [c for c in chunks if isinstance(c.get("text"), str) and len(c["text"]) >= min_chars]
    if not eligible:
        return []

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for c in eligible:
        d = _domain_of(c.get("url", ""), c.get("source_path", ""))
        by_domain[d].append(c)

    domains = sorted(by_domain.keys())
    rng.shuffle(domains)
    queues = {d: list(by_domain[d]) for d in domains}
    for d in queues:
        rng.shuffle(queues[d])
    used = {d: 0 for d in domains}

    out: list[dict] = []
    seen: set[str] = set()
    while len(out) < n and any(queues[d] for d in domains):
        progressed = False
        for d in domains:
            if not queues[d]:
                continue
            if per_domain_cap is not None and used[d] >= per_domain_cap:
                continue
            c = queues[d].pop()
            cid = c.get("chunk_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            out.append(c)
            used[d] += 1
            progressed = True
            if len(out) >= n:
                break
        if not progressed:
            break
    return out


def _parse_language_mix(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        lang, prob = part.split(":", 1)
        try:
            out[lang.strip().lower()] = float(prob)
        except ValueError:
            continue
    if not out:
        out = {"sl": 0.7, "en": 0.3}
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def _pick_lang(mix: dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    last = "sl"
    for lang, p in mix.items():
        last = lang
        cum += p
        if r <= cum:
            return lang
    return last


def _system_prompt() -> str:
    return (
        "You are designing an evaluation set for a Slovenian/English RAG assistant for "
        "students at the Faculty of Computer Science (UL FRI). You will be given a "
        "passage from an official faculty document. Your job is to write realistic "
        "questions that a student might actually type into a chatbot, where the answer "
        "is contained (at least partially) in the passage. "
        "Output strictly valid JSON. Do not include any prose outside the JSON."
    )


def _user_prompt_for_chunk(chunk: dict, *, n: int, language: str) -> str:
    title = chunk.get("title") or ""
    section = chunk.get("section") or ""
    url = chunk.get("url") or ""
    text = (chunk.get("text") or "")[:3000]
    lang_label = "Slovenian" if language == "sl" else "English"
    return (
        f"PASSAGE TITLE: {title}\n"
        f"SECTION: {section}\n"
        f"SOURCE URL: {url}\n"
        f"PASSAGE:\n{text}\n\n"
        f"Generate exactly {n} {lang_label} questions a student might ask, where the "
        f"answer is contained in the passage above. Avoid trivia. Prefer questions "
        f"about deadlines, requirements, procedures, contacts, locations, eligibility. "
        f"For each question, provide 2-5 expected_keywords (short word stems in the "
        f"same language as the question). Return JSON in this exact shape:\n\n"
        '{\n'
        '  "questions": [\n'
        '    {"question": "...", "language": "' + language + '", "expected_keywords": ["...", "..."]}\n'
        '  ]\n'
        '}'
    )


def _negatives_prompt(n: int) -> str:
    return (
        f"Generate exactly {n} questions that are clearly OUT OF SCOPE for an "
        f"administrative student assistant of the Faculty of Computer Science (UL FRI) "
        f"in Ljubljana. Mix Slovenian and English (about half/half). Out-of-scope examples: "
        f"cooking, sports results, weather, finance, general trivia, jokes, asking the "
        f"bot to write code unrelated to academic administration. Return JSON in this "
        f"exact shape:\n\n"
        '{\n'
        '  "questions": [\n'
        '    {"question": "...", "language": "sl"},\n'
        '    {"question": "...", "language": "en"}\n'
        '  ]\n'
        '}'
    )


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
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
        print("ANTHROPIC_API_KEY not set. Add it to your environment or to a .env file.", file=sys.stderr)
        raise SystemExit(2)
    return anthropic.Anthropic()


def _call_claude(client, *, model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config.apply_run(args.run)

    chunks_path = config.CHUNKS_JSONL
    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}", file=sys.stderr)
        raise SystemExit(1)

    all_chunks = read_jsonl(chunks_path)
    print(f"Loaded {len(all_chunks)} chunks from {chunks_path}")

    sampled = _stratified_sample(
        all_chunks,
        n=int(args.num_chunks),
        seed=int(args.seed),
        min_chars=int(args.min_chunk_chars),
        per_domain_cap=args.limit_chunks_per_domain,
    )
    print(f"Sampled {len(sampled)} chunks")

    if args.dry_run:
        for c in sampled[:5]:
            d = _domain_of(c.get("url", ""), c.get("source_path", ""))
            print(f"  {c.get('chunk_id')} ({d}) [{len(c.get('text', ''))} chars]")
        print("(--dry-run; not calling API)")
        return

    model = args.model or config.EVAL_QUESTION_GEN_MODEL
    client = _make_client()
    rng = random.Random(int(args.seed))
    lang_mix = _parse_language_mix(args.language_mix)

    out_records: list[dict] = []
    for i, c in enumerate(sampled, 1):
        lang = _pick_lang(lang_mix, rng)
        try:
            raw = _call_claude(
                client,
                model=model,
                system=_system_prompt(),
                user=_user_prompt_for_chunk(c, n=int(args.questions_per_chunk), language=lang),
                max_tokens=int(args.max_output_tokens),
                temperature=float(args.temperature),
            )
            data = _extract_json(raw)
        except Exception as e:
            log.warning("[%d/%d] chunk %s failed: %s", i, len(sampled), c.get("chunk_id"), e)
            continue

        for q in (data.get("questions") or []):
            q_text = (q.get("question") or "").strip()
            if not q_text:
                continue
            kws = q.get("expected_keywords") or []
            if not isinstance(kws, list):
                kws = []
            kws = [str(k).strip() for k in kws if str(k).strip()]
            out_records.append({
                "question": q_text,
                "language": q.get("language") or lang,
                "expected_keywords": kws,
                "relevant_doc_ids": [],
                "relevant_chunk_ids": [],
                "source_chunk_id": c.get("chunk_id"),
                "source_doc_id": c.get("doc_id"),
                "source_url": c.get("url"),
                "auto_generated": True,
                "is_negative": False,
            })

        if i % 5 == 0 or i == len(sampled):
            print(f"  generated for {i}/{len(sampled)} chunks (records so far: {len(out_records)})")

        if args.sleep:
            time.sleep(float(args.sleep))

    if args.negatives > 0:
        try:
            raw = _call_claude(
                client,
                model=model,
                system=_system_prompt(),
                user=_negatives_prompt(int(args.negatives)),
                max_tokens=int(args.max_output_tokens),
                temperature=float(args.temperature),
            )
            data = _extract_json(raw)
            n_added = 0
            for q in (data.get("questions") or []):
                q_text = (q.get("question") or "").strip()
                if not q_text:
                    continue
                out_records.append({
                    "question": q_text,
                    "language": q.get("language") or "sl",
                    "expected_keywords": [],
                    "relevant_doc_ids": [],
                    "relevant_chunk_ids": [],
                    "auto_generated": True,
                    "is_negative": True,
                    "non_in_corpus": True,
                })
                n_added += 1
            print(f"Generated {n_added} negative questions")
        except Exception as e:
            log.warning("Negative generation failed: %s", e)

    out_path = config.EVAL_DIR / args.out_name
    write_jsonl(out_records, out_path)
    print(f"Wrote {len(out_records)} questions -> {out_path}")
    print(f"Next: python scripts/merge_eval_questions.py --run {args.run or 'default'} --src {args.out_name}")


if __name__ == "__main__":
    main()

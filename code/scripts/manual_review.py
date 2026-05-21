"""Manual review of judged answers; computes agreement with the Claude judge.

Usage:
  python scripts/manual_review.py --run <run> --judged judged_results_<model>.jsonl --n 25 --stratify
  python scripts/manual_review.py --run <run> --judged judged_results_<model>.jsonl --report-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual review of judged answers + agreement metrics")
    p.add_argument("--run", default=None)
    p.add_argument("--judged", required=True, help="Path or filename in <run>/eval/")
    p.add_argument("--out", default=None)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--full-text", action="store_true")
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--report-only", action="store_true")
    return p.parse_args()


def _resolve(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    return config.EVAL_DIR / name


def _stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    buckets: dict[int, list[dict]] = {}
    for r in rows:
        s = r.get("overall_score")
        if isinstance(s, (int, float)):
            buckets.setdefault(int(s), []).append(r)
    if not buckets:
        rng.shuffle(rows)
        return rows[:n]
    per = max(1, n // len(buckets))
    out: list[dict] = []
    for k in sorted(buckets.keys()):
        rng.shuffle(buckets[k])
        out.extend(buckets[k][:per])
    rng.shuffle(out)
    return out[:n]


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        s = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return s or (default or "")


def _agreement(rows: list[dict]) -> dict:
    paired = [
        r for r in rows
        if r.get("manual_overall") is not None and r.get("overall_score") is not None
    ]
    n = len(paired)
    out: dict = {"n": n}
    if n:
        diffs = [abs(int(r["manual_overall"]) - int(r["overall_score"])) for r in paired]
        out["score_exact_match"] = sum(1 for d in diffs if d == 0) / n
        out["score_within_1"] = sum(1 for d in diffs if d <= 1) / n
        out["score_mae"] = sum(diffs) / n

    hall_pairs = [
        (bool(r.get("manual_hallucination", False)), bool(r.get("hallucination", False)))
        for r in rows
        if r.get("manual_hallucination") is not None
    ]
    if hall_pairs:
        out["hallucination_match_rate"] = sum(1 for m, j in hall_pairs if m == j) / len(hall_pairs)
        out["hallucination_n"] = len(hall_pairs)
    return out


def main() -> None:
    args = parse_args()
    config.apply_run(args.run)

    judged_path = _resolve(args.judged)
    if not judged_path.exists():
        print(f"Not found: {judged_path}", file=sys.stderr)
        raise SystemExit(1)

    out_name = args.out or f"manual_review_{judged_path.stem}.jsonl"
    out_path = config.EVAL_DIR / out_name

    if args.report_only:
        if not out_path.exists():
            print(f"No manual review file at {out_path}", file=sys.stderr)
            raise SystemExit(1)
        rows = read_jsonl(out_path)
        print(json.dumps(_agreement(rows), indent=2, ensure_ascii=False))
        return

    rows = read_jsonl(judged_path)
    rows = [r for r in rows if "judge_error" not in r]
    print(f"Loaded {len(rows)} judged rows")

    rng = random.Random(int(args.seed))
    if args.stratify:
        sample = _stratified(rows, int(args.n), rng)
    else:
        sample = rng.sample(rows, min(int(args.n), len(rows)))

    out_records: list[dict] = []
    if out_path.exists():
        out_records = read_jsonl(out_path)
        already = {r.get("question") for r in out_records}
        skipped = sum(1 for s in sample if s.get("question") in already)
        sample = [s for s in sample if s.get("question") not in already]
        print(f"Skipping {skipped} already-reviewed rows; reviewing {len(sample)} new")

    for i, r in enumerate(sample, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(sample)}] Q: {r.get('question')}")
        print(f"LANG: {r.get('language')}")
        print()
        print("RETRIEVED:")
        for j, c in enumerate(r.get("retrieved", []), 1):
            t = c.get("text", "") or c.get("preview", "") or ""
            if not args.full_text and len(t) > args.max_chars:
                t = t[: args.max_chars] + "..."
            print(f"  ({j}) {c.get('title', '')} — {c.get('section', '')}")
            print(textwrap.indent(t, "      "))
        print()
        print("ANSWER:")
        print(textwrap.indent((r.get("answer") or "").strip(), "  "))
        print()
        if r.get("reference_answer"):
            print("REFERENCE:")
            print(textwrap.indent(r["reference_answer"], "  "))
            print()
        print("JUDGE (Claude):")
        for k in ("faithfulness", "answer_relevance", "context_relevance", "overall_score", "hallucination", "refusal"):
            print(f"  {k}: {r.get(k)}")
        if r.get("reasoning"):
            print(f"  reasoning: {r.get('reasoning')}")

        print("\nYour rating (blank skips this row):")
        score = _ask("  overall (1-5)")
        if not score:
            continue
        try:
            score_i = int(score)
        except ValueError:
            continue
        hall = _ask("  hallucination? (y/n)", default="n").lower().startswith("y")
        notes = _ask("  notes (optional)")

        out = dict(r)
        out["manual_overall"] = score_i
        out["manual_hallucination"] = hall
        out["manual_notes"] = notes
        out_records.append(out)

        write_jsonl(out_records, out_path)

    print()
    print(f"Wrote {len(out_records)} -> {out_path}")
    print()
    print("Agreement vs Claude judge:")
    print(json.dumps(_agreement(out_records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

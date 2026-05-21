"""Merge auto-generated questions into <run>/eval/questions.jsonl with dedup.

Usage:
  python scripts/merge_eval_questions.py --run <run> --src questions_generated.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge generated questions into the eval set with dedup")
    p.add_argument("--run", default=None)
    p.add_argument("--src", default="questions_generated.jsonl")
    p.add_argument("--dst", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main() -> None:
    args = parse_args()
    config.apply_run(args.run)

    src_path = config.EVAL_DIR / args.src
    dst_path = Path(args.dst) if args.dst else config.EVAL_QUESTIONS_FILE
    if not src_path.exists():
        print(f"Source not found: {src_path}", file=sys.stderr)
        raise SystemExit(1)

    src = read_jsonl(src_path)
    existing = read_jsonl(dst_path) if dst_path.exists() else []
    seen = {_norm(q.get("question", "")) for q in existing}

    added: list[dict] = []
    skipped = 0
    for q in src:
        key = _norm(q.get("question", ""))
        if not key:
            skipped += 1
            continue
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        q.setdefault("expected_keywords", [])
        q.setdefault("relevant_doc_ids", [])
        q.setdefault("relevant_chunk_ids", [])
        added.append(q)

    print(f"Source: {len(src)}  Existing: {len(existing)}  New: {len(added)}  Skipped/dup: {skipped}")

    if args.dry_run:
        return

    merged = list(existing) + added
    write_jsonl(merged, dst_path)
    print(f"Wrote {len(merged)} -> {dst_path}")
    print()
    print("Suggested next steps:")
    print(
        f"  python scripts/bootstrap_eval_questions.py --run {args.run or 'default'} "
        "--hybrid --rerank --rerank-candidate-k 20 --no-append-questions"
    )
    print(
        f"  python scripts/annotate_eval.py --run {args.run or 'default'} --top-k 8 --hybrid --only-missing"
    )


if __name__ == "__main__":
    main()

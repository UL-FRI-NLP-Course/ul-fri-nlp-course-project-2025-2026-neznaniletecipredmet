"""Convert a questions_full_v2.json (JSON array) to <run>/eval/questions.jsonl

The cluster's scripts/evaluate.py reads questions.jsonl (one JSON object per line)
with fields: question, language, expected_keywords, is_negative, optional
relevant_doc_ids, relevant_chunk_ids. Our cleaned eval set lives in
questions_full_v2.json as a JSON array with reference_answer + audit metadata.

This script:
- reads the v2 JSON array
- writes one line per question into <run>/eval/questions.jsonl
- keeps reference_answer in the JSONL (extra field, ignored by evaluate.py
  but useful for downstream scripts)
- preserves audit_status / needs_review for traceability

Usage:
  python scripts/prepare_cluster_questions.py \
      --questions questions_full_v2.json \
      --run 2026-05-20__full__fri4_ul2__v1__e5_base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert questions_full_v2.json -> <run>/eval/questions.jsonl")
    p.add_argument("--questions", required=True, help="Path to JSON array (e.g. questions_full_v2.json)")
    p.add_argument("--run", required=True, help="Run name (writes to <run>/eval/questions.jsonl)")
    p.add_argument("--out", default=None, help="Override output path (default: <run>/eval/questions.jsonl)")
    p.add_argument(
        "--keep-fields",
        default="question,language,expected_keywords,is_negative,reference_answer,audit_status,needs_review,non_in_corpus,relevant_doc_ids,relevant_chunk_ids",
        help="Comma-separated list of fields to keep per row",
    )
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


def main() -> None:
    args = parse_args()
    config.apply_run(args.run)

    src_path = _resolve_questions_path(args.questions)
    if not src_path.exists():
        print(f"Questions file not found: {src_path}", file=sys.stderr)
        raise SystemExit(1)

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Expected JSON array in {src_path}, got {type(data).__name__}", file=sys.stderr)
        raise SystemExit(1)

    keep = [s.strip() for s in args.keep_fields.split(",") if s.strip()]

    out_path = Path(args.out) if args.out else (config.EVAL_DIR / "questions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in data:
            if not isinstance(row, dict):
                continue
            obj = {k: row.get(k) for k in keep if k in row}
            if not obj.get("question"):
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows -> {out_path}")
    print()
    print("Cluster expects this file at the same path inside the same run folder.")
    print(f"  {out_path.relative_to(config.BASE_DIR.parent)}")


if __name__ == "__main__":
    main()

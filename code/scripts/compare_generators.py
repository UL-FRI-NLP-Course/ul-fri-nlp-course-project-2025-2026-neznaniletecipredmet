"""Compare generator models on the same RAG retrieval setup.

Workflow:
  1. cluster runs scripts/evaluate.py for each generator model and writes
     <run>/eval/results_<model_slug>.jsonl
  2. you rsync those files back to the local Mac
  3. this script:
       a. for each results_*.jsonl, runs scripts/judge_answers.py --results
          with --reference-file pointing at questions_full_v2.json so the
          Claude judge can grade against the cleaned references
       b. parses the resulting judged_*.jsonl files
       c. emits <run>/eval/model_comparison.md with one row per model

The judge sweep uses Haiku by default (consistent ranking, low cost). Pass
--judge-model claude-sonnet-4-6 to use Sonnet for the headline grade on the
single winning model.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/compare_generators.py \
      --run 2026-05-20__full__fri4_ul2__v1__e5_base \
      --reference-file ../code/questions_full_v2.json \
      --judge-model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
JUDGE = SCRIPTS_DIR / "judge_answers.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge & compare cluster-generated model outputs")
    p.add_argument("--run", required=True, help="Run name whose eval/ dir contains results_*.jsonl")
    p.add_argument("--reference-file", required=True,
                   help="Path to JSON array with reference_answer per question (e.g. questions_full_v2.json)")
    p.add_argument("--judge-model", default="claude-haiku-4-5",
                   help="Claude model to use for judging (default: claude-haiku-4-5)")
    p.add_argument("--top-k-context", type=int, default=4,
                   help="How many retrieved chunks to show the judge per row (default: 4)")
    p.add_argument("--max-chunk-chars", type=int, default=900,
                   help="Truncate each chunk to N chars in the judge prompt (default: 900)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--include-pattern", default="results_*.jsonl",
                   help="Glob inside <run>/eval/ to pick model output files (default: results_*.jsonl)")
    p.add_argument("--exclude-pattern", default="results_retrieval_only.jsonl",
                   help="Glob to exclude (default: results_retrieval_only.jsonl)")
    p.add_argument("--skip-existing-judge", action="store_true",
                   help="If a judged_<results>.jsonl already exists, skip rejudging that model")
    p.add_argument("--out-md", default=None,
                   help="Where to write the comparison markdown (default: <run>/eval/model_comparison.md)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of rows judged per model (smoke test)")
    return p.parse_args()


def _resolve_reference_file(name_or_path: str) -> Path:
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


def _model_label_from_filename(p: Path) -> str:
    name = p.name
    if name.startswith("results_"):
        name = name[len("results_"):]
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return name


def _judge_one(results_path: Path, *, run_name: str, judge_model: str,
               reference_file: Path, top_k_context: int, max_chunk_chars: int,
               temperature: float, limit: int | None,
               skip_existing: bool) -> Path | None:
    eval_dir = results_path.parent
    out_name = f"judged_{results_path.name}"
    out_path = eval_dir / out_name
    if skip_existing and out_path.exists():
        log.info("[skip-judge] %s exists", out_path)
        return out_path

    cmd = [
        sys.executable, str(JUDGE),
        "--run", run_name,
        "--results", results_path.name,
        "--reference-file", str(reference_file),
        "--model", judge_model,
        "--top-k-context", str(top_k_context),
        "--max-chunk-chars", str(max_chunk_chars),
        "--temperature", str(temperature),
        "--out", out_name,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    log.info("Judging %s: %s", _model_label_from_filename(results_path), " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        log.error("judge_answers.py failed (exit %d) for %s", res.returncode, results_path.name)
        return None
    log.info("  judge wall time: %.1fs", elapsed)
    return out_path


def _aggregate_judged(path: Path) -> dict:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    refusals = [r for r in rows if r.get("refusal")]
    in_scope = [r for r in rows if not r.get("refusal")]

    def _avg(rs: list[dict], key: str) -> float | None:
        nums = [float(r[key]) for r in rs if isinstance(r.get(key), (int, float))]
        return statistics.mean(nums) if nums else None

    def _rate(rs: list[dict], key: str) -> float | None:
        if not rs:
            return None
        return sum(1 for r in rs if r.get(key) is True) / len(rs)

    return {
        "n": len(rows),
        "n_in_scope": len(in_scope),
        "n_refusals": len(refusals),
        "in_scope_faithfulness": _avg(in_scope, "faithfulness"),
        "in_scope_answer_relevance": _avg(in_scope, "answer_relevance"),
        "in_scope_context_relevance": _avg(in_scope, "context_relevance"),
        "in_scope_overall": _avg(in_scope, "overall_score"),
        "in_scope_hallucination_rate": _rate(in_scope, "hallucination"),
        "refusal_overall": _avg(refusals, "overall_score"),
        "refusal_faithfulness": _avg(refusals, "faithfulness"),
    }


def _aggregate_results_meta(path: Path) -> dict:
    """Pull per-model metadata from the original results_*.jsonl (gen time, kw hit)."""
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return {}

    def _avg(key: str) -> float | None:
        nums = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return statistics.mean(nums) if nums else None

    return {
        "n_questions": len(rows),
        "avg_generation_time_s": _avg("generation_time_s"),
        "keyword_hit_rate": _avg("keyword_hit_rate"),
    }


def _fmt(x: object, decimals: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def _md_table(rows: list[tuple[str, dict, dict]]) -> str:
    cols = [
        ("n_in_scope",                  "In-scope n"),
        ("in_scope_faithfulness",       "Faithfulness"),
        ("in_scope_answer_relevance",   "Ans rel"),
        ("in_scope_context_relevance",  "Ctx rel"),
        ("in_scope_hallucination_rate", "Hallucination"),
        ("in_scope_overall",            "Overall (1-5)"),
        ("refusal_overall",             "Refusal (1-5)"),
    ]
    header = "| Model | " + " | ".join(c[1] for c in cols) + " | Avg gen time (s) | KW hit |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|---:|---:|"
    lines = [header, sep]
    for label, agg, meta in rows:
        cells = []
        for key, _ in cols:
            v = agg.get(key)
            if key == "in_scope_hallucination_rate" and isinstance(v, float):
                cells.append(f"{v*100:.1f}%")
            else:
                cells.append(_fmt(v))
        gen_time = meta.get("avg_generation_time_s")
        kw_hit = meta.get("keyword_hit_rate")
        kw_str = f"{kw_hit*100:.1f}%" if isinstance(kw_hit, float) else "—"
        lines.append(
            f"| `{label}` | " + " | ".join(cells) + f" | {_fmt(gen_time, 1)} | {kw_str} |"
        )
    return "\n".join(lines)


def _pick_winner(rows: list[tuple[str, dict, dict]]) -> tuple[str, dict, dict] | None:
    scored = [(label, agg, meta, agg.get("in_scope_faithfulness"))
              for label, agg, meta in rows
              if isinstance(agg.get("in_scope_faithfulness"), (int, float))]
    if not scored:
        return None
    scored.sort(key=lambda t: t[3], reverse=True)
    return scored[0][0], scored[0][1], scored[0][2]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.apply_run(args.run)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    eval_dir = config.EVAL_DIR
    if not eval_dir.exists():
        log.error("Eval dir does not exist: %s", eval_dir)
        raise SystemExit(1)

    results_files = sorted(eval_dir.glob(args.include_pattern))
    if args.exclude_pattern:
        excluded = set(eval_dir.glob(args.exclude_pattern))
        results_files = [p for p in results_files if p not in excluded]
    if not results_files:
        log.error("No %s files found in %s", args.include_pattern, eval_dir)
        raise SystemExit(1)

    reference_file = _resolve_reference_file(args.reference_file)
    if not reference_file.exists():
        log.error("Reference file not found: %s", reference_file)
        raise SystemExit(1)

    log.info("Run:            %s", args.run)
    log.info("Eval dir:       %s", eval_dir)
    log.info("Models found:   %d", len(results_files))
    for p in results_files:
        log.info("  - %s", p.name)
    log.info("Reference file: %s", reference_file)
    log.info("Judge model:    %s", args.judge_model)

    matrix_rows: list[tuple[str, dict, dict]] = []
    for results_path in results_files:
        label = _model_label_from_filename(results_path)
        log.info("\n=== %s ===", label)

        judged_path = _judge_one(
            results_path,
            run_name=args.run,
            judge_model=args.judge_model,
            reference_file=reference_file,
            top_k_context=int(args.top_k_context),
            max_chunk_chars=int(args.max_chunk_chars),
            temperature=float(args.temperature),
            limit=args.limit,
            skip_existing=args.skip_existing_judge,
        )
        if judged_path is None or not judged_path.exists():
            log.warning("Skipping aggregation for %s (no judged file)", label)
            continue

        agg = _aggregate_judged(judged_path)
        meta = _aggregate_results_meta(results_path)
        matrix_rows.append((label, agg, meta))

    if not matrix_rows:
        log.error("No matrix rows produced; nothing to write.")
        raise SystemExit(1)

    out_md = Path(args.out_md) if args.out_md else (eval_dir / "model_comparison.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Generator-model comparison\n\n"
        f"- Run:               `{args.run}`\n"
        f"- Reference file:    `{reference_file}`\n"
        f"- Judge model:       `{args.judge_model}` (temperature={args.temperature})\n"
        f"- Retrieval context: top-{args.top_k_context} chunks\n\n"
    )
    table = _md_table(matrix_rows)

    winner = _pick_winner(matrix_rows)
    winner_md = ""
    if winner:
        wlabel, wagg, _wmeta = winner
        winner_md = (
            f"\n\n## Best by in-scope faithfulness\n\n"
            f"- **Model**: `{wlabel}`\n"
            f"- **In-scope faithfulness**: {_fmt(wagg.get('in_scope_faithfulness'))}\n"
            f"- **In-scope hallucination**: "
            f"{(_fmt(wagg.get('in_scope_hallucination_rate')*100, 1) + '%') if isinstance(wagg.get('in_scope_hallucination_rate'), float) else '—'}\n"
            f"- **Overall (1-5)**: {_fmt(wagg.get('in_scope_overall'))}\n\n"
            f"For the report's headline number, re-judge this model with Sonnet:\n\n"
            f"```\n"
            f"python scripts/compare_generators.py \\\n"
            f"    --run {args.run} \\\n"
            f"    --reference-file {reference_file} \\\n"
            f"    --judge-model claude-sonnet-4-6 \\\n"
            f"    --include-pattern 'results_{wlabel}.jsonl' \\\n"
            f"    --out-md {eval_dir}/model_comparison_winner_sonnet.md\n"
            f"```\n"
        )

    out_md.write_text(header + table + winner_md + "\n", encoding="utf-8")
    print()
    print(table)
    if winner:
        print()
        print(f"Winner: {winner[0]} -> faithfulness={_fmt(winner[1].get('in_scope_faithfulness'))}")
    print(f"\nWrote -> {out_md}")


if __name__ == "__main__":
    main()

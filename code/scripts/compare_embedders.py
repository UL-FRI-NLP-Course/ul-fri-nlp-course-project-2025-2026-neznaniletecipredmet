"""Compare embedders on the same crawl across multiple retrieval modes.

For each embedding model, creates a derived run that symlinks raw/ from the
source run, sets EMBEDDING_MODEL, and rebuilds parsed/chunks/index. Then for
each retrieval mode, invokes judge_answers.py --from-questions to get a
faithfulness number, and aggregates everything into eval_matrix.md.

Designed to use Haiku as the judge for the comparison sweeps; the winning
configuration can then be re-judged with Sonnet for the report's headline
metric.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/compare_embedders.py \
      --source-run 2026-05-20__full__fri4_ul2__v1 \
      --questions ../code/questions_full_v2.json \
      --top-k 4 \
      --embedders intfloat/multilingual-e5-base,intfloat/multilingual-e5-large \
      --modes dense,hybrid,hybrid_rerank \
      --judge-model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BUILD_INDEX = SCRIPTS_DIR / "build_index.py"
JUDGE = SCRIPTS_DIR / "judge_answers.py"

DEFAULT_EMBEDDERS = [
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
]
DEFAULT_MODES = ["dense", "hybrid", "hybrid_rerank"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embedder x retrieval-mode sweep")
    p.add_argument("--source-run", required=True,
                   help="Existing run name whose raw/ data is reused (the crawl)")
    p.add_argument("--questions", required=True,
                   help="Path to JSON array (e.g. questions_full_v2.json)")
    p.add_argument("--top-k", type=int, default=4,
                   help="Retrieval top-k for the judge (default: 4)")
    p.add_argument("--embedders", default=",".join(DEFAULT_EMBEDDERS),
                   help=f"Comma-separated embedding model names (default: {','.join(DEFAULT_EMBEDDERS)})")
    p.add_argument("--modes", default=",".join(DEFAULT_MODES),
                   help=f"Comma-separated retrieval modes (default: {','.join(DEFAULT_MODES)})")
    p.add_argument("--judge-model", default="claude-haiku-4-5",
                   help="Judge model for the sweep (default: claude-haiku-4-5)")
    p.add_argument("--max-chunk-chars", type=int, default=900,
                   help="Truncate each chunk to N chars in the judge prompt")
    p.add_argument("--skip-existing-index", action="store_true",
                   help="If a derived run already has a built index, skip rebuilding it")
    p.add_argument("--skip-existing-judge", action="store_true",
                   help="If a judged_<mode>.jsonl already exists, skip rerunning judge")
    p.add_argument("--out-md", default=None,
                   help="Where to write the eval_matrix.md (default: <source-run>/eval/eval_matrix.md)")
    return p.parse_args()


def _short_tag(model: str) -> str:
    last = model.split("/")[-1]
    last = last.replace("multilingual-", "").replace("-", "_")
    return last


def _mode_flags(mode: str) -> list[str]:
    if mode == "dense":
        return []
    if mode == "hybrid":
        return ["--hybrid"]
    if mode == "dense_rerank":
        return ["--rerank"]
    if mode == "hybrid_rerank":
        return ["--hybrid", "--rerank"]
    raise ValueError(f"unknown mode: {mode}")


def _ensure_symlinked_raw(source_run_dir: Path, derived_run_dir: Path) -> None:
    derived_run_dir.mkdir(parents=True, exist_ok=True)
    src_raw = source_run_dir / "raw"
    dst_raw = derived_run_dir / "raw"
    if not src_raw.exists():
        raise FileNotFoundError(f"Source raw/ does not exist: {src_raw}")

    if dst_raw.exists():
        if dst_raw.is_symlink():
            current_target = os.readlink(dst_raw)
            if Path(current_target).resolve() == src_raw.resolve():
                return
            log.warning("Replacing existing raw/ symlink at %s", dst_raw)
            dst_raw.unlink()
        else:
            log.warning("Removing existing real directory at %s before symlinking", dst_raw)
            shutil.rmtree(dst_raw)
    dst_raw.symlink_to(src_raw.resolve(), target_is_directory=True)
    log.info("Symlinked %s -> %s", dst_raw, src_raw)


def _build_index(derived_run: str, embedder: str, *, skip_existing: bool) -> bool:
    derived_dir = config.RUNS_DIR / derived_run
    index_file = derived_dir / "index" / "index.faiss"
    if skip_existing and index_file.exists():
        log.info("[skip-build] %s exists", index_file)
        return True

    env = dict(os.environ)
    env["EMBEDDING_MODEL"] = embedder

    cmd = [sys.executable, str(BUILD_INDEX), "--run", derived_run, "--mode",
           "update" if index_file.exists() else "new"]
    log.info("Building index: %s", " ".join(cmd))
    log.info("  with EMBEDDING_MODEL=%s", embedder)
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        log.error("build_index.py failed (exit %d) for %s", res.returncode, derived_run)
        return False
    return True


def _judge(derived_run: str, embedder: str, mode: str,
           questions_path: Path, top_k: int, judge_model: str,
           max_chunk_chars: int, *, skip_existing: bool) -> Path | None:
    derived_dir = config.RUNS_DIR / derived_run
    out_name = f"judged_{mode}_k{top_k}.jsonl"
    out_path = derived_dir / "eval" / out_name
    if skip_existing and out_path.exists():
        log.info("[skip-judge] %s exists", out_path)
        return out_path

    env = dict(os.environ)
    env["EMBEDDING_MODEL"] = embedder

    cmd = [
        sys.executable, str(JUDGE),
        "--run", derived_run,
        "--from-questions", str(questions_path),
        "--retrieval-top-k", str(top_k),
        "--max-chunk-chars", str(max_chunk_chars),
        "--out", out_name,
        "--model", judge_model,
    ]
    cmd += _mode_flags(mode)

    log.info("Judging: %s", " ".join(cmd))
    log.info("  with EMBEDDING_MODEL=%s", embedder)
    t0 = time.time()
    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        log.error("judge_answers.py failed (exit %d) for %s/%s", res.returncode, derived_run, mode)
        return None
    log.info("  judge wall time: %.1fs", elapsed)
    return out_path


def _aggregate_judged(path: Path) -> dict:
    if not path.exists():
        return {}
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
        "refusal_faithfulness": _avg(refusals, "faithfulness"),
        "refusal_overall": _avg(refusals, "overall_score"),
    }


def _fmt(x: object, decimals: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def _md_table(rows: list[tuple[str, str, dict]]) -> str:
    cols = [
        ("n_in_scope", "In-scope n"),
        ("in_scope_faithfulness", "Faithfulness"),
        ("in_scope_hallucination_rate", "Hallucination"),
        ("in_scope_context_relevance", "Ctx rel"),
        ("in_scope_overall", "Overall"),
        ("refusal_faithfulness", "Refusal faith."),
    ]
    header = "| Embedder | Mode | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|---|" + "|".join("---:" for _ in cols) + "|"
    lines = [header, sep]
    for embedder, mode, agg in rows:
        cells = []
        for key, _ in cols:
            v = agg.get(key)
            if key == "in_scope_hallucination_rate" and isinstance(v, float):
                cells.append(f"{v*100:.1f}%")
            else:
                cells.append(_fmt(v))
        lines.append(f"| `{embedder}` | `{mode}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pick_winner(rows: list[tuple[str, str, dict]]) -> tuple[str, str, dict] | None:
    scored = [(e, m, a, a.get("in_scope_faithfulness")) for e, m, a in rows
              if isinstance(a.get("in_scope_faithfulness"), (int, float))]
    if not scored:
        return None
    scored.sort(key=lambda t: t[3], reverse=True)
    e, m, a, _ = scored[0]
    return e, m, a


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source_dir = config.RUNS_DIR / args.source_run
    if not source_dir.exists():
        log.error("Source run does not exist: %s", source_dir)
        raise SystemExit(1)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        for cand in (config.BASE_DIR / args.questions,
                     Path(__file__).resolve().parents[2] / args.questions):
            if cand.exists():
                questions_path = cand
                break
    if not questions_path.exists():
        log.error("Questions file not found: %s", args.questions)
        raise SystemExit(1)
    questions_path = questions_path.resolve()

    embedders = [s.strip() for s in args.embedders.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]

    log.info("Source run: %s", source_dir)
    log.info("Questions:  %s", questions_path)
    log.info("Embedders:  %s", embedders)
    log.info("Modes:      %s", modes)
    log.info("Top-k:      %d", args.top_k)
    log.info("Judge:      %s", args.judge_model)

    matrix_rows: list[tuple[str, str, dict]] = []

    for embedder in embedders:
        derived_run = f"{args.source_run}__{_short_tag(embedder)}"
        derived_dir = config.RUNS_DIR / derived_run
        log.info("\n=== Embedder: %s -> derived run: %s ===", embedder, derived_run)

        try:
            _ensure_symlinked_raw(source_dir, derived_dir)
        except Exception as e:
            log.error("symlink raw failed for %s: %s", derived_run, e)
            continue

        ok = _build_index(derived_run, embedder, skip_existing=args.skip_existing_index)
        if not ok:
            log.error("Skipping judge sweep for %s due to build failure", embedder)
            continue

        for mode in modes:
            log.info("--- Judging: embedder=%s mode=%s ---", embedder, mode)
            judged_path = _judge(
                derived_run, embedder, mode,
                questions_path=questions_path,
                top_k=int(args.top_k),
                judge_model=args.judge_model,
                max_chunk_chars=int(args.max_chunk_chars),
                skip_existing=args.skip_existing_judge,
            )
            if judged_path is None:
                continue
            agg = _aggregate_judged(judged_path)
            matrix_rows.append((embedder, mode, agg))

    if not matrix_rows:
        log.error("No matrix rows produced; nothing to report.")
        raise SystemExit(1)

    out_md = Path(args.out_md) if args.out_md else (source_dir / "eval" / "eval_matrix.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Embedder x retrieval-mode comparison\n\n"
        f"- Source crawl run: `{args.source_run}`\n"
        f"- Questions: `{questions_path}`\n"
        f"- Top-k: {args.top_k}\n"
        f"- Judge model: `{args.judge_model}` (temperature=0)\n"
        f"- Modes: {', '.join(modes)}\n\n"
    )
    table = _md_table(matrix_rows)

    winner = _pick_winner(matrix_rows)
    winner_md = ""
    if winner:
        we, wm, wa = winner
        winner_md = (
            f"\n\n## Best by in-scope faithfulness\n\n"
            f"- **Embedder**: `{we}`\n"
            f"- **Mode**: `{wm}`\n"
            f"- **In-scope faithfulness**: {_fmt(wa.get('in_scope_faithfulness'))}\n"
            f"- **In-scope hallucination**: {_fmt(wa.get('in_scope_hallucination_rate'))}\n"
            f"- **Derived run**: `{args.source_run}__{_short_tag(we)}`\n\n"
            f"For the report's headline number, re-judge this configuration with Sonnet:\n\n"
            f"```\n"
            f"EMBEDDING_MODEL={we} python scripts/judge_answers.py "
            f"--run {args.source_run}__{_short_tag(we)} "
            f"--from-questions {questions_path} "
            f"--retrieval-top-k {args.top_k} {' '.join(_mode_flags(wm))} "
            f"--model claude-sonnet-4-6 "
            f"--out judged_winner_sonnet.jsonl\n"
            f"```\n"
        )

    out_md.write_text(header + table + winner_md + "\n", encoding="utf-8")
    print()
    print(table)
    if winner:
        print()
        print(f"Winner: {winner[0]} / {winner[1]} -> "
              f"faithfulness={_fmt(winner[2].get('in_scope_faithfulness'))}")
    print(f"\nWrote -> {out_md}")


if __name__ == "__main__":
    main()

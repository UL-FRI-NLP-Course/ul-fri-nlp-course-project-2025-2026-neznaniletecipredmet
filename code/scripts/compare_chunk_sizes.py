"""Compare chunk sizes (and overlaps) on a fixed embedder + retrieval mode.

For each chunk_size in --chunk-sizes, creates a derived run that symlinks raw/
from the source run, sets CHUNK_SIZE (and CHUNK_OVERLAP) and EMBEDDING_MODEL,
rebuilds parsed/chunks/index, then runs judge_answers --from-questions and
aggregates everything into chunk_size_comparison.md.

Designed to be run AFTER scripts/compare_embedders.py picks a winner; pass that
embedder via --embedder and the winning retrieval mode via --mode.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/compare_chunk_sizes.py \
      --source-run 2026-05-20__full__fri4_ul2__v1 \
      --questions ../code/questions_full_v2.json \
      --top-k 4 \
      --embedder intfloat/multilingual-e5-large \
      --mode dense \
      --chunk-sizes 200,300,400 \
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk-size sweep on a fixed embedder + retrieval mode")
    p.add_argument("--source-run", required=True,
                   help="Existing run name whose raw/ data is reused (the crawl)")
    p.add_argument("--questions", required=True,
                   help="Path to JSON array (e.g. questions_full_v2.json)")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--embedder", default="intfloat/multilingual-e5-base",
                   help="Embedding model to hold constant across chunk sizes")
    p.add_argument("--mode", default="dense",
                   choices=["dense", "hybrid", "dense_rerank", "hybrid_rerank"],
                   help="Retrieval mode to hold constant")
    p.add_argument("--chunk-sizes", default="200,300,400",
                   help="Comma-separated chunk sizes to evaluate (default: 200,300,400)")
    p.add_argument("--chunk-overlap-frac", type=float, default=0.2,
                   help="Overlap as fraction of chunk size (default: 0.2 -> 80 for cs=400)")
    p.add_argument("--judge-model", default="claude-haiku-4-5")
    p.add_argument("--max-chunk-chars", type=int, default=900,
                   help="Truncate each chunk to N chars in the judge prompt")
    p.add_argument("--skip-existing-index", action="store_true")
    p.add_argument("--skip-existing-judge", action="store_true")
    p.add_argument("--out-md", default=None,
                   help="Path to chunk_size_comparison.md (default: <source-run>/eval/chunk_size_comparison.md)")
    return p.parse_args()


def _short_embedder_tag(model: str) -> str:
    last = model.split("/")[-1]
    return last.replace("multilingual-", "").replace("-", "_")


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
            dst_raw.unlink()
        else:
            shutil.rmtree(dst_raw)
    dst_raw.symlink_to(src_raw.resolve(), target_is_directory=True)


def _build_index(derived_run: str, embedder: str, chunk_size: int, chunk_overlap: int,
                 *, skip_existing: bool) -> bool:
    derived_dir = config.RUNS_DIR / derived_run
    index_file = derived_dir / "index" / "index.faiss"
    if skip_existing and index_file.exists():
        log.info("[skip-build] %s exists", index_file)
        return True

    env = dict(os.environ)
    env["EMBEDDING_MODEL"] = embedder
    env["CHUNK_SIZE"] = str(chunk_size)
    env["CHUNK_OVERLAP"] = str(chunk_overlap)

    cmd = [sys.executable, str(BUILD_INDEX), "--run", derived_run,
           "--mode", "update" if index_file.exists() else "new"]
    log.info("Building index: %s", " ".join(cmd))
    log.info("  env: EMBEDDING_MODEL=%s CHUNK_SIZE=%d CHUNK_OVERLAP=%d",
             embedder, chunk_size, chunk_overlap)
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        log.error("build_index.py failed for %s (exit %d)", derived_run, res.returncode)
        return False
    return True


def _judge(derived_run: str, embedder: str, chunk_size: int, chunk_overlap: int,
           mode: str, questions_path: Path, top_k: int, judge_model: str,
           max_chunk_chars: int, *, skip_existing: bool) -> Path | None:
    derived_dir = config.RUNS_DIR / derived_run
    out_name = f"judged_{mode}_k{top_k}_cs{chunk_size}.jsonl"
    out_path = derived_dir / "eval" / out_name
    if skip_existing and out_path.exists():
        log.info("[skip-judge] %s exists", out_path)
        return out_path

    env = dict(os.environ)
    env["EMBEDDING_MODEL"] = embedder
    env["CHUNK_SIZE"] = str(chunk_size)
    env["CHUNK_OVERLAP"] = str(chunk_overlap)

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
    t0 = time.time()
    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        log.error("judge_answers.py failed (exit %d)", res.returncode)
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
        "in_scope_faithfulness": _avg(in_scope, "faithfulness"),
        "in_scope_context_relevance": _avg(in_scope, "context_relevance"),
        "in_scope_overall": _avg(in_scope, "overall_score"),
        "in_scope_hallucination_rate": _rate(in_scope, "hallucination"),
    }


def _count_chunks(derived_run: str) -> int | None:
    chunks_path = config.RUNS_DIR / derived_run / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        return None
    n = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _fmt(x: object, decimals: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def _md_table(rows: list[tuple[int, int, int, dict]]) -> str:
    lines = [
        "| Chunk size | Overlap | Total chunks | In-scope n | Faithfulness | Hallucination | Ctx rel | Overall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cs, co, n_chunks, agg in rows:
        v = agg.get("in_scope_hallucination_rate")
        hall = f"{v*100:.1f}%" if isinstance(v, float) else "—"
        lines.append(
            f"| {cs} | {co} | "
            f"{_fmt(n_chunks)} | "
            f"{_fmt(agg.get('n_in_scope'))} | "
            f"{_fmt(agg.get('in_scope_faithfulness'))} | "
            f"{hall} | "
            f"{_fmt(agg.get('in_scope_context_relevance'))} | "
            f"{_fmt(agg.get('in_scope_overall'))} |"
        )
    return "\n".join(lines)


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

    chunk_sizes = [int(s.strip()) for s in args.chunk_sizes.split(",") if s.strip()]
    if not chunk_sizes:
        log.error("No chunk sizes provided")
        raise SystemExit(1)

    embedder = args.embedder
    mode = args.mode

    log.info("Source run: %s", source_dir)
    log.info("Questions:  %s", questions_path)
    log.info("Embedder:   %s", embedder)
    log.info("Mode:       %s", mode)
    log.info("Top-k:      %d", args.top_k)
    log.info("Judge:      %s", args.judge_model)
    log.info("Chunk sizes:%s", chunk_sizes)

    rows: list[tuple[int, int, int, dict]] = []

    for cs in chunk_sizes:
        co = max(1, int(round(cs * float(args.chunk_overlap_frac))))
        derived_run = f"{args.source_run}__{_short_embedder_tag(embedder)}__cs{cs}"
        derived_dir = config.RUNS_DIR / derived_run
        log.info("\n=== Chunk size %d (overlap %d) -> %s ===", cs, co, derived_run)

        try:
            _ensure_symlinked_raw(source_dir, derived_dir)
        except Exception as e:
            log.error("symlink raw failed: %s", e)
            continue

        ok = _build_index(derived_run, embedder, cs, co, skip_existing=args.skip_existing_index)
        if not ok:
            continue

        n_chunks = _count_chunks(derived_run) or 0

        judged_path = _judge(
            derived_run, embedder, cs, co, mode,
            questions_path=questions_path,
            top_k=int(args.top_k),
            judge_model=args.judge_model,
            max_chunk_chars=int(args.max_chunk_chars),
            skip_existing=args.skip_existing_judge,
        )
        if judged_path is None:
            continue
        agg = _aggregate_judged(judged_path)
        rows.append((cs, co, n_chunks, agg))

    if not rows:
        log.error("No chunk-size rows produced; nothing to report.")
        raise SystemExit(1)

    out_md = Path(args.out_md) if args.out_md else (source_dir / "eval" / "chunk_size_comparison.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Chunk-size comparison\n\n"
        f"- Source crawl run: `{args.source_run}`\n"
        f"- Embedder:         `{embedder}`\n"
        f"- Retrieval mode:   `{mode}` (top-k={args.top_k})\n"
        f"- Questions:        `{questions_path}`\n"
        f"- Judge model:      `{args.judge_model}` (temperature=0)\n"
        f"- Overlap fraction: {args.chunk_overlap_frac}\n\n"
    )

    table = _md_table(rows)
    out_md.write_text(header + table + "\n", encoding="utf-8")
    print()
    print(table)
    print(f"\nWrote -> {out_md}")


if __name__ == "__main__":
    main()

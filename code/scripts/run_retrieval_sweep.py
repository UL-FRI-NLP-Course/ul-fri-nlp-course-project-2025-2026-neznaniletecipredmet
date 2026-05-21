"""Run evaluate.py --retrieval-only across {dense, hybrid, dense+rerank, hybrid+rerank}
and aggregate into a Markdown comparison table.

Usage:
  python scripts/run_retrieval_sweep.py --run <run> --top-k 4 --rerank-candidate-k 20

Produces:
  <run>/eval/results_dense.jsonl
  <run>/eval/results_hybrid.jsonl
  <run>/eval/results_dense_rerank.jsonl
  <run>/eval/results_hybrid_rerank.jsonl
  <run>/eval/retrieval_comparison.md
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import read_jsonl


SCRIPT = Path(__file__).resolve().parent / "evaluate.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a 4-way retrieval evaluation sweep")
    p.add_argument("--run", default=None)
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--rerank-candidate-k", type=int, default=config.RERANK_CANDIDATE_K)
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of: dense,hybrid,dense_rerank,hybrid_rerank")
    return p.parse_args()


def _aggregate_one(rows: list[dict], top_k: int) -> dict:
    n = len(rows)
    out: dict = {"n": n}
    if not n:
        return out

    doc_anno = [r for r in rows if r.get("retrieval_hit") is not None]
    if doc_anno:
        out["recall_at_k"] = sum(1 for r in doc_anno if r.get("retrieval_hit")) / len(doc_anno)
        out["precision_at_k"] = sum(float(r.get("precision_at_k") or 0.0) for r in doc_anno) / len(doc_anno)
        out["mrr_at_k"] = sum(float(r.get("mrr_at_k") or 0.0) for r in doc_anno) / len(doc_anno)
        out["ndcg_at_k"] = sum(float(r.get("ndcg_at_k") or 0.0) for r in doc_anno) / len(doc_anno)
        out["doc_annotated_n"] = len(doc_anno)

    chunk_anno = [r for r in rows if r.get("chunk_hit") is not None]
    if chunk_anno:
        out["chunk_recall_at_k"] = sum(1 for r in chunk_anno if r.get("chunk_hit")) / len(chunk_anno)
        out["chunk_precision_at_k"] = sum(float(r.get("chunk_precision_at_k") or 0.0) for r in chunk_anno) / len(chunk_anno)
        out["chunk_mrr_at_k"] = sum(float(r.get("chunk_mrr_at_k") or 0.0) for r in chunk_anno) / len(chunk_anno)
        out["chunk_ndcg_at_k"] = sum(float(r.get("chunk_ndcg_at_k") or 0.0) for r in chunk_anno) / len(chunk_anno)
        out["chunk_annotated_n"] = len(chunk_anno)

    negs = [r for r in rows if r.get("is_negative")]
    if negs:
        out["negatives_n"] = len(negs)
        out["negatives_correct_weak_rate"] = sum(1 for r in negs if r.get("retrieval_weak")) / len(negs)

    times = [float(r["retrieval_time_s"]) for r in rows if r.get("retrieval_time_s") is not None]
    if times:
        out["avg_retrieval_time_s"] = sum(times) / len(times)

    out["top_k"] = top_k
    return out


def _md_table(per_config: dict[str, dict]) -> str:
    cols = [
        ("recall_at_k", "Recall@k"),
        ("precision_at_k", "Precision@k"),
        ("mrr_at_k", "MRR@k"),
        ("ndcg_at_k", "nDCG@k"),
        ("chunk_recall_at_k", "Chunk Recall@k"),
        ("chunk_mrr_at_k", "Chunk MRR@k"),
        ("chunk_ndcg_at_k", "Chunk nDCG@k"),
        ("negatives_correct_weak_rate", "Neg. weak-rate"),
        ("avg_retrieval_time_s", "Avg time (s)"),
    ]
    header = "| Config | " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for name, data in per_config.items():
        row = [name]
        for key, _ in cols:
            v = data.get(key)
            if v is None:
                row.append("—")
            elif key == "avg_retrieval_time_s":
                row.append(f"{float(v):.3f}")
            else:
                row.append(f"{float(v):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _run_one(run: str | None, label: str, extra_args: list[str], *, skip_existing: bool) -> Path | None:
    target_path = config.EVAL_DIR / f"results_{label}.jsonl"
    if skip_existing and target_path.exists():
        print(f"[skip] {label}: {target_path} already exists")
        return target_path

    cmd = [sys.executable, str(SCRIPT), "--retrieval-only"]
    if run:
        cmd += ["--run", run]
    cmd += extra_args
    print(f"\n=== Running config: {label} ===")
    print("  " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[error] config '{label}' failed with exit {res.returncode}")
        return None

    src = config.EVAL_DIR / "results_retrieval_only.jsonl"
    if not src.exists():
        print(f"[error] expected output {src} missing for config '{label}'")
        return None
    shutil.move(str(src), str(target_path))
    print(f"  wrote -> {target_path}")
    return target_path


def main() -> None:
    args = parse_args()
    config.apply_run(args.run)

    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    common = ["--top-k", str(args.top_k)]

    plan = [
        ("dense", []),
        ("hybrid", ["--hybrid"]),
        ("dense_rerank", ["--rerank", "--rerank-candidate-k", str(args.rerank_candidate_k)]),
        ("hybrid_rerank", ["--hybrid", "--rerank", "--rerank-candidate-k", str(args.rerank_candidate_k)]),
    ]
    if args.rerank_model:
        for name, _ in plan:
            if "rerank" in name:
                pass
        plan = [
            (n, a + (["--rerank-model", args.rerank_model] if "rerank" in n else []))
            for n, a in plan
        ]

    if only:
        plan = [p for p in plan if p[0] in only]

    produced: dict[str, Path] = {}
    for label, extras in plan:
        path = _run_one(args.run, label, common + extras, skip_existing=args.skip_existing)
        if path is not None:
            produced[label] = path

    if not produced:
        print("No results produced.")
        raise SystemExit(1)

    per_config: dict[str, dict] = {}
    for label, path in produced.items():
        rows = read_jsonl(path)
        per_config[label] = _aggregate_one(rows, top_k=int(args.top_k))

    table = _md_table(per_config)
    print()
    print("Comparison:")
    print(table)

    out_md = config.EVAL_DIR / "retrieval_comparison.md"
    header = f"# Retrieval comparison (top_k={args.top_k}, rerank_candidate_k={args.rerank_candidate_k})\n\n"
    extra = []
    for label, data in per_config.items():
        ann = []
        if "doc_annotated_n" in data:
            ann.append(f"doc_annotated_n={data['doc_annotated_n']}")
        if "chunk_annotated_n" in data:
            ann.append(f"chunk_annotated_n={data['chunk_annotated_n']}")
        if "negatives_n" in data:
            ann.append(f"negatives_n={data['negatives_n']}")
        if ann:
            extra.append(f"- {label}: " + ", ".join(ann))
    notes = ("\n\n## Annotation counts\n" + "\n".join(extra)) if extra else ""
    out_md.write_text(header + table + notes + "\n", encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()

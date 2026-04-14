"""
Quick test: embed a question and show the top retrieved chunks.
No generation model needed - just tests the retrieval pipeline.

Usage: python scripts/test_retrieval.py --run <run_name>
    python scripts/test_retrieval.py --run <run_name> --question "Kako se prijavim na izpit?"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.retrieval import load, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_QUESTIONS = [
    "Kako se prijavim na izpit?",
    "Koliko krat lahko opravljam izpit?",
    "Kako se vpišem v višji letnik?",
    "Kaj je rok za oddajo diplomskega dela?",
    "Kakšna je šolnina za izredni študij?",
]


def _file_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _index_ready(run_root: Path) -> bool:
    return _file_nonempty(run_root / "index" / "index.faiss") and _file_nonempty(
        run_root / "index" / "metadata.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run/dataset name (stored under code/data/runs/<name>/)")
    parser.add_argument("--question", default=None, help="Question to test")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--all", action="store_true", help="Run all default questions")
    return parser.parse_args()


def test_question(question: str, top_k: int) -> None:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    result = retrieve(question, top_k=top_k)

    if result["retrieval_weak"]:
        print("WARN: retrieval weak - no chunk passed the score threshold")

    for i, chunk in enumerate(result["chunks"], 1):
        title = chunk.get("title", "")
        section = chunk.get("section", "")
        score = chunk.get("score", 0.0)
        source = f"{title} - {section}" if section and section != "main" else title
        print(f"\n[{i}] {source}  (score: {score:.4f})")
        print(f"    {chunk['text'][:300].replace(chr(10), ' ')}...")


def main() -> None:
    args = parse_args()

    name = (args.run or config.DEFAULT_RUN_NAME).strip() or config.DEFAULT_RUN_NAME
    run_root = config.RUNS_DIR / name

    # Always reuse/create the run folder.
    config.apply_run(name, mode="update")

    if not _index_ready(run_root):
        print(f"No index found for run '{name}'.")
        print(f"Expected: {config.FAISS_INDEX_FILE} and {config.FAISS_META_FILE}")
        print("Build the index first:")
        print(f"  python scripts/build_index.py --run {name}")
        raise SystemExit(2)

    print(f"Loading index from {config.INDEX_DIR}...")
    t0 = time.perf_counter()
    load()
    t1 = time.perf_counter()
    print("Index loaded.\n")
    print(f"Load time: {t1 - t0:.2f}s\n")

    # Preload the embedding model once so any model/HF logs show up before
    # printing question blocks.
    from src.embeddings import get_model

    get_model(config.EMBEDDING_MODEL)

    if args.question:
        questions = [args.question]
    elif args.all:
        questions = DEFAULT_QUESTIONS
    else:
        questions = DEFAULT_QUESTIONS[:2]

    for q in questions:
        tq0 = time.perf_counter()
        test_question(q, args.top_k)
        tq1 = time.perf_counter()
        print(f"Query time: {tq1 - tq0:.2f}s")


if __name__ == "__main__":
    main()

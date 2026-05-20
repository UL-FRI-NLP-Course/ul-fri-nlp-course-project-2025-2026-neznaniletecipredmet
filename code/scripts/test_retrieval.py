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
    # Negative / out-of-domain: expected not to be answered from the UL FRI corpus.
    "Kakšna je vremenska napoved v Ljubljani jutri?",
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


def _pick_meta_date(meta: dict, keys: list[str]) -> str:
    for key in keys:
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default=None,
        help="Run/dataset name (stored under config.RUNS_DIR; configurable via NLP_RAG_DATA_DIR)",
    )
    parser.add_argument("--question", default=None, help="Question to test")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--all", action="store_true", help="Run all default questions")
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=config.DEFAULT_USE_HYBRID,
        help="Use hybrid retrieval (BM25 + dense)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=config.DEFAULT_USE_RERANK,
        help="Rerank retrieved candidates with a cross-encoder",
    )
    parser.add_argument("--rerank-model", default=None, help="Cross-encoder model name (overrides config.RERANK_MODEL)")
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=None,
        help="How many candidates to retrieve before reranking (overrides config.RERANK_CANDIDATE_K)",
    )
    return parser.parse_args()


def test_question(
    question: str,
    top_k: int,
    *,
    use_hybrid: bool,
    use_rerank: bool,
    rerank_model: str | None,
    rerank_candidate_k: int | None,
) -> None:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    result = retrieve(
        question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        rerank_model=rerank_model,
        rerank_candidate_k=rerank_candidate_k,
    )

    if result["retrieval_weak"]:
        print("WARN: retrieval weak - no chunk passed the score threshold")

    for i, chunk in enumerate(result["chunks"], 1):
        title = chunk.get("title", "")
        section = chunk.get("section", "")
        score = chunk.get("score", 0.0)
        pre = chunk.get("pre_rerank_score", None)
        vec = chunk.get("vector_score", None)
        pre_recency = chunk.get("pre_recency_score", None)
        recency_boost = chunk.get("recency_boost", None)
        recency_score = chunk.get("recency_score", None)
        meta = chunk.get("metadata", {}) or {}
        source = f"{title} - {section}" if section and section != "main" else title
        score_line = f"score: {score:.4f}"
        if pre is not None:
            score_line += f" | pre_rerank: {float(pre):.4f}"
        if vec is not None and pre is None:
            score_line += f" | vector: {float(vec):.4f}"
        if pre_recency is not None:
            delta = float(recency_boost) if recency_boost is not None else (score - float(pre_recency))
            score_line += f" | pre_recency: {float(pre_recency):.4f} | recency_delta: {delta:+.4f}"
            if recency_score is not None:
                score_line += f" | recency_score: {float(recency_score):.4f}"
        print(f"\n[{i}] {source}  ({score_line})")
        created_at = _pick_meta_date(meta, ["created_at", "published_at", "saved_at"])
        modified_at = _pick_meta_date(meta, ["sitemap_lastmod", "modified_at", "http_last_modified", "http_date"])
        if created_at or modified_at:
            created_out = created_at or "n/a"
            modified_out = modified_at or "n/a"
            print(f"    dates: created={created_out} | modified={modified_out}")
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
        test_question(
            q,
            args.top_k,
            use_hybrid=args.hybrid,
            use_rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_candidate_k=args.rerank_candidate_k,
        )
        tq1 = time.perf_counter()
        print(f"Query time: {tq1 - tq0:.2f}s")


if __name__ == "__main__":
    main()

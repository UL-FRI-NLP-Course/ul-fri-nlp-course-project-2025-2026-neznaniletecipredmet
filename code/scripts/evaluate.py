"""
Evaluate a single model on the manually-written question set.
Usage: python scripts/evaluate.py [--model MODEL_NAME]

Eval set format (data/eval/questions.jsonl):
{
  "question": "...",
  "language": "sl" | "en",
  "expected_keywords": ["keyword1", "keyword2"],
  "relevant_doc_ids": ["doc_id_1", ...]
}
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.generation import Generator
from src.pipeline import answer_question
from src.retrieval import load
from src.utils import read_jsonl, write_jsonl

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--run", default=None, help="Run name (data/runs/<name>)")
    parser.add_argument("--model", default=config.GENERATION_MODEL)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--hybrid", action="store_true")
    return parser.parse_args()


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def retrieval_hit(chunks: list[dict], relevant_doc_ids: list[str]) -> bool:
    retrieved_ids = {c["doc_id"] for c in chunks}
    return bool(retrieved_ids & set(relevant_doc_ids))


def evaluate(questions: list[dict], generator: Generator, top_k: int, use_hybrid: bool) -> list[dict]:
    results = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        lang = q.get("language", "sl")
        keywords = q.get("expected_keywords", [])
        relevant_ids = q.get("relevant_doc_ids", [])

        print(f"[{i}/{len(questions)}] {question[:60]}...", end=" ", flush=True)
        t0 = time.time()

        result = answer_question(
            question=question,
            top_k=top_k,
            generator=generator,
            use_hybrid=use_hybrid,
        )

        elapsed = time.time() - t0
        khr = keyword_hit_rate(result["answer"], keywords)
        r_hit = retrieval_hit(result["retrieved_chunks"], relevant_ids)

        print(f"kw_hit={khr:.2f} ret_hit={int(r_hit)} t={elapsed:.1f}s")

        results.append({
            "question": question,
            "language": lang,
            "answer": result["answer"],
            "keyword_hit_rate": khr,
            "retrieval_hit": r_hit,
            "retrieval_weak": result["retrieval_weak"],
            "scores": result["retrieval_scores"],
            "generation_time_s": elapsed,
        })

    return results


def print_summary(results: list[dict], model_name: str) -> None:
    total = len(results)
    if total == 0:
        print("No results.")
        return

    overall_khr = sum(r["keyword_hit_rate"] for r in results) / total
    overall_ret = sum(1 for r in results if r["retrieval_hit"]) / total
    avg_time = sum(r["generation_time_s"] for r in results) / total

    sl = [r for r in results if r["language"] == "sl"]
    en = [r for r in results if r["language"] == "en"]

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Questions: {total} (sl={len(sl)}, en={len(en)})")
    print(f"Retrieval recall@{config.TOP_K}: {overall_ret:.2%}")
    print(f"Keyword hit rate (overall): {overall_khr:.2%}")
    if sl:
        sl_khr = sum(r["keyword_hit_rate"] for r in sl) / len(sl)
        print(f"Keyword hit rate (sl): {sl_khr:.2%}")
    if en:
        en_khr = sum(r["keyword_hit_rate"] for r in en) / len(en)
        print(f"Keyword hit rate (en): {en_khr:.2%}")
    print(f"Avg generation time: {avg_time:.1f}s")
    print(f"{'='*50}\n")


def main() -> None:
    args = parse_args()

    config.apply_run(getattr(args, "run", None))

    if not config.EVAL_QUESTIONS_FILE.exists():
        print(f"Eval file not found: {config.EVAL_QUESTIONS_FILE}")
        print("Create data/eval/questions.jsonl with your evaluation questions.")
        sys.exit(1)

    questions = read_jsonl(config.EVAL_QUESTIONS_FILE)
    print(f"Loaded {len(questions)} evaluation questions")

    print("Loading index...")
    load()

    print(f"Loading model: {args.model}")
    generator = Generator(model_name=args.model)

    results = evaluate(questions, generator, args.top_k, args.hybrid)

    print_summary(results, args.model)

    model_slug = args.model.replace("/", "_")
    out_path = config.EVAL_DIR / f"results_{model_slug}.jsonl"
    write_jsonl(results, out_path)
    print(f"Detailed results saved to {out_path}")

    generator.unload()


if __name__ == "__main__":
    main()

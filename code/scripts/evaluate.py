"""
Evaluate a single model on the manually-written question set.
Usage: python scripts/evaluate.py [--model MODEL_NAME]

Eval set format (<run>/eval/questions.jsonl):
{
  "question": "...",
  "language": "sl" | "en",
  "expected_keywords": ["keyword1", "keyword2"],
    "relevant_doc_ids": ["doc_id_1", ...],
    "relevant_chunk_ids": ["chunk_id_1", ...]
}
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.eval_metrics import (
    chunk_summaries,
    retrieval_chunk_metrics,
    retrieval_keyword_proxy,
    retrieval_metrics,
)
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
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and only evaluate retrieval (fast, CPU-friendly)",
    )
    parser.add_argument("--rerank", action="store_true", help="Rerank retrieved candidates with a cross-encoder")
    parser.add_argument("--rerank-model", default=None, help="Cross-encoder model name (overrides config.RERANK_MODEL)")
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=None,
        help="How many candidates to retrieve before reranking (overrides config.RERANK_CANDIDATE_K)",
    )
    return parser.parse_args()


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)




def evaluate(
    questions: list[dict],
    generator,
    top_k: int,
    use_hybrid: bool,
    *,
    use_rerank: bool,
    rerank_model: str | None,
    rerank_candidate_k: int | None,
) -> list[dict]:
    from src.pipeline import answer_question

    results = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        lang = q.get("language", "sl")
        keywords = q.get("expected_keywords", [])
        relevant_ids = q.get("relevant_doc_ids", [])
        relevant_chunk_ids = q.get("relevant_chunk_ids", [])
        is_negative = bool(q.get("is_negative", False))

        print(f"[{i}/{len(questions)}] {question[:60]}...", end=" ", flush=True)
        t0 = time.time()

        result = answer_question(
            question=question,
            top_k=top_k,
            generator=generator,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            rerank_model=rerank_model,
            rerank_candidate_k=rerank_candidate_k,
        )

        elapsed = time.time() - t0
        khr = keyword_hit_rate(result["answer"], keywords)
        r_metrics = retrieval_metrics(result["retrieved_chunks"], relevant_ids, top_k=top_k)
        c_metrics = retrieval_chunk_metrics(result["retrieved_chunks"], relevant_chunk_ids, top_k=top_k)
        kw_proxy = retrieval_keyword_proxy(result["retrieved_chunks"], keywords, top_k=top_k)

        negative_ok = None
        if is_negative:
            negative_ok = bool(result.get("retrieval_weak", False))

        ret_hit_display = r_metrics["retrieval_hit"]
        ret_hit_str = "NA" if ret_hit_display is None else str(int(bool(ret_hit_display)))
        print(f"kw_hit={khr:.2f} ret_hit={ret_hit_str} t={elapsed:.1f}s")

        results.append({
            "question": question,
            "language": lang,
            "answer": result["answer"],
            "keyword_hit_rate": khr,
            **r_metrics,
            **c_metrics,
            **kw_proxy,
            "retrieval_weak": result["retrieval_weak"],
            "is_negative": is_negative,
            "negative_ok": negative_ok,
            "scores": result["retrieval_scores"],
            "retrieved": chunk_summaries(result.get("retrieved_chunks", [])),
            "generation_time_s": elapsed,
        })

    return results


def print_summary(results: list[dict], model_name: str, *, top_k: int) -> None:
    total = len(results)
    if total == 0:
        print("No results.")
        return

    khr_vals = [r.get("keyword_hit_rate") for r in results if r.get("keyword_hit_rate") is not None]
    overall_khr = (sum(float(v) for v in khr_vals) / len(khr_vals)) if khr_vals else None

    doc_annotated = [r for r in results if r.get("retrieval_hit") is not None]
    overall_ret = None
    overall_prec = None
    overall_mrr = None
    overall_ndcg = None
    if doc_annotated:
        overall_ret = sum(1 for r in doc_annotated if r.get("retrieval_hit")) / len(doc_annotated)
        overall_prec = sum(float(r.get("precision_at_k") or 0.0) for r in doc_annotated) / len(doc_annotated)
        overall_mrr = sum(float(r.get("mrr_at_k") or 0.0) for r in doc_annotated) / len(doc_annotated)
        overall_ndcg = sum(float(r.get("ndcg_at_k") or 0.0) for r in doc_annotated) / len(doc_annotated)

    chunk_annotated = [r for r in results if r.get("chunk_hit") is not None]
    overall_chunk_ret = None
    overall_chunk_prec = None
    overall_chunk_mrr = None
    overall_chunk_ndcg = None
    if chunk_annotated:
        overall_chunk_ret = sum(1 for r in chunk_annotated if r.get("chunk_hit")) / len(chunk_annotated)
        overall_chunk_prec = sum(float(r.get("chunk_precision_at_k") or 0.0) for r in chunk_annotated) / len(chunk_annotated)
        overall_chunk_mrr = sum(float(r.get("chunk_mrr_at_k") or 0.0) for r in chunk_annotated) / len(chunk_annotated)
        overall_chunk_ndcg = sum(float(r.get("chunk_ndcg_at_k") or 0.0) for r in chunk_annotated) / len(chunk_annotated)

    kw_proxy_annotated = [r for r in results if r.get("retrieved_keyword_hit") is not None]
    overall_kw_ret = None
    if kw_proxy_annotated:
        overall_kw_ret = sum(1 for r in kw_proxy_annotated if r.get("retrieved_keyword_hit")) / len(kw_proxy_annotated)
    time_vals = []
    for r in results:
        if r.get("generation_time_s") is not None:
            time_vals.append(float(r["generation_time_s"]))
        elif r.get("retrieval_time_s") is not None:
            time_vals.append(float(r["retrieval_time_s"]))
    avg_time = (sum(time_vals) / len(time_vals)) if time_vals else None

    sl = [r for r in results if r["language"] == "sl"]
    en = [r for r in results if r["language"] == "en"]
    negatives = [r for r in results if r.get("is_negative")]

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Questions: {total} (sl={len(sl)}, en={len(en)})")
    if overall_ret is not None:
        print(f"Retrieval recall@{top_k}: {overall_ret:.2%} (doc_id annotated n={len(doc_annotated)})")
        print(f"Retrieval precision@{top_k}: {overall_prec:.2%}")
        print(f"Retrieval MRR@{top_k}: {overall_mrr:.2%}")
        print(f"Retrieval nDCG@{top_k}: {overall_ndcg:.2%}")
    else:
        print("Retrieval metrics (doc_id): NA (add relevant_doc_ids to questions.jsonl)")

    if overall_chunk_ret is not None:
        print(f"Chunk recall@{top_k}: {overall_chunk_ret:.2%} (chunk_id annotated n={len(chunk_annotated)})")
        print(f"Chunk precision@{top_k}: {overall_chunk_prec:.2%}")
        print(f"Chunk MRR@{top_k}: {overall_chunk_mrr:.2%}")
        print(f"Chunk nDCG@{top_k}: {overall_chunk_ndcg:.2%}")
    else:
        print("Retrieval metrics (chunk_id): NA (add relevant_chunk_ids to questions.jsonl)")
    if overall_kw_ret is not None:
        print(f"Retrieval keyword proxy@{top_k}: {overall_kw_ret:.2%} (n={len(kw_proxy_annotated)})")
    if overall_khr is not None:
        print(f"Keyword hit rate (overall): {overall_khr:.2%}")
        sl_khr_vals = [r.get("keyword_hit_rate") for r in sl if r.get("keyword_hit_rate") is not None]
        en_khr_vals = [r.get("keyword_hit_rate") for r in en if r.get("keyword_hit_rate") is not None]
        if sl_khr_vals:
            sl_khr = sum(float(v) for v in sl_khr_vals) / len(sl_khr_vals)
            print(f"Keyword hit rate (sl): {sl_khr:.2%}")
        if en_khr_vals:
            en_khr = sum(float(v) for v in en_khr_vals) / len(en_khr_vals)
            print(f"Keyword hit rate (en): {en_khr:.2%}")
    if avg_time is not None:
        label = "Avg generation time" if any(r.get("generation_time_s") is not None for r in results) else "Avg retrieval time"
        print(f"{label}: {avg_time:.2f}s")
    if negatives:
        ok = [r for r in negatives if r.get("negative_ok") is True]
        print(f"Negative questions: {len(negatives)} | retrieval_weak true: {len(ok)}/{len(negatives)}")
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

    if args.retrieval_only:
        # Retrieval-only path: no generator needed.
        from src.retrieval import retrieve

        results = []
        for i, q in enumerate(questions, 1):
            question = q["question"]
            lang = q.get("language", "sl")
            keywords = q.get("expected_keywords", [])
            relevant_ids = q.get("relevant_doc_ids", [])
            relevant_chunk_ids = q.get("relevant_chunk_ids", [])
            is_negative = bool(q.get("is_negative", False))

            print(f"[{i}/{len(questions)}] {question[:60]}...", end=" ", flush=True)
            t0 = time.time()

            r = retrieve(
                question,
                top_k=args.top_k,
                use_hybrid=args.hybrid,
                use_rerank=args.rerank,
                rerank_model=args.rerank_model,
                rerank_candidate_k=args.rerank_candidate_k,
            )
            elapsed = time.time() - t0

            r_metrics = retrieval_metrics(r["chunks"], relevant_ids, top_k=args.top_k)
            c_metrics = retrieval_chunk_metrics(r["chunks"], relevant_chunk_ids, top_k=args.top_k)
            kw_proxy = retrieval_keyword_proxy(r["chunks"], keywords, top_k=args.top_k)

            negative_ok = None
            if is_negative:
                negative_ok = bool(r.get("retrieval_weak", False))

            ret_hit_display = r_metrics["retrieval_hit"]
            ret_hit_str = "NA" if ret_hit_display is None else str(int(bool(ret_hit_display)))
            print(f"ret_hit={ret_hit_str} weak={int(bool(r.get('retrieval_weak')))} t={elapsed:.2f}s")

            results.append({
                "question": question,
                "language": lang,
                **r_metrics,
                **c_metrics,
                **kw_proxy,
                "retrieval_weak": r.get("retrieval_weak", False),
                "is_negative": is_negative,
                "negative_ok": negative_ok,
                "scores": r.get("scores", []),
                "retrieved": chunk_summaries(r.get("chunks", [])),
                "retrieval_time_s": elapsed,
            })
    else:
        print(f"Loading model: {args.model}")
        from src.generation import Generator
        generator = Generator(model_name=args.model)

        results = evaluate(
            questions,
            generator,
            args.top_k,
            args.hybrid,
            use_rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_candidate_k=args.rerank_candidate_k,
        )

    print_summary(results, "retrieval-only" if args.retrieval_only else args.model, top_k=args.top_k)

    model_slug = "retrieval_only" if args.retrieval_only else args.model.replace("/", "_")
    out_path = config.EVAL_DIR / f"results_{model_slug}.jsonl"
    write_jsonl(results, out_path)
    print(f"Detailed results saved to {out_path}")

    if not args.retrieval_only:
        generator.unload()


if __name__ == "__main__":
    main()

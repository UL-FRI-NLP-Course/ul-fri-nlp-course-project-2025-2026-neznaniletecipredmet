"""
Quick test: embed a question and show the top retrieved chunks.
No generation model needed — just tests the retrieval pipeline.

Usage: python scripts/test_retrieval.py
       python scripts/test_retrieval.py --question "Kako se prijavim na izpit?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.retrieval import load, retrieve

DEFAULT_QUESTIONS = [
    "Kako se prijavim na izpit?",
    "Koliko krat lahko opravljam izpit?",
    "Kako se vpišem v višji letnik?",
    "Kaj je rok za oddajo diplomskega dela?",
    "Kakšna je šolnina za izredni študij?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        print("⚠  Retrieval weak — no chunk passed the score threshold")

    for i, chunk in enumerate(result["chunks"], 1):
        title = chunk.get("title", "")
        section = chunk.get("section", "")
        score = chunk.get("score", 0.0)
        source = f"{title} — {section}" if section and section != "main" else title
        print(f"\n[{i}] {source}  (score: {score:.4f})")
        print(f"    {chunk['text'][:300].replace(chr(10), ' ')}...")


def main() -> None:
    args = parse_args()

    print(f"Loading index from {config.INDEX_DIR}...")
    load()
    print("Index loaded.\n")

    if args.question:
        questions = [args.question]
    elif args.all:
        questions = DEFAULT_QUESTIONS
    else:
        questions = DEFAULT_QUESTIONS[:2]

    for q in questions:
        test_question(q, args.top_k)


if __name__ == "__main__":
    main()

"""
Ask a question to the FRI student-office RAG chatbot using a prebuilt index.

QUICK START
-----------
1. Download index.faiss and metadata.json from Google Drive:
   https://drive.google.com/drive/folders/1FOESoezBwMJ8q9DR3cYrYjHSmsVSoAsM

2. Place the files in:
   code/data/runs/2026-05-20__full__fri4_ul2__v1/index/

3. Run (retrieval only, no GPU needed):
   python scripts/ask.py --question "Kdaj so uradne ure študentskega referata?"

4. Run with a generator model (GPU recommended):
   python scripts/ask.py \\
       --question "Kdaj so uradne ure študentskega referata?" \\
       --model meta-llama/Llama-3.1-8B-Instruct

The --index-dir flag lets you point to any directory that contains
index.faiss and metadata.json, in case you put the files elsewhere:
   python scripts/ask.py --index-dir /path/to/index --question "..."
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure `code/` is on the path regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# Default index location: code/data/runs/<run>/index/
_DEFAULT_RUN = "2026-05-20__full__fri4_ul2__v1"
_DEFAULT_INDEX_DIR = (
    Path(__file__).parent.parent / "data" / "runs" / _DEFAULT_RUN / "index"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ask a question to the FRI RAG chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--index-dir",
        default=None,
        metavar="DIR",
        help=(
            "Folder containing index.faiss and metadata.json. "
            f"Defaults to {_DEFAULT_INDEX_DIR}"
        ),
    )
    p.add_argument(
        "--question", "-q",
        default=None,
        help="Question to ask. If omitted the script will prompt you.",
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "Hugging Face model name for answer generation "
            "(e.g. meta-llama/Llama-3.1-8B-Instruct). "
            "Omit this flag to run retrieval only without a GPU."
        ),
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    p.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=30,
        help="Candidate pool size for the cross-encoder reranker (default: 30)",
    )
    p.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    p.add_argument("--no-hybrid", action="store_true", help="Use dense-only retrieval instead of BM25+dense")
    return p.parse_args()


def _resolve_index_dir(arg: str | None) -> Path:
    if arg is not None:
        return Path(arg).resolve()
    return _DEFAULT_INDEX_DIR.resolve()


def _check_index(index_dir: Path) -> None:
    missing = [f for f in ("index.faiss", "metadata.json") if not (index_dir / f).exists()]
    if missing:
        print(f"ERROR: the following files are missing from {index_dir}:")
        for f in missing:
            print(f"  {f}")
        print()
        print("Download them from Google Drive and place them in that folder:")
        print("  https://drive.google.com/drive/folders/1FOESoezBwMJ8q9DR3cYrYjHSmsVSoAsM")
        print()
        print(f"Expected location:  {index_dir}")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # --- Point config at the right index ---
    index_dir = _resolve_index_dir(args.index_dir)
    _check_index(index_dir)

    config.INDEX_DIR = index_dir
    config.FAISS_INDEX_FILE = index_dir / "index.faiss"
    config.FAISS_META_FILE = index_dir / "metadata.json"

    # --- Load the index ---
    from src.retrieval import load, retrieve

    print(f"Loading index from {index_dir} ...")
    t0 = time.perf_counter()
    load()
    print(f"Index loaded in {time.perf_counter() - t0:.1f}s\n")

    # --- Get the question ---
    question = args.question
    if not question:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
    if not question:
        print("No question provided.")
        sys.exit(1)

    print(f"Question: {question}\n")

    # --- Retrieve ---
    t0 = time.perf_counter()
    result = retrieve(
        question,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_rerank=not args.no_rerank,
        rerank_candidate_k=args.rerank_candidate_k,
    )
    print(f"Retrieval done in {time.perf_counter() - t0:.1f}s")

    if result["retrieval_weak"]:
        print("WARNING: retrieval confidence is low — the answer may not be grounded in the corpus.")
    print()

    print("Top retrieved passages:")
    print("-" * 60)
    for i, chunk in enumerate(result["chunks"], 1):
        title = chunk.get("title", "")
        section = chunk.get("section", "")
        score = chunk.get("score", 0.0)
        source = f"{title} — {section}" if section and section not in ("main", "") else title
        print(f"[{i}] {source}  (score: {score:.4f})")
        print(f"    {chunk['text'][:250].replace(chr(10), ' ')}...")
        print()

    # --- Generate ---
    if args.model:
        print(f"Generating answer with {args.model} ...")
        print("(This downloads the model on first run — may take a few minutes.)\n")

        from src.generation import Generator
        from src.prompting import build_prompt
        from src.utils import detect_language

        language = detect_language(question)
        messages = build_prompt(
            question=question,
            chunks=result["chunks"],
            language=language,
            retrieval_weak=result["retrieval_weak"],
        )

        gen = Generator(args.model)
        t0 = time.perf_counter()
        answer = gen.generate(messages)
        print(f"Generation done in {time.perf_counter() - t0:.1f}s\n")

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
    else:
        print("(Showing retrieved passages only. Pass --model <name> to generate a full answer.)")
        print("Example: --model meta-llama/Llama-3.1-8B-Instruct")


if __name__ == "__main__":
    main()

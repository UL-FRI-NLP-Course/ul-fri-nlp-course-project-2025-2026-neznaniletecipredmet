"""Retrieval helpers.

This module wraps the vector store + embedding model into a simple API that
scripts can use without duplicating boilerplate.

Does not call any LLMs.
"""

from __future__ import annotations

import logging

import numpy as np

import config
from src.embeddings import embed_queries
from src.utils import detect_language
from src.vector_store import load_index, search

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

log = logging.getLogger(__name__)

_INDEX: object | None = None
_CHUNKS: list[dict] | None = None


def load() -> None:
    """Load the FAISS index + metadata into memory."""
    global _INDEX, _CHUNKS
    _INDEX, _CHUNKS = load_index(use_gpu=False)


def _ensure_loaded() -> tuple[object, list[dict]]:
    if _INDEX is None or _CHUNKS is None:
        load()
    assert _INDEX is not None
    assert _CHUNKS is not None
    return _INDEX, _CHUNKS


def _hybrid_retrieve(question: str, chunks: list[dict], semantic_results: list[dict], top_k: int) -> list[dict]:
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(question.lower().split())

    candidate_k = min(top_k * 4, len(chunks))
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
    semantic_indices = {chunks.index(r) if r in chunks else -1 for r in semantic_results}

    candidate_indices = list(set(list(bm25_top_indices)) | semantic_indices - {-1})

    sem_scores = {i: 0.0 for i in candidate_indices}
    for r in semantic_results:
        for i, c in enumerate(chunks):
            if c.get("chunk_id") == r.get("chunk_id"):
                sem_scores[i] = float(r["score"])
                break

    sem_vals = np.array([sem_scores.get(i, 0.0) for i in candidate_indices])
    bm25_vals = np.array([bm25_scores[i] for i in candidate_indices])

    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    fused = 0.5 * _norm(sem_vals) + 0.5 * _norm(bm25_vals)
    top = np.argsort(fused)[::-1][:top_k]

    results = []
    for rank_idx in top:
        chunk_idx = candidate_indices[rank_idx]
        result = dict(chunks[chunk_idx])
        result["score"] = float(fused[rank_idx])
        results.append(result)
    return results


def retrieve(
    question: str,
    *,
    top_k: int = config.TOP_K,
    language: str | None = None,
    score_threshold: float = config.RETRIEVAL_SCORE_THRESHOLD,
    use_hybrid: bool = False,
) -> dict:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if language is None:
        language = detect_language(question, default=config.DEFAULT_LANGUAGE)

    index, chunks = _ensure_loaded()

    query_embedding = embed_queries([question])
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)

    semantic_results = search(query_embedding, index, chunks, top_k=top_k)

    if use_hybrid and _BM25_AVAILABLE:
        results = _hybrid_retrieve(question, chunks, semantic_results, top_k)
    else:
        if use_hybrid and not _BM25_AVAILABLE:
            log.warning("rank_bm25 not installed, falling back to semantic-only retrieval")
        results = semantic_results

    seen = set()
    deduped = []
    for r in results:
        key = r.get("text", "")[:120]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    results = deduped[:top_k]

    scores = [float(r.get("score", 0.0)) for r in results]
    passing = [s for s in scores if s >= score_threshold]
    retrieval_weak = len(passing) == 0

    return {
        "chunks": results,
        "scores": scores,
        "retrieval_weak": retrieval_weak,
        "language": language,
    }

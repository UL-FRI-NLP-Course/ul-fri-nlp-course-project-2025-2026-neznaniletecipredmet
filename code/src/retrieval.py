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
from src.reranking import rerank_chunks
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
    tokenized_corpus = [c.get("text", "").lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(question.lower().split())

    chunk_id_to_index: dict[str, int] = {}
    for i, c in enumerate(chunks):
        cid = c.get("chunk_id")
        if isinstance(cid, str) and cid and cid not in chunk_id_to_index:
            chunk_id_to_index[cid] = i

    # Candidate pool = BM25 top (lexical) union semantic top.
    candidate_k = min(max(top_k * 4, top_k), len(chunks))
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

    semantic_top_indices: list[int] = []
    for r in semantic_results:
        cid = r.get("chunk_id")
        if isinstance(cid, str) and cid in chunk_id_to_index:
            semantic_top_indices.append(chunk_id_to_index[cid])

    candidate_indices = sorted(set(map(int, bm25_top_indices)) | set(semantic_top_indices))

    semantic_score_by_index: dict[int, float] = {}
    for r in semantic_results:
        cid = r.get("chunk_id")
        if isinstance(cid, str) and cid in chunk_id_to_index:
            semantic_score_by_index[chunk_id_to_index[cid]] = float(r.get("score", 0.0))

    sem_vals = np.array([semantic_score_by_index.get(i, 0.0) for i in candidate_indices], dtype=np.float32)
    bm25_vals = np.array([bm25_scores[i] for i in candidate_indices], dtype=np.float32)

    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    fused = 0.5 * _norm(sem_vals) + 0.5 * _norm(bm25_vals)
    top = np.argsort(fused)[::-1][:top_k]

    results = []
    for rank_idx in top:
        chunk_idx = candidate_indices[rank_idx]
        result = dict(chunks[chunk_idx])
        result["bm25_score"] = float(bm25_scores[chunk_idx])
        result["vector_score"] = float(semantic_score_by_index.get(chunk_idx, 0.0))
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
    use_rerank: bool = False,
    rerank_model: str | None = None,
    rerank_candidate_k: int | None = None,
) -> dict:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if language is None:
        language = detect_language(question, default=config.DEFAULT_LANGUAGE)

    index, chunks = _ensure_loaded()

    candidate_k = top_k
    if use_rerank:
        default_candidate_k = getattr(config, "RERANK_CANDIDATE_K", max(top_k * 5, 20))
        candidate_k = max(top_k, int(rerank_candidate_k or default_candidate_k))
    candidate_k = min(candidate_k, len(chunks))

    query_embedding = embed_queries([question])
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)

    semantic_results = search(query_embedding, index, chunks, top_k=candidate_k)
    for r in semantic_results:
        # Preserve vector-search score even if later stages overwrite `score`.
        if "vector_score" not in r and "score" in r:
            r["vector_score"] = float(r["score"])

    if use_hybrid and _BM25_AVAILABLE:
        results = _hybrid_retrieve(question, chunks, semantic_results, candidate_k)
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
    results = deduped[:candidate_k]

    if use_rerank:
        model_name = rerank_model or getattr(config, "RERANK_MODEL", None)
        results = rerank_chunks(question, results, model_name=model_name, top_k=top_k)
    else:
        results = results[:top_k]

    scores = [float(r.get("score", 0.0)) for r in results]

    # `score` may be overwritten by reranking; keep the "weak retrieval" signal
    # based on the original retrieval scores (cosine / hybrid-fused).
    base_scores = [float(r.get("pre_rerank_score", r.get("score", 0.0))) for r in results]
    passing = [s for s in base_scores if s >= score_threshold]
    retrieval_weak = len(passing) == 0

    return {
        "chunks": results,
        "scores": scores,
        "retrieval_weak": retrieval_weak,
        "language": language,
    }

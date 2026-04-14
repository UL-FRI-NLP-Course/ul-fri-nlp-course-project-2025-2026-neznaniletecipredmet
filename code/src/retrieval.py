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

log = logging.getLogger(__name__)

_INDEX: object | None = None
_CHUNKS: list[dict] | None = None


def load() -> None:
    """Load the FAISS index + metadata into memory."""
    global _INDEX, _CHUNKS
    _INDEX, _CHUNKS = load_index()


def _ensure_loaded() -> tuple[object, list[dict]]:
    if _INDEX is None or _CHUNKS is None:
        load()
    assert _INDEX is not None
    assert _CHUNKS is not None
    return _INDEX, _CHUNKS


def retrieve(
    question: str,
    *,
    top_k: int = config.TOP_K,
    language: str | None = None,
    score_threshold: float = config.RETRIEVAL_SCORE_THRESHOLD,
) -> dict:
    """Retrieve top chunks for a question.

    Returns a dict shaped for scripts/test_retrieval.py:
      {"chunks": [...], "retrieval_weak": bool, "language": "sl"|"en"}

    Notes:
        - If no chunk passes 'score_threshold', we still return the top-k results, but set 'retrieval_weak=True'.
    - Language filtering is optional; if enabled, it only filters when the
            stored chunks have 'language' field.
    """

    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if language is None:
        language = detect_language(question, default=config.DEFAULT_LANGUAGE)

    index, chunks = _ensure_loaded()

    query_embedding = embed_queries([question])
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)

    results = search(query_embedding, index, chunks, top_k=top_k)

    # Optional language filtering (conservative to avoid surprising empties).
    if language in config.SUPPORTED_LANGUAGES:
        filtered = [r for r in results if r.get("language") in (None, "", language)]
        if filtered:
            results = filtered

    passing = [r for r in results if float(r.get("score", 0.0)) >= score_threshold]
    retrieval_weak = len(passing) == 0

    return {
        "chunks": results,
        "retrieval_weak": retrieval_weak,
        "language": language,
    }

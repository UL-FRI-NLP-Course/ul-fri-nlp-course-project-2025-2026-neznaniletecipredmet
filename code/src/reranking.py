"""Reranking implementation using CrossEncoders."""

from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)

_RERANKER: object | None = None
_RERANKER_NAME: str | None = None


def _get_default_model_name() -> str | None:
    return getattr(config, "RERANK_MODEL", None)


def get_reranker(model_name: str | None = None) -> object:
    """Return a cached CrossEncoder instance."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "CrossEncoder reranking requires `sentence-transformers` (already in requirements.txt)."
        ) from e

    global _RERANKER, _RERANKER_NAME

    name = (model_name or _get_default_model_name() or "").strip() or None
    if name is None:
        raise ValueError(
            "No reranker model configured. Set config.RERANK_MODEL or pass `rerank_model` to retrieve()."
        )

    if _RERANKER is None or _RERANKER_NAME != name:
        log.info("Loading reranker model: %s", name)
        _RERANKER = CrossEncoder(name)
        _RERANKER_NAME = name
        log.info("Reranker loaded")

    return _RERANKER


def _pair_text(chunk: dict) -> str:
    """Build the chunk text used for reranking."""

    title = (chunk.get("title") or "").strip()
    section = (chunk.get("section") or "").strip()
    body = (chunk.get("text") or "").strip()

    header = title
    if section and section not in ("main", ""):
        header = f"{title} — {section}" if title else section

    if header:
        return f"{header}\n\n{body}"
    return body


def rerank_chunks(
    question: str,
    chunks: list[dict],
    *,
    model_name: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Rerank chunks for a question using a cross-encoder.

    Returns a *new list* (dict-copied) sorted by rerank score desc.
    Sets:
    - `rerank_score`: cross-encoder score
    - `score`: final score used for ordering (same as rerank_score)

    Preserves any existing `score` by storing it in `pre_rerank_score`.
    """

    if not chunks:
        return []

    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    reranker = get_reranker(model_name=model_name)

    pairs = [(question, _pair_text(c)) for c in chunks]

    try:
        scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception as e:  # pragma: no cover
        log.exception("Reranker predict() failed")
        raise RuntimeError(f"Reranking failed: {e}") from e

    # Convert numpy/scalar outputs to plain floats.
    scores_list = [float(s) for s in scores]

    order = sorted(range(len(chunks)), key=lambda i: scores_list[i], reverse=True)

    reranked: list[dict] = []
    for i in order:
        c = dict(chunks[i])
        if "pre_rerank_score" not in c and "score" in c:
            c["pre_rerank_score"] = float(c.get("score", 0.0))
        c["rerank_score"] = float(scores_list[i])
        c["score"] = float(scores_list[i])
        reranked.append(c)

    if top_k is not None:
        return reranked[: int(top_k)]
    return reranked

"""Retrieval helpers.

This module wraps the vector store + embedding model into a simple API that
scripts can use without duplicating boilerplate.

Does not call any LLMs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np

import config
from src.embeddings import embed_queries
from src.reranking import rerank_chunks
from src.utils import detect_language, ensure_utc, parse_datetime
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


def _pick_doc_datetime(metadata: dict) -> datetime | None:
    date_fields = getattr(
        config,
        "RECENCY_DATE_FIELDS",
        ["created_at", "published_at", "modified_at", "http_last_modified"],
    )
    for key in date_fields:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            dt = parse_datetime(value)
            if dt is not None:
                return ensure_utc(dt)
    return None


def _apply_recency_boost(results: list[dict]) -> list[dict]:
    weight = float(getattr(config, "RECENCY_WEIGHT", 0.0) or 0.0)
    half_life_days = float(getattr(config, "RECENCY_HALF_LIFE_DAYS", 0.0) or 0.0)
    if weight <= 0.0 or half_life_days <= 0.0:
        return results

    now = datetime.now(timezone.utc)
    for r in results:
        meta = r.get("metadata", {}) or {}
        doc_dt = _pick_doc_datetime(meta)
        if doc_dt is None:
            continue

        age_days = max(0.0, (now - doc_dt).total_seconds() / 86400.0)
        recency = 0.5 ** (age_days / half_life_days)
        if "pre_recency_score" not in r and "score" in r:
            r["pre_recency_score"] = float(r.get("score", 0.0))
        boost = weight * recency
        r["recency_score"] = float(recency)
        r["recency_boost"] = float(boost)
        r["score"] = float(r.get("score", 0.0)) + boost

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results


def _apply_domain_bias(results: list[dict]) -> list[dict]:
    if not getattr(config, "DOMAIN_BIAS_ENABLE", False):
        return results

    fri_bias = float(getattr(config, "DOMAIN_BIAS_FRI", 0.0) or 0.0)
    ul_bias = float(getattr(config, "DOMAIN_BIAS_UL", 0.0) or 0.0)
    other_ul_penalty = float(getattr(config, "DOMAIN_BIAS_OTHER_UL", 0.0) or 0.0)

    for r in results:
        url = r.get("url") or ""
        netloc = _extract_domain(url)
        delta = 0.0
        if netloc == "fri.uni-lj.si":
            delta = fri_bias
        elif netloc == "uni-lj.si" or netloc.endswith(".uni-lj.si"):
            # Favor the main uni-lj.si slightly; penalize other faculties.
            if netloc == "uni-lj.si":
                delta = ul_bias
            else:
                delta = other_ul_penalty
        if delta != 0.0:
            if "pre_domain_score" not in r:
                r["pre_domain_score"] = float(r.get("score", 0.0))
            r.setdefault("domain_bias", 0.0)
            r["domain_bias"] = float(r.get("domain_bias", 0.0)) + delta
            r["score"] = float(r.get("score", 0.0)) + delta

    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return netloc.split(":", 1)[0]


def _domain_allowed(netloc: str, allowed: list[str]) -> bool:
    if not allowed:
        return True
    for domain in allowed:
        d = (domain or "").lower().strip()
        if not d:
            continue
        if netloc == d or netloc.endswith("." + d) or netloc.endswith(d):
            return True
    return False


def _apply_filters(
    results: list[dict],
    *,
    language: str,
    filter_language: bool,
    allowed_domains: list[str],
    strict: bool,
    target_k: int,
) -> list[dict]:
    if not filter_language and not allowed_domains:
        return results

    filtered: list[dict] = []
    for r in results:
        if filter_language:
            lang = r.get("language")
            if lang and lang != language:
                continue
        if allowed_domains:
            url = r.get("url") or ""
            netloc = _extract_domain(url)
            if netloc and not _domain_allowed(netloc, allowed_domains):
                continue
        filtered.append(r)

    if strict:
        return filtered

    if len(filtered) >= target_k:
        return filtered[:target_k]

    seen: set[str] = set()
    for r in filtered:
        key = str(r.get("chunk_id") or r.get("text", "")[:120])
        seen.add(key)
    for r in results:
        key = str(r.get("chunk_id") or r.get("text", "")[:120])
        if key in seen:
            continue
        filtered.append(r)
        seen.add(key)
        if len(filtered) >= target_k:
            break

    return filtered


def retrieve(
    question: str,
    *,
    top_k: int = config.TOP_K,
    language: str | None = None,
    score_threshold: float = config.RETRIEVAL_SCORE_THRESHOLD,
    use_hybrid: bool | None = None,
    use_rerank: bool | None = None,
    rerank_model: str | None = None,
    rerank_candidate_k: int | None = None,
    filter_language: bool | None = None,
    allowed_domains: list[str] | None = None,
    strict_filters: bool | None = None,
) -> dict:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if language is None:
        language = detect_language(question, default=config.DEFAULT_LANGUAGE)

    index, chunks = _ensure_loaded()

    if use_hybrid is None:
        use_hybrid = bool(getattr(config, "DEFAULT_USE_HYBRID", False))
    if use_rerank is None:
        use_rerank = bool(getattr(config, "DEFAULT_USE_RERANK", False))
    if filter_language is None:
        filter_language = bool(getattr(config, "RETRIEVAL_FILTER_LANGUAGE", False))
    if allowed_domains is None:
        allowed_domains = list(getattr(config, "RETRIEVAL_ALLOWED_DOMAINS", []) or [])
    if strict_filters is None:
        strict_filters = bool(getattr(config, "RETRIEVAL_FILTER_STRICT", False))

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

    results = _apply_filters(
        results,
        language=language,
        filter_language=bool(filter_language),
        allowed_domains=allowed_domains,
        strict=bool(strict_filters),
        target_k=candidate_k,
    )

    if use_rerank:
        model_name = rerank_model or getattr(config, "RERANK_MODEL", None)
        results = rerank_chunks(question, results, model_name=model_name, top_k=top_k)
    else:
        results = results[:top_k]

    # `score` may be overwritten by reranking; keep the "weak retrieval" signal
    # based on the original retrieval scores (cosine / hybrid-fused).
    base_scores = [float(r.get("pre_rerank_score", r.get("score", 0.0))) for r in results]
    passing = [s for s in base_scores if s >= score_threshold]
    retrieval_weak = len(passing) == 0

    results = _apply_recency_boost(results)
    results = _apply_domain_bias(results)
    scores = [float(r.get("score", 0.0)) for r in results]

    return {
        "chunks": results,
        "scores": scores,
        "retrieval_weak": retrieval_weak,
        "language": language,
    }

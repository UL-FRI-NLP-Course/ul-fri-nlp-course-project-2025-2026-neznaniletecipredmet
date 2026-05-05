"""Evaluation metrics.

Core metric implementations for retrieval evaluation, separated 
from model dependencies for testing.
"""

from __future__ import annotations

import math


def _retrieved_doc_ids(chunks: list[dict]) -> list[str]:
    ids: list[str] = []
    for c in chunks:
        doc_id = c.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            ids.append(doc_id)
    return ids


def _retrieved_chunk_ids(chunks: list[dict]) -> list[str]:
    ids: list[str] = []
    for c in chunks:
        chunk_id = c.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            ids.append(chunk_id)
    return ids


def _keyword_in_chunk(chunk: dict, keywords: list[str]) -> bool:
    if not keywords:
        return False
    text = (chunk.get("text") or "").lower()
    return any(kw.lower() in text for kw in keywords)


def retrieval_metrics(chunks: list[dict], relevant_doc_ids: list[str], top_k: int) -> dict:
    """Doc-id based retrieval metrics.

    Metrics are only meaningful if `relevant_doc_ids` is provided.
    """

    relevant = [rid for rid in relevant_doc_ids if isinstance(rid, str) and rid]
    retrieved = _retrieved_doc_ids(chunks)[: int(top_k)]

    if not relevant:
        return {
            "retrieval_hit": None,
            "recall_at_k": None,
            "mrr_at_k": None,
            "ndcg_at_k": None,
            "first_relevant_rank": None,
            "retrieved_doc_ids": retrieved,
        }

    relevant_set = set(relevant)

    # Deduplicate doc_ids in the retrieved list (multiple chunks from same doc).
    retrieved_unique: list[str] = []
    seen = set()
    for doc_id in retrieved:
        if doc_id not in seen:
            seen.add(doc_id)
            retrieved_unique.append(doc_id)

    hit = any(doc_id in relevant_set for doc_id in retrieved_unique)

    inter = [doc_id for doc_id in retrieved_unique if doc_id in relevant_set]
    recall_at_k = len(inter) / len(relevant_set)
    precision_at_k = len(inter) / max(len(retrieved_unique), 1)

    first_rank = None
    for i, doc_id in enumerate(retrieved_unique, 1):
        if doc_id in relevant_set:
            first_rank = i
            break

    mrr = 0.0 if first_rank is None else 1.0 / first_rank

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_unique, 1):
        if doc_id in relevant_set:
            dcg += 1.0 / (1.0 if i == 1 else math.log2(i + 1))

    ideal_hits = min(len(relevant_set), len(retrieved_unique))
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / (1.0 if i == 1 else math.log2(i + 1))

    ndcg = 0.0 if idcg == 0.0 else dcg / idcg

    return {
        "retrieval_hit": bool(hit),
        "recall_at_k": float(recall_at_k),
        "precision_at_k": float(precision_at_k),
        "mrr_at_k": float(mrr),
        "ndcg_at_k": float(ndcg),
        "first_relevant_rank": first_rank,
        "retrieved_doc_ids": retrieved,
        "retrieved_doc_ids_unique": retrieved_unique,
    }


def retrieval_chunk_metrics(chunks: list[dict], relevant_chunk_ids: list[str], top_k: int) -> dict:
    """Chunk-id based retrieval metrics (recommended for RAG)."""

    relevant = [rid for rid in relevant_chunk_ids if isinstance(rid, str) and rid]
    retrieved = _retrieved_chunk_ids(chunks)[: int(top_k)]

    if not relevant:
        return {
            "chunk_hit": None,
            "chunk_recall_at_k": None,
            "chunk_precision_at_k": None,
            "chunk_mrr_at_k": None,
            "chunk_ndcg_at_k": None,
            "chunk_first_relevant_rank": None,
            "retrieved_chunk_ids": retrieved,
        }

    relevant_set = set(relevant)

    # De-dupe chunk ids (usually unique, but keep it robust).
    retrieved_unique: list[str] = []
    seen = set()
    for cid in retrieved:
        if cid not in seen:
            seen.add(cid)
            retrieved_unique.append(cid)

    hit = any(cid in relevant_set for cid in retrieved_unique)

    inter = [cid for cid in retrieved_unique if cid in relevant_set]
    recall_at_k = len(inter) / len(relevant_set)
    precision_at_k = len(inter) / max(len(retrieved_unique), 1)

    first_rank = None
    for i, cid in enumerate(retrieved_unique, 1):
        if cid in relevant_set:
            first_rank = i
            break

    mrr = 0.0 if first_rank is None else 1.0 / first_rank

    dcg = 0.0
    for i, cid in enumerate(retrieved_unique, 1):
        if cid in relevant_set:
            dcg += 1.0 / (1.0 if i == 1 else math.log2(i + 1))

    ideal_hits = min(len(relevant_set), len(retrieved_unique))
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / (1.0 if i == 1 else math.log2(i + 1))

    ndcg = 0.0 if idcg == 0.0 else dcg / idcg

    return {
        "chunk_hit": bool(hit),
        "chunk_recall_at_k": float(recall_at_k),
        "chunk_precision_at_k": float(precision_at_k),
        "chunk_mrr_at_k": float(mrr),
        "chunk_ndcg_at_k": float(ndcg),
        "chunk_first_relevant_rank": first_rank,
        "retrieved_chunk_ids": retrieved,
        "retrieved_chunk_ids_unique": retrieved_unique,
    }


def retrieval_keyword_proxy(chunks: list[dict], expected_keywords: list[str], top_k: int) -> dict:
    """Fallback retrieval proxy when relevant_doc_ids aren't annotated yet."""

    keywords = [k for k in expected_keywords if isinstance(k, str) and k.strip()]
    if not keywords:
        return {
            "retrieved_keyword_hit": None,
            "retrieved_keyword_first_rank": None,
        }

    first_rank = None
    for i, c in enumerate(chunks[: int(top_k)], 1):
        if _keyword_in_chunk(c, keywords):
            first_rank = i
            break

    return {
        "retrieved_keyword_hit": first_rank is not None,
        "retrieved_keyword_first_rank": first_rank,
    }


def chunk_summaries(chunks: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for c in chunks:
        summaries.append({
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "title": c.get("title"),
            "section": c.get("section"),
            "url": c.get("url"),
            "score": c.get("score"),
            "pre_rerank_score": c.get("pre_rerank_score"),
            "rerank_score": c.get("rerank_score"),
            "vector_score": c.get("vector_score"),
            "bm25_score": c.get("bm25_score"),
        })
    return summaries

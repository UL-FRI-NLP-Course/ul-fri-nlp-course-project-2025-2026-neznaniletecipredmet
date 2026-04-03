"""
FAISS vector index: build, save, load, and search.
"""

import json
import logging

import faiss
import numpy as np

import config

log = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dim)
    return index


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_FILE))
    with open(config.FAISS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info("Saved index to %s", config.FAISS_INDEX_FILE)
    log.info("Saved metadata to %s", config.FAISS_META_FILE)


def load_index() -> tuple[faiss.Index, list[dict]]:
    if not config.FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {config.FAISS_INDEX_FILE}")
    if not config.FAISS_META_FILE.exists():
        raise FileNotFoundError(f"Metadata not found: {config.FAISS_META_FILE}")

    index = faiss.read_index(str(config.FAISS_INDEX_FILE))
    with open(config.FAISS_META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    log.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index, chunks


def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = config.TOP_K,
) -> list[dict]:
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        result = dict(chunks[idx])
        result["score"] = float(score)
        results.append(result)

    return results

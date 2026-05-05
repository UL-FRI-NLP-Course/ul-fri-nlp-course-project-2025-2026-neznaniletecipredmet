"""
FAISS vector index: build, save, load, and search.
"""

import json
import logging

import faiss
import numpy as np

import config

log = logging.getLogger(__name__)

def _to_gpu(index: faiss.Index, gpu_id: int = 0) -> faiss.Index:
    ngpu = faiss.get_num_gpus()
    if ngpu == 0:
        log.warning("No GPUs found, using CPU")
        return index

    res = faiss.StandardGpuResources()

    # (Iztok) to bo prestavilo indeks samo na eno GPE, kasneje lahko
    # probamo porazdelit z index_cpu_to_all_gpus
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    log.info("Moved FAISS index to GPU %d", gpu_id)
    return gpu_index


def build_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dim)

    if use_gpu:
        index = _to_gpu(index)

    return index


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # convert the index into a CPU index to allow serializing
    # as direct saving of an index from the GPU is not supported
    cpu_index = faiss.index_gpu_to_cpu(index) if hasattr(index, "getDevice") else index

    faiss.write_index(cpu_index, str(config.FAISS_INDEX_FILE))
    with open(config.FAISS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info("Saved index to %s", config.FAISS_INDEX_FILE)
    log.info("Saved metadata to %s", config.FAISS_META_FILE)


def load_index(use_gpu: bool) -> tuple[faiss.Index, list[dict]]:
    if not config.FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {config.FAISS_INDEX_FILE}")
    if not config.FAISS_META_FILE.exists():
        raise FileNotFoundError(f"Metadata not found: {config.FAISS_META_FILE}")

    index = faiss.read_index(str(config.FAISS_INDEX_FILE))

    if use_gpu:
        index = _to_gpu(index)

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

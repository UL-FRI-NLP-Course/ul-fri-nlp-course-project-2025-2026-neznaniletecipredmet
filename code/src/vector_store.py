"""
FAISS vector index: build, save, load, and search.
"""

import json
import logging
import os

import faiss
import numpy as np

import config

log = logging.getLogger(__name__)


def _index_info_file_path():
    return config.INDEX_DIR / "embedding_info.json"

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

    cpu_index = faiss.index_gpu_to_cpu(index) if hasattr(index, "getDevice") else index

    faiss.write_index(cpu_index, str(config.FAISS_INDEX_FILE))
    with open(config.FAISS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    info = {
        "embedding_model": config.EMBEDDING_MODEL,
        "dim": int(cpu_index.d),
        "ntotal": int(cpu_index.ntotal),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }
    with open(_index_info_file_path(), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    log.info("Saved index to %s", config.FAISS_INDEX_FILE)
    log.info("Saved metadata to %s", config.FAISS_META_FILE)
    log.info("Saved index info to %s", _index_info_file_path())


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

    info_path = _index_info_file_path()
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            stored_model = info.get("embedding_model")
            stored_chunk_size = info.get("chunk_size")
            stored_chunk_overlap = info.get("chunk_overlap")
            mismatches = []
            if stored_model and stored_model != config.EMBEDDING_MODEL:
                mismatches.append(
                    f"embedding_model mismatch: index built with '{stored_model}', "
                    f"current config.EMBEDDING_MODEL='{config.EMBEDDING_MODEL}'"
                )
            if stored_chunk_size is not None and stored_chunk_size != config.CHUNK_SIZE:
                mismatches.append(
                    f"chunk_size mismatch: index built with {stored_chunk_size}, "
                    f"current config.CHUNK_SIZE={config.CHUNK_SIZE}"
                )
            if stored_chunk_overlap is not None and stored_chunk_overlap != config.CHUNK_OVERLAP:
                mismatches.append(
                    f"chunk_overlap mismatch: index built with {stored_chunk_overlap}, "
                    f"current config.CHUNK_OVERLAP={config.CHUNK_OVERLAP}"
                )
            if mismatches:
                msg = "Index/config mismatch detected:\n  " + "\n  ".join(mismatches)
                if os.environ.get("STRICT_INDEX_VALIDATION", "").strip() in ("1", "true", "True"):
                    raise RuntimeError(msg)
                log.warning(msg)
                log.warning(
                    "Continuing despite mismatch (set STRICT_INDEX_VALIDATION=1 to make this fatal). "
                    "If you see retrieval garbage, set EMBEDDING_MODEL to the value above and retry."
                )
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Could not read embedding_info.json (%s); skipping safety check", e)
    else:
        log.warning(
            "No embedding_info.json found next to %s. This index was built before "
            "embedding-info tracking was added; cannot verify embedding model match.",
            config.FAISS_INDEX_FILE,
        )

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

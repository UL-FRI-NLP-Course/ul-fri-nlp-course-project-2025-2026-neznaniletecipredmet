"""
Embedding generation using sentence-transformers.
Handles E5-style query/passage prefixes automatically.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger(__name__)

_model_instance: SentenceTransformer | None = None
_loaded_model_name: str | None = None


def _is_e5_model(model_name: str) -> bool:
    return "multilingual-e5" in model_name or model_name.startswith("intfloat/e5")


def get_model(model_name: str = config.EMBEDDING_MODEL) -> SentenceTransformer:
    global _model_instance, _loaded_model_name
    if _model_instance is None or _loaded_model_name != model_name:
        log.info("Loading embedding model: %s", model_name)
        _model_instance = SentenceTransformer(model_name)
        _loaded_model_name = model_name
        log.info("Embedding model loaded")
    return _model_instance


def embed_queries(
    texts: list[str],
    model_name: str = config.EMBEDDING_MODEL,
) -> np.ndarray:
    model = get_model(model_name)
    if _is_e5_model(model_name):
        texts = [f"query: {t}" for t in texts]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def embed_passages(
    texts: list[str],
    model_name: str = config.EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    model = get_model(model_name)
    if _is_e5_model(model_name):
        texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_chunks(chunks: list[dict], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks with model %s", len(texts), model_name)
    embeddings = embed_passages(texts, model_name=model_name)
    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings

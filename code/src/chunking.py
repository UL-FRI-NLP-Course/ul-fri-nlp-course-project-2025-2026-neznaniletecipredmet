"""
Text chunking with section-aware sliding window.
Chunk size and overlap are measured in whitespace-split tokens.
"""

import logging

import config
from src.utils import is_noise, make_chunk_id

log = logging.getLogger(__name__)


def _split_tokens(text: str) -> list[str]:
    return text.split()


def _chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    tokens = _split_tokens(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    doc_id = doc["doc_id"]
    title = doc.get("title", "")
    source_path = doc.get("source_path", "")
    url = doc.get("url", "")
    language = doc.get("language", "sl")
    metadata = doc.get("metadata", {})
    sections = doc.get("sections", [])

    if not sections:
        sections = [{"section": "main", "text": doc.get("text", "")}]

    chunks = []
    chunk_index = 0

    for section_entry in sections:
        section_name = section_entry.get("section", "main")
        text = section_entry.get("text", "").strip()

        if not text or is_noise(text):
            continue

        section_prefix = f"{title} - {section_name}" if section_name != "main" else title

        for raw_chunk in _chunk_text(text):
            if is_noise(raw_chunk):
                continue

            contextualized = f"{section_prefix}\n\n{raw_chunk}" if section_prefix else raw_chunk

            chunks.append({
                "chunk_id": make_chunk_id(doc_id, chunk_index),
                "doc_id": doc_id,
                "text": contextualized,
                "title": title,
                "section": section_name,
                "source_path": source_path,
                "url": url,
                "language": language,
                "metadata": metadata,
            })
            chunk_index += 1

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        log.info("Chunked '%s' -> %d chunks", doc.get("title", doc["doc_id"]), len(doc_chunks))

    log.info("Total chunks: %d from %d documents", len(all_chunks), len(documents))
    return all_chunks

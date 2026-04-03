"""
Parse all documents in data/raw/ and build the FAISS vector index.
Run this once before starting the app or evaluation.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.chunking import chunk_documents
from src.embeddings import embed_chunks
from src.parse_docs import parse_directory
from src.utils import write_jsonl
from src.vector_store import build_index, save_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=== Step 1: Parsing documents from %s ===", config.RAW_DIR)
    documents = parse_directory(config.RAW_DIR)

    if not documents:
        log.error("No documents parsed. Run scripts/collect_data.py first.")
        sys.exit(1)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(documents, config.PARSED_JSONL)
    log.info("Saved %d parsed documents to %s", len(documents), config.PARSED_JSONL)

    log.info("=== Step 2: Chunking documents ===")
    chunks = chunk_documents(documents)

    if not chunks:
        log.error("No chunks produced. Check document content.")
        sys.exit(1)

    write_jsonl(chunks, config.CHUNKS_JSONL)
    log.info("Saved %d chunks to %s", len(chunks), config.CHUNKS_JSONL)

    log.info("=== Step 3: Embedding chunks ===")
    embeddings = embed_chunks(chunks)

    log.info("=== Step 4: Building and saving FAISS index ===")
    index = build_index(embeddings)
    save_index(index, chunks)

    log.info("=== Done. Index ready at %s ===", config.INDEX_DIR)


if __name__ == "__main__":
    main()

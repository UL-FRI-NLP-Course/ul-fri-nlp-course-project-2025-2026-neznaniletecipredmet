"""Build a vector index for a specific run.

Parses documents from '/d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/raw/' (plus optional manual files
from 'raw_dataset/files/'), chunks them, embeds them, and saves a FAISS index
under '/d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/index/'.
"""

import argparse
import logging
import shutil
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run/dataset name (stored under /d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<name>/)")
    parser.add_argument(
        "--mode",
        choices=["new", "update"],
        default=None,
        help=(
            "Index mode. If omitted: uses 'update' when an index already exists for the run, "
            "otherwise uses 'new'.\n"
            "- new: require that /d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/index/index.faiss does not exist\n"
            "- update: require that /d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/index/index.faiss exists"
        ),
    )
    args = parser.parse_args()

    name = (args.run or config.DEFAULT_RUN_NAME).strip() or config.DEFAULT_RUN_NAME
    run_root = config.RUNS_DIR / name
    index_file = run_root / "index" / "index.faiss"

    mode = args.mode
    if mode is None:
        mode = "update" if index_file.exists() else "new"
        log.info("Mode not provided; inferred mode='%s' for run '%s'", mode, name)

    if mode == "update" and not index_file.exists():
        log.error("No existing index found for run '%s' (missing %s). Use --mode new.", name, index_file)
        sys.exit(2)

    if mode == "new" and index_file.exists():
        log.error("Index already exists for run '%s' (%s). Use --mode update.", name, index_file)
        sys.exit(2)

    # For indexing we always (re)use the run folder; mode only controls whether
    # we expect an index to already exist.
    config.apply_run(name, mode="update")

    # Snapshot image description sidecars (manual inputs) into the run folder.
    # This makes the run self-describing even if raw_dataset/ changes later.
    try:
        base = config.RAW_DATASET_FILES_DIR
        if base.exists():
            for img in base.rglob("*"):
                if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                sidecar_txt = img.with_suffix(".txt")
                sidecar_md = img.with_suffix(".md")
                sidecar = sidecar_txt if sidecar_txt.exists() else sidecar_md if sidecar_md.exists() else None
                if sidecar is None:
                    continue

                rel = sidecar.relative_to(base)
                dest = config.IMAGE_DESCRIPTIONS_SNAPSHOT_DIR / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sidecar, dest)
    except Exception as e:
        log.warning("Could not snapshot image descriptions: %s", e)

    input_dirs = [config.RAW_DIR]
    for d in getattr(config, "EXTRA_RAW_INPUT_DIRS", []):
        if d and Path(d).exists():
            input_dirs.append(Path(d))

    log.info("=== Step 1: Parsing documents from %d input dir(s) ===", len(input_dirs))

    documents: list[dict] = []
    for d in input_dirs:
        log.info("Parsing: %s", d)
        documents.extend(parse_directory(d))

    if not documents:
        log.error("No documents parsed. Run scripts/collect_data.py and/or add files under raw_dataset/files/.")
        sys.exit(1)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(documents, config.PARSED_JSONL)
    log.info("Saved %d parsed documents to %s", len(documents), config.PARSED_JSONL)

    # Export a compact list of sources used for this run (URLs, local paths, hashes).
    sources: list[dict] = []
    for d in documents:
        meta = d.get("metadata", {}) or {}
        sources.append({
            "doc_id": d.get("doc_id"),
            "title": d.get("title"),
            "url": d.get("url"),
            "source_path": d.get("source_path"),
            "file_type": meta.get("file_type"),
            "sha256": meta.get("sha256"),
            "saved_at": meta.get("saved_at"),
            "downloaded_from": meta.get("downloaded_from"),
        })
    write_jsonl(sources, config.SOURCES_JSONL)
    log.info("Saved %d source record(s) to %s", len(sources), config.SOURCES_JSONL)

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

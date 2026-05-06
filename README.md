# NLP course project: Chatbot for UL FRI students

This repository contains a small pipeline for collecting publicly available UL/FRI web content, building a searchable vector index, and testing retrieval (RAG-ready).

## Data

We use publicly available data from official Fakulteta za racunalnistvo in informatiko (FRI) and Univerza v Ljubljani (UL) websites.
Seed links are tracked in [raw_dataset/data_links.txt](raw_dataset/data_links.txt).

## Quickstart

See [code/QUICKSTART.md](code/QUICKSTART.md) for the exact commands.

You can run commands either from inside `code/` (using `scripts/...`) or from the repository root using the wrapper scripts (`collect_data.py`, `build_index.py`, `test_retrieval.py`).

High-level flow:

1) Collect data (crawl + download):

	- Script: `code/scripts/collect_data.py`
	- Output: `/d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/raw/` and crawl manifests

2) Build index (parse -> chunk -> embed -> save FAISS):

	- Script: `code/scripts/build_index.py`
	- Output: `/d/hpc/projects/onj_fri/neznani-leteci-predmet/data/runs/<run>/index/` (FAISS + metadata) and `processed/` JSONL

3) Test retrieval (no LLM required):

	- Script: `code/scripts/test_retrieval.py`
	- Uses the run-specific index and prints top chunks

## Notes

- Some PDF parsing paths can be memory-intensive (especially layout/table extraction). If you hit memory errors on Windows, consider increasing the pagefile or running indexing on ARNES.
- OCR: scanned PDFs can be OCR'd during indexing when PDF parsing uses Docling and a working OCR backend is available. Standalone images are indexed via sidecar text files.
- Cluster usage: SLURM helper scripts are in `code/slurm/` (e.g. `collect_data.sh`, `build_index.sh`, `evaluate.sh`).
- Future work: ranking (e.g., BM25 or rerankers) and an LLM wrapper can be added on top of the current retrieval outputs.

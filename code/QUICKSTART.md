# Quickstart — Data Collection & Index Building

## Setup

```bash
conda create -n nlp-rag python=3.11 -y
conda activate nlp-rag
cd code
pip install -r requirements.txt
```

## Step 1 — Collect data

```bash
python scripts/collect_data.py
```

Downloads HTML pages and PDFs from URLs listed in `../raw_dataset/data_links.txt` into `data/raw/`.

## Step 2 — Build index

```bash
python scripts/build_index.py
```

Parses documents → chunks them → embeds with `intfloat/multilingual-e5-base` → saves FAISS index to `data/index/`.

## Step 3 — Test retrieval

```bash
python scripts/test_retrieval.py
```

Runs a few test queries against the index and prints the top retrieved chunks. No LLM needed.

## Config

All settings (chunk size, embedding model, top-k, ...) are in `config.py`.

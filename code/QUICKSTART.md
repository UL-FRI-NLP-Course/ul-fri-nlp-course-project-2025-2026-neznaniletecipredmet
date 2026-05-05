# Quickstart - Data Collection & Index Building

## Setup

Option A: conda

```bash
conda create -n nlp-rag python=3.11 -y
conda activate nlp-rag
cd code
pip install -r requirements.txt
```

Option B: venv (PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd code
python -m pip install -r requirements.txt
```



### Optional: OCR for scanned PDFs (Docling)

If your PDFs are scanned (no embedded text), indexing can still extract text via OCR when
PDF parsing uses Docling and an OCR backend is available (otherwise a less cpu friendly OCR is used).

This project prefers Tesseract on CPU only machines, and lets Docling auto pick the best available OCR backend when a CUDA GPU is available.

(Preferable) For a lightweight, CPU-only setup on Windows, install Tesseract OCR:

1) Install Tesseract
	- Option A (if you use winget):
	  - `winget install -e --id UB-Mannheim.TesseractOCR`
	- Option B: install the "Tesseract OCR" Windows build from UB Mannheim
	  and ensure `tesseract.exe` is on your PATH.

2) Install language data for Slovenian
	- Make sure `slv` is available (and optionally `eng`).

3) Verify from PowerShell:
	- `tesseract --version`
	- `tesseract --list-langs`

If OCR is not working during indexing, it usually means either Tesseract is not installed,
or the Slovenian language data is missing.

All commands below assume you're running them from inside `code/`.

There are also wrapper functions with the same names available, so you can run from `root`:
- `collect_data.py`
- `build_index.py`
- `test_retrieval.py`
- `evaluate.py`


## Step 1 - Collect data

### Run naming convention

Runs are stored under `code/data/runs/<run_name>/...`.

Use a run name that makes it obvious what you crawled and with what settings. Like for example:

`YYYY-MM-DD__<scope>__fri10_ul3__vN`

Examples:
- `2026-04-12__seedlinks-v1__fri10_ul3__v1`
- `2026-04-20__more-faq-pages__fri10_ul3__v2`

Where:
- `fri10` means FRI pages are crawled up to depth 10
- `ul3` means other `*.uni-lj.si` pages are crawled up to depth 3
- `vN` crawl depth on other websites

```bash
python scripts/collect_data.py --run 2026-04-12__seedlinks-v1__fri10_ul3__v1 --mode new
```

Downloads HTML pages, PDFs, and DOCXs from URLs listed in `../raw_dataset/data_links.txt` into `code/data/runs/<run_name>/raw/`.

Optional: override crawl depths per run:

```bash
python scripts/collect_data.py --run 2026-04-12__seedlinks-v1__fri6_ul2__v1 --mode new --depth-fri 6 --depth-ul 2 --depth-v 1
```

To update an existing run (re-crawl and add any new pages/PDFs), reuse the same name:

```bash
python scripts/collect_data.py --run 2026-04-12__seedlinks-v1__fri10_ul3__v1 --mode update
```

Optional: you can also place manually added PDFs / images under `../raw_dataset/files/`.
Those files will be parsed during index building.

## Step 2 - Build index

```bash
python scripts/build_index.py --run 2026-04-12__seedlinks-v1__fri10_ul3__v1
```

Parses documents -> chunks them -> embeds with `intfloat/multilingual-e5-base` -> saves FAISS index to:

- `code/data/runs/<run_name>/index/index.faiss`
- `code/data/runs/<run_name>/index/metadata.json`

Note: the crawler writes `code/data/runs/<run_name>/raw/manifest.jsonl` so parsed documents keep the original URL and crawl timestamp.

## Step 3 - Test retrieval

```bash
python scripts/test_retrieval.py --run 2026-04-12__seedlinks-v1__fri10_ul3__v1
```

Runs a few test queries against the index and prints the top retrieved chunks.

Optional flags:
- `--hybrid` for BM25+dense hybrid
- `--rerank` (plus `--rerank-candidate-k`) to rerank candidates with a cross-encoder

## Step 4 — Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select the run in the sidebar, then ask questions in Slovenian or English.

By default runs in **retrieval-only mode** (no GPU needed) — shows retrieved chunks without generating an answer.
To get a full LLM answer, uncheck "Samo iskanje" in the sidebar and pick a model (small models like `Qwen/Qwen2.5-1.5B-Instruct` work locally; GaMS/Llama/Mistral require a GPU).

Optional: enable **Rerank (cross-encoder)** in the sidebar to reorder retrieved chunks.

## Step 5 — Evaluate

Place evaluation questions in `code/data/runs/<run_name>/eval/questions.jsonl`:

```json
{"question": "Koliko krat lahko opravljam izpit?", "language": "sl", "expected_keywords": ["trikrat"], "relevant_doc_ids": []}
```

Then run:

```bash
python scripts/evaluate.py --run <run_name> --model cjvt/GaMS3-12B-Instruct
```

Retrieval-only (no LLM):

```bash
python scripts/evaluate.py --run <run_name> --retrieval-only
```

With reranking:

```bash
python scripts/evaluate.py --run <run_name> --retrieval-only --rerank --rerank-candidate-k 20
```

### Fast annotation of `relevant_doc_ids`

To get real IR metrics (recall/MRR/nDCG), you need to fill `relevant_doc_ids` in the eval file.

Use the interactive helper (shows retrieved chunks and lets you pick the relevant ones). By default it stores both `relevant_doc_ids` and `relevant_chunk_ids`:

```bash
python scripts/annotate_eval.py --run <run_name> --top-k 10 --hybrid
```

To store only document ids:

```bash
python scripts/annotate_eval.py --run <run_name> --top-k 10 --hybrid --doc-only
```

You can also run it from repo root:

```bash
python annotate_eval.py --run <run_name> --top-k 10 --hybrid
```

Results are saved to `code/data/runs/<run_name>/eval/results_<model>.jsonl`.

On ARNES HPC (fill in the placeholders in `slurm/*.sh` first):

```bash
sbatch --export=RUN_NAME=<run_name>,MODEL_NAME=cjvt/GaMS3-12B-Instruct slurm/evaluate.sh
sbatch --export=RUN_NAME=<run_name> slurm/compare_models.sh
```

## Config

All settings (chunk size, embedding model, top-k, ...) are in `config.py`.

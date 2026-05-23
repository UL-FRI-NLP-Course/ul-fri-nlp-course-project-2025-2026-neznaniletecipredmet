# FRI Student Chatbot — RAG Pipeline

A retrieval-augmented chatbot that answers UL FRI administrative questions in Slovenian and English. It crawls official UL/FRI pages, chunks and embeds the text, stores everything in a FAISS index, and at query time uses a hybrid BM25 + dense retriever, a cross-encoder reranker, and an instruction-tuned LLM to generate an answer grounded in the retrieved passages.

This is the **`eval-pipeline`** branch — the final submission. The default branch contains earlier exploratory work.

---

## Run it yourself

You do not need to re-crawl or re-embed anything. Download the prebuilt index from Google Drive and point the retrieval script at it.

**Step 1 — Clone and set up**

```bash
git clone -b eval-pipeline https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet
cd ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet

conda create -n fri-rag python=3.11 -y && conda activate fri-rag
pip install -r code/requirements.txt
```

**Step 2 — Download the prebuilt index**

Download the two files from Google Drive:

📁 **[Google Drive — prebuilt index](https://drive.google.com/drive/folders/1FOESoezBwMJ8q9DR3cYrYjHSmsVSoAsM?usp=sharing)**

The folder contains `index.faiss` and `metadata.json`. Place them here:

```
code/data/runs/2026-05-20__full__fri4_ul2__v1/index/index.faiss
code/data/runs/2026-05-20__full__fri4_ul2__v1/index/metadata.json
```

Create the directory if it does not exist yet:

```bash
mkdir -p code/data/runs/2026-05-20__full__fri4_ul2__v1/index
```

**Step 3 — Ask a question**

```bash
python test_retrieval.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --query "Kdaj so uradne ure študentskega referata?"
```

The script retrieves the top passages and prints them. To get a full generated answer add `--generate` and set up your `HF_TOKEN` (see API keys below).

---

## Repository layout

```
.
├── README.md
├── annotate_eval.py          # thin wrappers around code/scripts/ entry points
├── build_index.py
├── collect_data.py
├── evaluate.py
├── test_retrieval.py
├── code/
│   ├── QUICKSTART.md         # step-by-step guide for the full pipeline
│   ├── config.py             # all paths, model names, chunk sizes, etc.
│   ├── requirements.txt
│   ├── questions_full_v2.json           # evaluation set (51 in-scope + 10 negatives)
│   ├── questions_audit_triage.md        # per-question rationale for reference rewrites
│   ├── data/runs/
│   │   └── 2026-05-20__full__fri4_ul2__v1/     # the winning full-crawl run
│   │       ├── index/        # ← put the Google Drive files here
│   │       ├── processed/    # chunks.jsonl, sources.jsonl
│   │       └── eval/         # SUMMARY.md, comparison tables, raw judged JSONLs
│   ├── scripts/              # one script per pipeline stage (see below)
│   ├── slurm/                # SLURM launchers for the ARNES cluster
│   ├── src/                  # library code (chunking, embedding, retrieval, …)
│   └── tests/
├── raw_dataset/
│   ├── data_links.txt        # seed URLs for the crawler
│   └── files/                # hand-curated PDFs
└── report/
    ├── report.tex / report.pdf
    └── fig/
```

## What ships with the repo (and what doesn't)

| | What | Notes |
|---|---|---|
| ✅ | Prebuilt FAISS index + metadata | **download from Google Drive above** (too large for git) |
| ✅ | `code/questions_full_v2.json` | Sonnet-audited evaluation set, the gold used in the report |
| ✅ | 25-question dev set with `relevant_doc_ids` | source of the document-level retrieval metrics |
| ✅ | All eval JSONLs and aggregation tables | every number in the report traces back to these |
| ✅ | All scripts (crawler, parser, indexer, retrieval, judge, audit) | full pipeline in one CLI per stage |
| ❌ | `parsed.jsonl` (~114 MB) and raw HTML/PDF (`raw/`, ~1 GB) | git-ignored; regenerate with `collect_data.py` + `build_index.py` |
| ❌ | Model weights | downloaded on demand from Hugging Face Hub |

---

## API keys

| Variable | Used by | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM-as-judge, reference audit, question generation | <https://console.anthropic.com/> |
| `HF_TOKEN` | Llama, Mistral, GaMS (gated models on HF Hub) | <https://huggingface.co/settings/tokens> |

Export as environment variables or put them in a `.env` file at the repo root — `python-dotenv` is loaded automatically. Plain retrieval (no generation, no judge) needs neither key.

---

## Models used

No model is trained from scratch. Everything is downloaded at run time from Hugging Face Hub or called via the Anthropic API.

| Role | Model |
|---|---|
| Embedder (final config) | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) |
| Embedder (sweep) | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base), [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) |
| Cross-encoder reranker | [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) |
| Generator (local smoke-test only, not evaluated) | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Generator (cluster, evaluated in report) | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), [`cjvt/GaMS3-12B-Instruct`](https://huggingface.co/cjvt/GaMS3-12B-Instruct) |
| LLM judge (sweeps) | `claude-haiku-4-5` (Anthropic API) |
| LLM judge (final + audit) | `claude-sonnet-4-6` (Anthropic API) |

Llama and Mistral are gated — you need to accept their licences on HF Hub before your token will work.

---

## Scripts

All scripts live under `code/scripts/` and are also accessible via the top-level wrappers (`python build_index.py`, etc.).

### Building the index

| Script | What it does |
|---|---|
| `collect_data.py` | Crawls the seed URLs (sitemaps + RSS, dedup by SHA-256), writes `raw/` + manifest |
| `build_index.py` | Parses raw → chunks → embeds → FAISS index (`DISABLE_OCR=1` skips OCR for a ~16× speedup) |
| `test_retrieval.py` | Smoke-tests retrieval against a saved index |
| `evaluate.py` | Generates answers with a chosen LLM and writes `eval/results_*.jsonl` |

### Evaluation

| Script | What it does |
|---|---|
| `generate_questions.py` | Offline question generation with Claude |
| `audit_references.py` | Sonnet-graded fact-check of reference answers against the corpus |
| `apply_reference_audit.py` | Applies the fixed audit policy to produce `questions_full_v2.json` |
| `judge_answers.py` | RAGAS-style LLM judge (faithfulness, hallucination, refusal, overall) |
| `compare_embedders.py` | Embedder × retrieval-mode sweep |
| `compare_chunk_sizes.py` | Chunk-size sweep on the winning configuration |
| `compare_generators.py` | Per-generator judging of cluster results |

### Cluster handoff

| Script | What it does |
|---|---|
| `prepare_cluster_questions.py` | Converts `questions_full_v2.json` to the JSONL format used on the cluster |
| `sync_to_cluster.sh` | rsync the run + index to the cluster |
| `sync_from_cluster.sh` | rsync the result files back |
| `code/slurm/` | SLURM job scripts (assume ARNES `onj_fri` partition, CUDA 12.2, Python 3.11) |

---

## Reproducing the report numbers

**Final operating configuration** (selected by human evaluation): `bge-m3` + 400-token chunks + hybrid (BM25 + dense) + cross-encoder rerank + top-k=6 + rerank-candidate-k=30.

**Automated-sweep winner** (LLM-as-judge): `multilingual-e5-base` + 400-token chunks + hybrid + cross-encoder rerank + top-k=4. The commands below reproduce the automated-sweep numbers using this configuration; the human evaluation scores (Table 3 in the report) were produced by human raters and are not re-runnable via script.

**Reference-grounding ceiling (Sonnet):**

```bash
cd code
python scripts/judge_answers.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --from-questions questions_full_v2.json \
    --retrieval-top-k 4 --hybrid --rerank --rerank-candidate-k 20 \
    --top-k-context 4 \
    --model claude-sonnet-4-6 --temperature 0.0 \
    --out judged_reference_sonnet_n61.jsonl
```

**Generator comparison (Sonnet on cluster outputs):**

```bash
python scripts/compare_generators.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --reference-file questions_full_v2.json \
    --judge-model claude-sonnet-4-6 \
    --out-md model_comparison_sonnet.md
```

Pre-run results are already committed:

- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/eval_matrix.md` — embedder × retrieval mode
- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/chunk_size_comparison.md` — chunk-size sweep
- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/model_comparison_sonnet.md` — generators + ceiling
- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/SUMMARY.md` — top-line summary

For the complete step-by-step walkthrough see [`code/QUICKSTART.md`](code/QUICKSTART.md).

Alternative — `ask.py` (ad-hoc queries)
------------------------------------

Use `ask.py` for quick, interactive exploration of a prebuilt index or for single ad-hoc queries from your laptop (retrieval-only by default). It's handy when you want to inspect retrieved passages or generate a single answer without running the full eval pipeline.

- **When to use:** troubleshooting, debugging retrieval quality, spot-checking answers, or trying a model interactively.
- **Default behavior:** retrieval-only (no GPU required). To generate an answer, pass `--model <hf-model>` and provide a valid `HF_TOKEN`.
- **Index location:** by default `ask.py` looks for index files in `code/data/runs/2026-05-20__full__fri4_ul2__v1/index`. Use `--index-dir` to point elsewhere.

Examples:

```bash
# Retrieval-only (no GPU needed)
python code/scripts/ask.py --question "Kdaj so uradne ure študentskega referata?"

# Generate an answer with a HF model (downloads model on first run)
python code/scripts/ask.py \
    --question "Kdaj so uradne ure študentskega referata?" \
    --model meta-llama/Llama-3.1-8B-Instruct

# Point to a custom index directory
python code/scripts/ask.py --index-dir /path/to/index --question "..."
```

Note: for batch evaluation or reproducible runs use `test_retrieval.py` / `evaluate.py` and the `code/scripts/prepare_cluster_questions.py` workflow instead of `ask.py`.

On the cluster (SLURM)
----------------------

You can run `ask.py` on the ARNES cluster via the provided SLURM launcher `code/slurm/ask_on_cluster.sh`. This is useful to run a single non-interactive query using the cluster's GPUs (or for heavier generator models).

Examples (submit from the repo root):

```bash
# Retrieval-only (no model)
QUESTION="Kdaj so uradne ure študentskega referata?" \
    sbatch code/slurm/ask_on_cluster.sh

# Generate with a HF model (GPU required)
QUESTION="Kdaj so uradne ure študentskega referata?" \
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
HF_TOKEN="$HF_TOKEN" \
    sbatch code/slurm/ask_on_cluster.sh

# Use a different prebuilt index run
INDEX_RUN="2026-05-20__full__fri4_ul2__v1" \
QUESTION="..." \
    sbatch code/slurm/ask_on_cluster.sh
```

Notes:
- Set `QUESTION` (required) — the script runs non-interactively and reads the question from the environment.
- Optionally set `MODEL_NAME` to run generation on a GPU; otherwise the job will run retrieval-only.
- You can override the index location via `INDEX_DIR` or point to another run via `INDEX_RUN`.
- Provide `HF_TOKEN` in the environment for gated Hugging Face models.

---

## Data

The corpus is publicly reachable UL/FRI web content. Seed URLs are in [`raw_dataset/data_links.txt`](raw_dataset/data_links.txt). The crawler respects `robots.txt`, deduplicates by SHA-256, and filters to Slovenian and English content.

OCR with Docling + Tesseract takes about 4 hours for ~500 PDFs. Set `DISABLE_OCR=1` to use the fast PyMuPDF-only path (~14 minutes) if you don't have scanned documents in your corpus.

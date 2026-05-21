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

On ARNES HPC:

```bash
sbatch --export=RUN_NAME=2026-04-12__seedlinks-v1__fri10_ul3__v1,MODE=new slurm/collect_data.sh
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

### Retrieval sweep (4 configs, local CPU)

```bash
python scripts/run_retrieval_sweep.py --run <run_name> --top-k 4 --rerank-candidate-k 20
```

Runs `dense`, `hybrid`, `dense_rerank`, `hybrid_rerank` and writes a comparison table at `code/data/runs/<run_name>/eval/retrieval_comparison.md`.

## Step 6 — Grow the eval set with Claude

Set your Anthropic key (`export ANTHROPIC_API_KEY=...` or add to `.env`), then:

```bash
python scripts/generate_questions.py --run <run_name> --num-chunks 80 --questions-per-chunk 2 --negatives 20
python scripts/merge_eval_questions.py --run <run_name> --src questions_generated.jsonl
python scripts/bootstrap_eval_questions.py --run <run_name> --hybrid --rerank --rerank-candidate-k 20 --no-append-questions
python scripts/annotate_eval.py --run <run_name> --top-k 8 --hybrid --only-missing  # spot-check
```

Both `JUDGE_MODEL` and `EVAL_QUESTION_GEN_MODEL` default to `claude-sonnet-4-6`; override via env vars or `--model`.

## Step 7 — LLM-as-judge (Claude)

After running `evaluate.py` with a generation model:

```bash
python scripts/judge_answers.py --run <run_name> --results results_Qwen_Qwen2.5-1.5B-Instruct.jsonl
```

Optionally pass `--reference-file ../code/questions_full.json` to give Claude the human-curated reference answers, and `--limit N` for a smoke test.

### Calibration: judge the reference answers themselves

To validate that the judge gives high scores to known-good answers (and to use the answers in `questions_full.json` without first running a generation model), use `--from-questions`. The script will run retrieval on the fly for each question and judge the `reference_answer` as if it were a model answer:

```bash
python scripts/judge_answers.py --run <run_name> \
    --from-questions ../code/questions_full.json --hybrid --rerank --rerank-candidate-k 20
```

Output goes to `<run>/eval/judged_reference_questions_full.jsonl`. Faithfulness/answer-relevance should be near 1.0 — if not, the rubric or retrieval needs work.

## Step 8 — Manual sanity check

```bash
python scripts/manual_review.py --run <run_name> --judged judged_results_Qwen_Qwen2.5-1.5B-Instruct.jsonl --n 25 --stratify
python scripts/manual_review.py --run <run_name> --judged judged_results_Qwen_Qwen2.5-1.5B-Instruct.jsonl --report-only
```

The first command iterates judged rows and lets you re-rate them; the second just prints the agreement statistics.

## Step 9 — Reference audit (clean the gold labels)

The judge-on-reference results revealed two distinct failure modes for in-scope questions: (a) retrieval ranking misses, and (b) reference answers that contain claims not present in the corpus. Step 9 fixes (b) by using Claude (Sonnet) as a strict fact-checker against a broad retrieval pool, then producing a cleaned eval set.

```bash
python scripts/audit_references.py \
    --run <run_name> \
    --questions ../code/questions_full.json \
    --top-k 30 --hybrid \
    --out ../code/questions_audited.json

python scripts/apply_reference_audit.py \
    --audited ../code/questions_audited.json \
    --out ../code/questions_full_v2.json
```

Outputs:
- `questions_audited.json` — every question with a verdict (`supported`, `partial`, `unsupported`), supported/unsupported claim lists, and an optional grounded rewrite.
- `questions_full_v2.json` — cleaned eval set: rewrites applied where Claude could ground them, references dropped where it couldn't. Use this for every downstream sweep.
- `questions_full_v2_triage.md` — markdown list of questions flagged `needs_review` for an optional human pass.

The audit uses Sonnet by default (the quality bottleneck for everything downstream). Override with `--model`.

## Step 10 — Embedder × retrieval-mode matrix

For each (embedder, mode) combination, this script symlinks `raw/` from a source crawl run, sets `EMBEDDING_MODEL`, rebuilds the index, runs `judge_answers.py --from-questions`, and aggregates everything into a single markdown table. The judge uses Haiku by default for cost; the *winning* configuration is then re-judged with Sonnet for the report headline number.

```bash
python scripts/compare_embedders.py \
    --source-run <run_name> \
    --questions ../code/questions_full_v2.json \
    --top-k 4 \
    --embedders intfloat/multilingual-e5-base,intfloat/multilingual-e5-large \
    --modes dense,hybrid,hybrid_rerank \
    --judge-model claude-haiku-4-5
```

Each derived run is named `<source>__<embedder_tag>` and gets its own `index/`, `processed/`, and `eval/`. The aggregated table is written to `code/data/runs/<source>/eval/eval_matrix.md` together with the recommended re-judge command for the winner.

After the sweep finishes, re-judge the winner with Sonnet:

```bash
EMBEDDING_MODEL=<winning_embedder> python scripts/judge_answers.py \
    --run <source>__<winning_tag> \
    --from-questions ../code/questions_full_v2.json \
    --retrieval-top-k 4 [--hybrid] [--rerank] \
    --model claude-sonnet-4-6 \
    --out judged_winner_sonnet.jsonl
```

## Step 11 — Chunk-size sweep on the winner

Once the embedder + retrieval mode are settled, sweep chunk sizes to see if smaller chunks (often better for short administrative queries) improve faithfulness.

```bash
python scripts/compare_chunk_sizes.py \
    --source-run <run_name> \
    --questions ../code/questions_full_v2.json \
    --embedder intfloat/multilingual-e5-large \
    --mode dense \
    --top-k 4 \
    --chunk-sizes 200,300,400 \
    --judge-model claude-haiku-4-5
```

Output: `code/data/runs/<source>/eval/chunk_size_comparison.md`.

## Step 12 — Generator-model comparison on cluster

Once the retrieval side is settled, the generator-model comparison (Qwen / GaMS / Llama / Mistral) runs on the HPC cluster because the bigger models don't fit on a Mac. The flow is local-prep → rsync → sbatch → rsync back → local LLM-judge.

```bash
# 1) Local: convert the cleaned JSON eval set to the JSONL format evaluate.py reads,
#    written into <run>/eval/questions.jsonl on the run that holds the winning index.
python scripts/prepare_cluster_questions.py \
    --questions questions_full_v2.json \
    --run <run_name>__e5_base

# 2) Local: rsync code, index, and eval set to the cluster.
CLUSTER_HOST=user@hpc.fri.uni-lj.si \
CLUSTER_PATH=/d/hpc/users/$USER/fri-rag \
RUN_NAME=<run_name>__e5_base \
./scripts/sync_to_cluster.sh

# 3) On the cluster: submit compare_models.sh (uses winning retrieval config by default).
ssh $CLUSTER_HOST
cd $CLUSTER_PATH/code/slurm
RUN_NAME=<run_name>__e5_base sbatch compare_models.sh

# 4) Local (after the slurm job finishes): rsync results back.
CLUSTER_HOST=... CLUSTER_PATH=... RUN_NAME=<run_name>__e5_base \
./scripts/sync_from_cluster.sh

# 5) Local: judge each model with Claude (Haiku for the sweep) and write the table.
python scripts/compare_generators.py \
    --run <run_name>__e5_base \
    --reference-file questions_full_v2.json \
    --judge-model claude-haiku-4-5
```

Outputs:
- `code/data/runs/<run>/eval/results_<model>.jsonl` — generator outputs (from cluster)
- `code/data/runs/<run>/eval/judged_results_<model>.jsonl` — Claude per-model judgments
- `code/data/runs/<run>/eval/model_comparison.md` — final aggregated table

For the winning generator, re-judge with Sonnet via the suggested command at the bottom of `model_comparison.md`.

## Config

All settings (chunk size, embedding model, top-k, judge / question-generation models, …) are in `config.py`. Some are now env-overridable for the comparison drivers:

- `EMBEDDING_MODEL` (default `intfloat/multilingual-e5-base`)
- `CHUNK_SIZE` (default `400`)
- `CHUNK_OVERLAP` (default `80`)
- `JUDGE_MODEL` (default `claude-sonnet-4-6`)
- `EVAL_QUESTION_GEN_MODEL` (default `claude-sonnet-4-6`)

Setting any of these in the environment (or in `.env`) overrides the hardcoded default. The sweep scripts use this mechanism to launch subprocess builds with different embedders / chunk sizes without touching `config.py`.

When an index is built, the embedder name and chunk parameters are now persisted to `index/embedding_info.json` next to the FAISS file. On subsequent loads `vector_store.load_index` warns if any of these disagree with the current config; set `STRICT_INDEX_VALIDATION=1` to make a mismatch fatal.

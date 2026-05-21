# NLP course project: Chatbot for UL FRI students

Retrieval-augmented chatbot grounded in publicly-available UL/FRI web content.
The pipeline crawls the official sites, parses HTML/PDF/DOCX, chunks and
embeds the resulting text, indexes it in FAISS, and answers Slovenian/English
questions with a hybrid (BM25 + dense) retriever, a cross-encoder reranker
and an instruction-tuned LLM.

> **For peer reviewers / graders.** This repository is public on the
> UL-FRI-NLP-Course GitHub organisation. The default branch contains earlier
> work; the **`eval-pipeline`** branch contains the final submission with all
> evaluation artefacts. We do **not** train any model from scratch — every
> model used is downloaded from the Hugging Face Hub at run time (or called via
> the public Anthropic API for the LLM-as-judge). All datasets are publicly
> reachable UL/FRI web pages; we ship the FAISS index, chunks, and the
> Sonnet-audited evaluation set so graders can re-judge without re-crawling.

## Where to look first

| | Path |
|---|---|
| Project report (PDF) | [`report/report.pdf`](report/report.pdf) |
| Report sources (LaTeX + bib) | [`report/report.tex`](report/report.tex), [`report/report.bib`](report/report.bib) |
| Reproducible recipe (steps 1-12) | [`code/QUICKSTART.md`](code/QUICKSTART.md) |
| Top-line evaluation results | [`code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/SUMMARY.md`](code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/SUMMARY.md) |
| Generator comparison (Sonnet judge: Llama vs GaMS vs Mistral vs reference ceiling) | [`code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/model_comparison_sonnet.md`](code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/model_comparison_sonnet.md) |
| Cleaned evaluation set (51 in-scope + 10 negatives, post-audit) | [`code/questions_full_v2.json`](code/questions_full_v2.json) |
| Manually annotated dev set (25 Q with `relevant_doc_ids`) | [`code/data/runs/default/eval/questions.jsonl`](code/data/runs/default/eval/questions.jsonl) |
| Sonnet audit triage report | [`code/questions_audit_triage.md`](code/questions_audit_triage.md) |

## Repository layout

```
.
├── README.md                       # this file
├── annotate_eval.py …              # convenience launchers for the original
│   build_index.py …                # assignment-style entry points; each one
│   collect_data.py …               # just execs code/scripts/<same_name>.py
│   evaluate.py, test_retrieval.py
├── code/
│   ├── QUICKSTART.md               # full step-by-step recipe (Steps 1-12)
│   ├── config.py                   # central paths, embedder, chunk-size,
│   │                               # retrieval, recency/domain bias, judge
│   ├── requirements.txt
│   ├── data/runs/                  # one folder per experimental run
│   │   └── 2026-05-20__full__fri4_ul2__v1/      # winning full-crawl run
│   │       ├── index/              # FAISS index + chunk metadata
│   │       ├── processed/          # chunks.jsonl, sources.jsonl
│   │       ├── inputs/             # seed-link snapshot
│   │       └── eval/               # SUMMARY.md + comparison tables
│   ├── scripts/                    # CLI entry points (see below)
│   ├── slurm/                      # SLURM launcher scripts for the cluster
│   ├── src/                        # library code (chunking, embeddings,
│   │                               # retrieval, reranking, generation, ...)
│   └── tests/
├── raw_dataset/
│   ├── data_links.txt              # seed URLs for the crawl
│   └── files/                      # hand-curated PDFs/images
└── report/
    ├── report.tex / report.pdf
    ├── ds_report.cls               # course report class
    └── fig/                        # report figures
```

## Scripts at a glance

The same scripts can be invoked either via the **top-level wrappers**
(`python build_index.py`, `python evaluate.py`, …) or directly under
`code/scripts/`:

### Pipeline (build the system)

| Script | What it does |
|---|---|
| `code/scripts/collect_data.py` | Politely crawls the seed URLs (sitemaps + RSS, dedup by SHA), writes `raw/` + manifest |
| `code/scripts/build_index.py` | Parses raw → chunks → embeds → FAISS (`DISABLE_OCR=1` skips OCR for ~16× speedup) |
| `code/scripts/test_retrieval.py` | Smoke-tests retrieval against a saved index |
| `code/scripts/evaluate.py` | Generates answers with a chosen LLM and writes `eval/results_*.jsonl` |

### Evaluation (LLM-as-judge)

| Script | What it does |
|---|---|
| `code/scripts/generate_questions.py` | Offline question generation with Claude (`temperature=0` for reproducibility) |
| `code/scripts/audit_references.py` | Sonnet-graded fact-check of reference answers against the corpus |
| `code/scripts/apply_reference_audit.py` | Applies a fixed policy to produce `questions_full_v2.json` |
| `code/scripts/judge_answers.py` | Strict-JSON RAGAS-style judge for either generated answers or reference answers |
| `code/scripts/compare_embedders.py` | Embedder × retrieval-mode sweep (Haiku judge) |
| `code/scripts/compare_chunk_sizes.py` | Chunk-size sweep on the winning embedder/retrieval |
| `code/scripts/compare_generators.py` | Per-generator LLM-as-judge of the cluster results |

### Cluster handoff (laptop ⇄ HPC)

| Script | What it does |
|---|---|
| `code/scripts/prepare_cluster_questions.py` | Converts `questions_full_v2.json` to the JSONL format used on the cluster |
| `code/scripts/sync_to_cluster.sh` | rsync the run + index to the cluster |
| `code/scripts/sync_from_cluster.sh` | rsync the result files back |
| `code/scripts/local_smoke_test.sh` | End-to-end smoke test of the cluster-bound generator-eval flow |
| `code/slurm/{collect_data,build_index,evaluate,compare_models}.sh` | SLURM job scripts |

## Quickstart

Set up the environment and run a small end-to-end test:

```bash
conda create -n fri-rag python=3.11 -y && conda activate fri-rag
cd code && pip install -r requirements.txt

# Use the prebuilt index that ships with this branch:
python scripts/test_retrieval.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --query "Kdaj so uradne ure tajništva FRI?"
```

For the full pipeline (collect → parse → embed → judge → cluster
generator comparison), follow [`code/QUICKSTART.md`](code/QUICKSTART.md)
end-to-end.

### API keys / secrets

| Variable | Used by | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM-as-judge (`judge_answers.py`, `audit_references.py`, question generation) | <https://console.anthropic.com/> |
| `HF_TOKEN` | gated generator models on the cluster (Llama, Mistral, GaMS) | <https://huggingface.co/settings/tokens> |

Both can be exported as environment variables or placed in a `.env` file at the
repo root (`python-dotenv` is loaded automatically). Plain retrieval does not
need any keys; only the judge and the cluster generators do.

## External resources

The pipeline does not train any model. Everything below is fetched on demand:

| Component | Model | Link |
|---|---|---|
| First-stage embedder (default, used for the headline numbers) | `intfloat/multilingual-e5-base` | <https://huggingface.co/intfloat/multilingual-e5-base> |
| First-stage embedder (alt, ablated against e5-base) | `intfloat/multilingual-e5-large` | <https://huggingface.co/intfloat/multilingual-e5-large> |
| First-stage embedder (alt, ablated against e5-base) | `BAAI/bge-m3` | <https://huggingface.co/BAAI/bge-m3> |
| Cross-encoder reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1> |
| Generator (laptop default) | `Qwen/Qwen2.5-1.5B-Instruct` | <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct> |
| Generator (cluster, Slovenian-tuned) | `cjvt/GaMS3-12B-Instruct` | <https://huggingface.co/cjvt/GaMS3-12B-Instruct> |
| Generator (cluster, multilingual baselines) | `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` | gated on Hugging Face Hub |
| LLM-as-judge (final re-judgment, reference audit) | `claude-sonnet-4-6` | Anthropic API |
| LLM-as-judge (cheap comparison sweeps) | `claude-haiku-4-5` | Anthropic API |

Corpus seed list (publicly reachable UL/FRI web pages) is in
[`raw_dataset/data_links.txt`](raw_dataset/data_links.txt).

## Reproducing the report numbers

The exact configuration used in the report's headline table is
**`intfloat/multilingual-e5-base` + 400-token chunks + hybrid (BM25 + dense)
+ cross-encoder rerank + top-k=4**.

**1. Reproduce the reference-grounding ceiling (Sonnet):**

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

**2. Reproduce the generator comparison (Sonnet on cluster outputs):**

```bash
python scripts/compare_generators.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --reference-file questions_full_v2.json \
    --judge-model claude-sonnet-4-6 \
    --out-md model_comparison_sonnet.md
```

Per-mode and per-embedder numbers used in the report's tables are in:
- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/eval_matrix.md` (embedder × retrieval mode)
- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/chunk_size_comparison.md` (chunk-size sweep)
- `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/model_comparison_sonnet.md` (generators + ceiling)

## Data

We use publicly-available UL / UL-FRI content. Seed URLs are in
[`raw_dataset/data_links.txt`](raw_dataset/data_links.txt). The crawler
respects `robots.txt`, dedups by SHA-256, and filters to Slovenian + English
content (langdetect on first 4 KB).

### What ships with the repository

| | What | Why |
|---|---|---|
| ✅ | FAISS index + chunk metadata of the winning configuration (`code/data/runs/2026-05-20__full__fri4_ul2__v1/`) | so graders can re-judge / re-query without re-crawling |
| ✅ | Cleaned evaluation set `code/questions_full_v2.json` (51 in-scope + 10 negatives, Sonnet-audited) | this is the gold used in the report's headline table |
| ✅ | Manually annotated 25-question dev set (`code/data/runs/default/eval/questions.jsonl`) with `relevant_doc_ids` | source of the document-level retrieval metrics in the report |
| ✅ | All evaluation JSONLs (`results_*.jsonl`, `judged_*.jsonl`) and aggregation tables (`*.md`) | every number in the report is reproducible from these |
| ✅ | Audit triage report (`code/questions_audit_triage.md`) | per-question rationale for the reference rewrites |
| ✅ | All scripts: crawler, parser, indexer, retrieval, judge, audit, cluster handoff | full pipeline is one CLI per stage |
| ❌ | Full-crawl `parsed.jsonl` (~114 MB) and raw HTML/PDF (`raw/`, ~1 GB) | git-ignored due to size; regenerable via `collect_data.py` + `build_index.py` |
| ❌ | Model weights | downloaded on demand from Hugging Face Hub |

## Notes for graders

* No model is trained from scratch in this project — see the
  *External resources* table above for the exact pretrained models we use and
  links to download them.
* The full-crawl `parsed.jsonl` (~114 MB) and the raw HTML/PDF directory
  (~1 GB) are intentionally git-ignored — both can be regenerated from the
  seed links via `collect_data.py` + `build_index.py`. Everything needed to
  reproduce retrieval and the LLM-as-judge results (FAISS index, chunks,
  metadata, eval JSONLs) is committed under
  `code/data/runs/2026-05-20__full__fri4_ul2__v1/`.
* OCR (`DISABLE_OCR=0`, Docling + Tesseract) takes ~4 hours for ~500 PDFs.
  The default in `build_index.py` is OCR-on; set `DISABLE_OCR=1` for a fast
  PyMuPDF-only pass (~14 minutes) when you don't have scanned PDFs.
* Cluster scripts in `code/slurm/` assume the ARNES `onj_fri` allocation;
  they `module load CUDA/12.2.0 Python/3.11` and create a project-local
  `.venv`.

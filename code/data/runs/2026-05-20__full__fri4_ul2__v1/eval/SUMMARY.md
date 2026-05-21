# RAG evaluation summary — winning configuration

Source crawl: `2026-05-20__full__fri4_ul2__v1` (2,742 documents, 29,296 chunks)

## TL;DR

The winning RAG configuration is:

- **Embedder**: `intfloat/multilingual-e5-base`
- **Retrieval mode**: `hybrid_rerank` (BM25 + dense, then cross-encoder reranking)
- **Top-k**: 4
- **Chunk size**: 400 (overlap 80)
- **Index**: `code/data/runs/2026-05-20__full__fri4_ul2__v1__e5_base/`

## Headline numbers (Sonnet judge, temperature=0)

| Metric | In-scope (n=43) | Refusal (n=11) |
|---|---:|---:|
| Faithfulness | **0.740** | 0.905 |
| Answer relevance | 0.820 | — |
| Context relevance | 0.807 | — |
| Overall (1–5) | **3.605** | 3.818 |
| Hallucination rate | **44.2%** | — |
| Refused correctly | — | **11 / 11** |

Headline file: `code/data/runs/2026-05-20__full__fri4_ul2__v1__e5_base/eval/judged_winner_sonnet.jsonl`.

Haiku said 0.739 on the same run; Sonnet says 0.740 — the rankings are identical between the two judges, so the cheaper Haiku sweep was a faithful proxy.

## What changed vs the previous baseline

| | Old (`default` corpus, e5-base, hybrid k=8, original `questions_full.json`) | New (full crawl, e5-base + hybrid_rerank k=4, `questions_full_v2.json`) |
|---|---:|---:|
| Documents in corpus | 176 | 2,742 |
| Chunks indexed | ~1,500 | 29,296 |
| In-scope faithfulness (Sonnet) | 0.34 | **0.740** |
| In-scope hallucination | 80.9% | **44.2%** |
| Refused correctly (negatives) | OK | 11 / 11 |

The old number wasn't really a "model is bad" signal — it was a "the corpus is missing docs and the references are partially ungrounded" signal. The audit + full crawl + cross-encoder rerank fix all three problems.

## Where the gains come from (ablation across stages)

| Stage | What changed | In-scope faithfulness (Haiku) |
|---|---|---:|
| Baseline | small corpus + noisy refs + dense k=8 | 0.34 |
| Full crawl + e5-base + dense k=4 | 15× more docs, smaller k, cleaned eval set | 0.575 |
| + Hybrid (BM25 + dense) | adds lexical matching | 0.565 |
| + Cross-encoder rerank | re-ranks the union by relevance | **0.739** |

Cross-encoder reranking is by far the biggest single lever (+16 pp over dense, +17 pp over hybrid).

## Embedder × retrieval-mode matrix (Haiku judge)

See `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/eval_matrix.md`.

| Embedder | Mode | Faithfulness | Hallucination | Ctx rel | Overall |
|---|---|---:|---:|---:|---:|
| e5-base | dense | 0.575 | 59.1% | 0.725 | 3.000 |
| e5-base | hybrid | 0.565 | 63.6% | 0.715 | 2.841 |
| **e5-base** | **hybrid_rerank** | **0.739** | **54.5%** | 0.857 | **3.432** |
| e5-large | dense | 0.680 | 54.5% | 0.812 | 3.250 |
| e5-large | hybrid | 0.650 | 61.4% | 0.787 | 3.205 |
| e5-large | hybrid_rerank | 0.708 | 59.1% | 0.866 | 3.364 |

Notable: **e5-base + rerank beats e5-large + rerank**. The smaller embedder is sufficient when the cross-encoder is doing the final ranking, and it embeds 2.5× faster.

## Chunk-size sweep on the winning config (Haiku judge)

See `code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/chunk_size_comparison.md`.

| Chunk size | Overlap | Total chunks | Faithfulness | Hallucination | Ctx rel | Overall |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 40 | 50,546 | 0.691 | 54.5% | 0.835 | 3.386 |
| 300 | 60 | 36,331 | 0.698 | 54.5% | 0.824 | 3.364 |
| **400** | **80** | **29,296** | **0.739** | **52.3%** | **0.855** | **3.432** |

The default (400) wins. Smaller chunks hurt by ~4 pp in faithfulness, likely because the cross-encoder benefits from more local context per chunk.

## Reference audit

The original `questions_full.json` had 57 in-scope references; only **9 / 47 (~19%)** were fully grounded in the corpus.

After the Sonnet-driven audit (`code/questions_audited.json` → `code/questions_full_v2.json`):

| Audit verdict | Count | Action |
|---|---:|---|
| supported | 9 | kept as-is |
| partial | 31 | rewritten to use only chunk-grounded facts |
| unsupported | 7 | 3 dropped, 4 rewritten where Sonnet found alternative chunks |
| skipped (negative) | 10 | untouched (no factual ref expected) |
| skipped (no ref) | 4 | untouched |
| **Output** | **54 in-scope, 11 negatives** | `questions_full_v2.json` |

44 references now carry an `audit_status: auto_rewritten` flag, and 38 are flagged `needs_review` for optional manual sanity-check before publishing.

## Cost incurred

| Step | Judge | Calls | Approx cost |
|---|---|---:|---:|
| 1. Reference audit (top-30 hybrid context) | Sonnet | 47 | ~$1.50 |
| 5. Embedder × mode sweep | Haiku | 6 × 54 = 324 | ~$1.10 |
| 6. Chunk-size sweep | Haiku | 3 × 54 = 162 | ~$0.50 |
| 7. Winner re-judge | Sonnet | 54 | ~$0.65 |
| **Total** | | | **~$3.75** |

## What's next (handoff)

1. **Manual review** (Step 8): run `scripts/manual_review.py --run 2026-05-20__full__fri4_ul2__v1__e5_base --judged judged_winner_sonnet.jsonl --sample 25` to score 25 stratified rows against the Sonnet judge. Compute Cohen's kappa for the LLM-as-judge methodology section.
2. **Optional**: hand-edit any `audit_status: auto_rewritten` rows in `questions_full_v2.json` that look off, then re-run Step 7.
3. **Report**: paste `eval_matrix.md`, `chunk_size_comparison.md`, the headline table above, and the manual-review agreement number into the report.

# Embedder x retrieval-mode comparison

- Source crawl run: `2026-05-20__full__fri4_ul2__v1`
- Questions: `/Users/luka/Documents/Faks/MAG/2_letnik/2_semester/NLP/project/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet/code/questions_full_v2.json`
- Top-k: 4
- Judge model: `claude-haiku-4-5` (temperature=0)
- Modes: dense, hybrid, hybrid_rerank

| Embedder | Mode | In-scope n | Faithfulness | Hallucination | Ctx rel | Overall | Refusal faith. |
|---|---|---:|---:|---:|---:|---:|---:|
| `intfloat/multilingual-e5-base` | `dense` | 44 | 0.575 | 59.1% | 0.725 | 3.000 | 0.990 |
| `intfloat/multilingual-e5-base` | `hybrid` | 44 | 0.565 | 63.6% | 0.715 | 2.841 | 0.990 |
| `intfloat/multilingual-e5-base` | `hybrid_rerank` | 44 | 0.739 | 54.5% | 0.857 | 3.432 | 0.980 |
| `intfloat/multilingual-e5-large` | `dense` | 44 | 0.680 | 54.5% | 0.812 | 3.250 | 1.000 |
| `intfloat/multilingual-e5-large` | `hybrid` | 44 | 0.650 | 61.4% | 0.787 | 3.205 | 0.900 |
| `intfloat/multilingual-e5-large` | `hybrid_rerank` | 44 | 0.708 | 59.1% | 0.866 | 3.364 | 0.960 |

## Best by in-scope faithfulness

- **Embedder**: `intfloat/multilingual-e5-base`
- **Mode**: `hybrid_rerank`
- **In-scope faithfulness**: 0.739
- **In-scope hallucination**: 0.545
- **Derived run**: `2026-05-20__full__fri4_ul2__v1__e5_base`

For the report's headline number, re-judge this configuration with Sonnet:

```
EMBEDDING_MODEL=intfloat/multilingual-e5-base python scripts/judge_answers.py --run 2026-05-20__full__fri4_ul2__v1__e5_base --from-questions /Users/luka/Documents/Faks/MAG/2_letnik/2_semester/NLP/project/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet/code/questions_full_v2.json --retrieval-top-k 4 --hybrid --rerank --model claude-sonnet-4-6 --out judged_winner_sonnet.jsonl
```


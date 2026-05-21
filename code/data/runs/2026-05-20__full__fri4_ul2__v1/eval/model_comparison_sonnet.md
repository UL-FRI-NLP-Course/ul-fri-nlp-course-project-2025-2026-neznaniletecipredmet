# Generator-model comparison

- Run:               `2026-05-20__full__fri4_ul2__v1`
- Reference file:    `questions_full_v2.json`
- Judge model:       `claude-sonnet-4-6` (temperature=0.0)
- Retrieval context: top-4 chunks

| Model | In-scope n | Faithfulness | Ans rel | Ctx rel | Hallucination | Overall (1-5) | Refusal (1-5) | Avg gen time (s) | KW hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cjvt_GaMS3-12B-Instruct` | 54 | 0.592 | 0.725 | 0.688 | 66.7% | 2.907 | 3.857 | 13.3 | 72.7% |
| `meta-llama_Llama-3.1-8B-Instruct` | 52 | 0.665 | 0.690 | 0.707 | 50.0% | 3.000 | 3.667 | 14.7 | 68.0% |
| `mistralai_Mistral-7B-Instruct-v0.3` | 51 | 0.653 | 0.730 | 0.729 | 58.8% | 2.961 | 3.800 | 17.6 | 73.8% |

## Best by in-scope faithfulness

- **Model**: `meta-llama_Llama-3.1-8B-Instruct`
- **In-scope faithfulness**: 0.665
- **In-scope hallucination**: 50.0%
- **Overall (1-5)**: 3.000

For the report's headline number, re-judge this model with Sonnet:

```
python scripts/compare_generators.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --reference-file questions_full_v2.json \
    --judge-model claude-sonnet-4-6 \
    --include-pattern 'results_meta-llama_Llama-3.1-8B-Instruct.jsonl' \
    --out-md /Users/luka/Documents/Faks/MAG/2_letnik/2_semester/NLP/project/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet/code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/model_comparison_winner_sonnet.md
```


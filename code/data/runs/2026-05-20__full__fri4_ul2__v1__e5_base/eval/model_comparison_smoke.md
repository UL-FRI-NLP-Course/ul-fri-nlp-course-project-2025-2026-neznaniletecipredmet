# Generator-model comparison

- Run:               `2026-05-20__full__fri4_ul2__v1__e5_base`
- Reference file:    `questions_full_v2.json`
- Judge model:       `claude-haiku-4-5` (temperature=0.0)
- Retrieval context: top-4 chunks

| Model | In-scope n | Faithfulness | Ans rel | Ctx rel | Hallucination | Overall (1-5) | Refusal (1-5) | Avg gen time (s) | KW hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen2.5-1.5B-Instruct` | 3 | 0.500 | 0.593 | 0.683 | 66.7% | 2.667 | — | 30.6 | 100.0% |

## Best by in-scope faithfulness

- **Model**: `Qwen_Qwen2.5-1.5B-Instruct`
- **In-scope faithfulness**: 0.500
- **In-scope hallucination**: 66.7%
- **Overall (1-5)**: 2.667

For the report's headline number, re-judge this model with Sonnet:

```
python scripts/compare_generators.py \
    --run 2026-05-20__full__fri4_ul2__v1__e5_base \
    --reference-file questions_full_v2.json \
    --judge-model claude-sonnet-4-6 \
    --include-pattern 'results_Qwen_Qwen2.5-1.5B-Instruct.jsonl' \
    --out-md /Users/luka/Documents/Faks/MAG/2_letnik/2_semester/NLP/project/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet/code/data/runs/2026-05-20__full__fri4_ul2__v1__e5_base/eval/model_comparison_winner_sonnet.md
```


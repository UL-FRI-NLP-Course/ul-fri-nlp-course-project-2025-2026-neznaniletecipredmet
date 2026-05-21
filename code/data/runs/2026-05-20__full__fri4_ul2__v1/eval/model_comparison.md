# Generator-model comparison

- Run:               `2026-05-20__full__fri4_ul2__v1`
- Reference file:    `questions_full_v2.json`
- Judge model:       `claude-haiku-4-5` (temperature=0.0)
- Retrieval context: top-4 chunks

| Model | In-scope n | Faithfulness | Ans rel | Ctx rel | Hallucination | Overall (1-5) | Refusal (1-5) | Avg gen time (s) | KW hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cjvt_GaMS3-12B-Instruct` | 55 | 0.630 | 0.746 | 0.728 | 61.8% | 3.109 | 4.167 | 13.3 | 72.7% |

## Best by in-scope faithfulness

- **Model**: `cjvt_GaMS3-12B-Instruct`
- **In-scope faithfulness**: 0.630
- **In-scope hallucination**: 61.8%
- **Overall (1-5)**: 3.109

For the report's headline number, re-judge this model with Sonnet:

```
python scripts/compare_generators.py \
    --run 2026-05-20__full__fri4_ul2__v1 \
    --reference-file questions_full_v2.json \
    --judge-model claude-sonnet-4-6 \
    --include-pattern 'results_cjvt_GaMS3-12B-Instruct.jsonl' \
    --out-md /Users/luka/Documents/Faks/MAG/2_letnik/2_semester/NLP/project/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet/code/data/runs/2026-05-20__full__fri4_ul2__v1/eval/model_comparison_winner_sonnet.md
```


# Chunk-size comparison

- Source crawl run: `2026-05-20__full__fri4_ul2__v1`
- Embedder:         `intfloat/multilingual-e5-base`
- Retrieval mode:   `hybrid_rerank` (top-k=4)
- Questions:        `/Users/luka/Documents/Faks/MAG/2_letnik/2_semester/NLP/project/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet/code/questions_full_v2.json`
- Judge model:      `claude-haiku-4-5` (temperature=0)
- Overlap fraction: 0.2

| Chunk size | Overlap | Total chunks | In-scope n | Faithfulness | Hallucination | Ctx rel | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 40 | 50546 | 44 | 0.691 | 54.5% | 0.835 | 3.386 |
| 300 | 60 | 36331 | 44 | 0.698 | 54.5% | 0.824 | 3.364 |
| 400 | 80 | 29296 | 44 | 0.739 | 52.3% | 0.855 | 3.432 |

## Best chunk size

- **Chunk size 400 (default)** wins at faithfulness 0.739, hallucination 52.3%, overall 3.432.
- Smaller chunks (200, 300) score roughly 4 points lower in faithfulness with similar hallucination.
- Hypothesis: with `hybrid_rerank` retrieval, the cross-encoder benefits from larger chunks providing more local context; smaller chunks fragment related sentences across multiple top-k items.

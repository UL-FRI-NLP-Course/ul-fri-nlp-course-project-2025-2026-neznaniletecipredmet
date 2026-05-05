import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Note: tests focus on pure metric helpers (no FAISS / no HF model downloads).
from src.eval_metrics import (
    chunk_summaries,
    retrieval_chunk_metrics,
    retrieval_keyword_proxy,
    retrieval_metrics,
)


class TestEvaluateMetrics(unittest.TestCase):
    def test_retrieval_metrics_no_relevant(self) -> None:
        chunks = [{"doc_id": "a"}, {"doc_id": "b"}]
        m = retrieval_metrics(chunks, relevant_doc_ids=[], top_k=2)
        self.assertIsNone(m["retrieval_hit"])
        self.assertEqual(m["retrieved_doc_ids"], ["a", "b"])

    def test_retrieval_metrics_hit_rank_and_mrr(self) -> None:
        chunks = [{"doc_id": "x"}, {"doc_id": "gold"}, {"doc_id": "y"}]
        m = retrieval_metrics(chunks, relevant_doc_ids=["gold"], top_k=3)
        self.assertTrue(m["retrieval_hit"])
        self.assertEqual(m["first_relevant_rank"], 2)
        self.assertAlmostEqual(m["mrr_at_k"], 0.5)
        self.assertGreaterEqual(m["ndcg_at_k"], 0.0)

    def test_keyword_proxy(self) -> None:
        chunks = [
            {"text": "this is unrelated"},
            {"text": "contains Erasmus procedure details"},
        ]
        p = retrieval_keyword_proxy(chunks, expected_keywords=["Erasmus"], top_k=2)
        self.assertTrue(p["retrieved_keyword_hit"])
        self.assertEqual(p["retrieved_keyword_first_rank"], 2)

    def test_chunk_metrics_hit_rank_and_mrr(self) -> None:
        chunks = [
            {"chunk_id": "c0", "doc_id": "x"},
            {"chunk_id": "gold_chunk", "doc_id": "gold"},
            {"chunk_id": "c2", "doc_id": "y"},
        ]
        m = retrieval_chunk_metrics(chunks, relevant_chunk_ids=["gold_chunk"], top_k=3)
        self.assertTrue(m["chunk_hit"])
        self.assertEqual(m["chunk_first_relevant_rank"], 2)
        self.assertAlmostEqual(m["chunk_mrr_at_k"], 0.5)

    def test_chunk_summaries(self) -> None:
        chunks = [
            {
                "doc_id": "a",
                "chunk_id": "a_chunk0001",
                "title": "Doc",
                "section": "main",
                "url": "https://example.com",
                "score": 0.9,
                "rerank_score": 3.2,
            }
        ]
        s = chunk_summaries(chunks)
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0]["doc_id"], "a")
        self.assertIn("score", s[0])
        self.assertIn("rerank_score", s[0])


if __name__ == "__main__":
    unittest.main()

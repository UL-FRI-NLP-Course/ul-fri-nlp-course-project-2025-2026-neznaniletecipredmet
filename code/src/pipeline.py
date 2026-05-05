"""
RAG pipeline: ties retrieval, prompting, and generation together.
"""

import logging

import config
from src.generation import Generator
from src.prompting import build_prompt
from src.retrieval import retrieve
from src.utils import detect_language

log = logging.getLogger(__name__)

_default_generator: Generator | None = None


def _get_default_generator() -> Generator:
    global _default_generator
    if _default_generator is None:
        _default_generator = Generator()
    return _default_generator


def answer_question(
    question: str,
    top_k: int = config.TOP_K,
    generator: Generator | None = None,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    rerank_model: str | None = None,
    rerank_candidate_k: int | None = None,
    return_prompt: bool = False,
) -> dict:
    question_language = detect_language(question)
    log.info("Question language: %s | Question: %s", question_language, question[:80])

    retrieval_result = retrieve(
        question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        rerank_model=rerank_model,
        rerank_candidate_k=rerank_candidate_k,
    )
    chunks = retrieval_result["chunks"]
    scores = retrieval_result["scores"]
    retrieval_weak = retrieval_result["retrieval_weak"]

    messages = build_prompt(
        question=question,
        chunks=chunks,
        language=question_language,
        retrieval_weak=retrieval_weak,
    )

    gen = generator if generator is not None else _get_default_generator()
    answer = gen.generate(messages)

    result = {
        "answer": answer,
        "retrieved_chunks": chunks,
        "retrieval_scores": scores,
        "retrieval_weak": retrieval_weak,
        "question_language": question_language,
    }

    if return_prompt:
        result["prompt_used"] = messages

    return result

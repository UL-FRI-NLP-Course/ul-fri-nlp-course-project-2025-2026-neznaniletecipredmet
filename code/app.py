import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import config
from src.prompting import build_prompt
from src.retrieval import load, retrieve
from src.utils import detect_language

st.set_page_config(page_title="FRI RAG Asistent", page_icon="🎓", layout="centered")

st.title("🎓 FRI Študentski asistent")
st.caption("Odgovori na podlagi uradnih dokumentov UL FRI · slovenščina in angleščina")


@st.cache_resource(show_spinner="Nalaganje indeksa...")
def load_index(run_name: str):
    _ = run_name
    load()


@st.cache_resource(show_spinner="Nalaganje modela...")
def load_generator(model_name: str):
    from src.generation import Generator
    return Generator(model_name=model_name)


with st.sidebar:
    st.header("⚙️ Nastavitve")

    available_runs = []
    runs_dir = config.RUNS_DIR
    if runs_dir.exists():
        available_runs = [d.name for d in sorted(runs_dir.iterdir()) if d.is_dir()]
    if not available_runs:
        available_runs = ["default"]

    selected_run = st.selectbox("Nabor podatkov", available_runs, index=0)

    st.divider()

    retrieval_only = st.checkbox(
        "Samo iskanje (brez LLM)",
        value=True,
        help="Prikaži najdene odlomke brez generiranja odgovora z LLM. Primerno za testiranje brez GPUja.",
    )

    selected_model = config.LOCAL_TEST_MODEL
    if not retrieval_only:
        model_choices = [config.LOCAL_TEST_MODEL] + config.COMPARISON_MODELS
        selected_model = st.selectbox("Model za generiranje", model_choices)

    top_k = st.slider("Število vrnjenih odlomkov (top-k)", min_value=1, max_value=10, value=config.TOP_K)
    use_hybrid = st.checkbox("Hibridno iskanje (BM25 + semantično)", value=False)

    use_rerank = st.checkbox("Rerank (cross-encoder)", value=False)
    rerank_candidate_k = None
    rerank_model = None
    if use_rerank:
        rerank_candidate_k = st.slider(
            "Rerank kandidatov (candidate-k)",
            min_value=top_k,
            max_value=50,
            value=getattr(config, "RERANK_CANDIDATE_K", max(top_k * 5, 20)),
            help="Najprej poišče candidate-k odlomkov, nato jih reranka in vrne top-k.",
        )
        rerank_model = st.text_input(
            "Rerank model",
            value=getattr(config, "RERANK_MODEL", ""),
            help="HuggingFace cross-encoder model name.",
        ).strip() or None

    st.divider()
    show_prompt = st.checkbox("Prikaži prompt (debug)", value=False)
    show_scores = st.checkbox("Prikaži ocene odlomkov", value=True)


config.apply_run(selected_run)
load_index(selected_run)

question = st.text_input(
    "Vaše vprašanje:",
    placeholder="Koliko krat lahko opravljam izpit? / How many times can I take an exam?",
)

if st.button("Vprašaj", type="primary") and question.strip():
    question_language = detect_language(question)
    lang_label = "🇸🇮 Slovenščina" if question_language == "sl" else "🇬🇧 Angleščina"
    st.caption(f"Zaznani jezik: {lang_label}")

    with st.spinner("Iščem relevantne odlomke..."):
        retrieval_result = retrieve(
            question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            rerank_model=rerank_model,
            rerank_candidate_k=rerank_candidate_k,
        )

    chunks = retrieval_result["chunks"]
    retrieval_weak = retrieval_result["retrieval_weak"]

    if retrieval_weak:
        st.warning("⚠️ Zaupanje v iskanje je nizko — odgovor morda ni relevanten za vaše vprašanje.")

    if not retrieval_only:
        generator = load_generator(selected_model)
        with st.spinner("Generiram odgovor..."):
            from src.pipeline import answer_question
            result = answer_question(
                question=question,
                top_k=top_k,
                generator=generator,
                use_hybrid=use_hybrid,
                return_prompt=show_prompt,
            )

        st.subheader("Odgovor")
        st.write(result["answer"])

        if show_prompt and "prompt_used" in result:
            with st.expander("🔍 Prompt poslan modelu"):
                for msg in result["prompt_used"]:
                    st.markdown(f"**{msg['role'].upper()}**")
                    st.code(msg["content"], language="text")

    with st.expander(f"📄 Viri ({len(chunks)} najdenih odlomkov)", expanded=retrieval_only):
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Neznano")
            section = chunk.get("section", "")
            score = chunk.get("score", 0.0)
            pre = chunk.get("pre_rerank_score", None)
            rerank_score = chunk.get("rerank_score", None)
            vec = chunk.get("vector_score", None)
            bm25 = chunk.get("bm25_score", None)
            url = chunk.get("url", "")
            source = f"{title} — {section}" if section and section not in ("main", "") else title

            header = f"**[{i}] {source}**"
            if show_scores:
                parts = [f"ocena: `{float(score):.3f}`"]
                if rerank_score is not None:
                    parts.append(f"rerank: `{float(rerank_score):.3f}`")
                if pre is not None:
                    parts.append(f"pre: `{float(pre):.3f}`")
                if vec is not None and pre is None:
                    parts.append(f"vec: `{float(vec):.3f}`")
                if bm25 is not None:
                    parts.append(f"bm25: `{float(bm25):.3f}`")
                header += " · " + " | ".join(parts)
            st.markdown(header)

            if url:
                st.markdown(f"🔗 [{url}]({url})")

            preview = chunk["text"][:600]
            if len(chunk["text"]) > 600:
                preview += "…"
            st.text(preview)
            st.divider()

    if show_prompt and retrieval_only:
        with st.expander("🔍 Prompt ki bi bil poslan modelu"):
            messages = build_prompt(question, chunks, language=question_language, retrieval_weak=retrieval_weak)
            for msg in messages:
                st.markdown(f"**{msg['role'].upper()}**")
                st.code(msg["content"], language="text")

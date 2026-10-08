"""Streamlit UI for ExamRAG. Run with:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from examrag.config import load_settings  # noqa: E402
from examrag.embeddings import Embedder  # noqa: E402
from examrag.llm import get_llm  # noqa: E402
from examrag.pipeline import index_pdf  # noqa: E402
from examrag.rag import LANGUAGES, MARK_OPTIONS, answer_question  # noqa: E402
from examrag.vectorstore import VectorStore  # noqa: E402

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ExamRAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading settings...")
def get_settings():
    return load_settings()


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


@st.cache_resource(show_spinner=False)
def get_store(persist_dir: str) -> VectorStore:
    return VectorStore(persist_dir)


settings = get_settings()
embedder = get_embedder(settings.embedding_model)
store = get_store(settings.chroma_dir)

st.title("📚 ExamRAG")
st.caption(
    "Exam-aware study assistant. Answers come only from your documents, "
    "with page references."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Your documents")
    uploads = st.file_uploader(
        "Upload notes, textbooks or past papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", disabled=not uploads, type="primary"):
        for up in uploads:
            path = UPLOAD_DIR / Path(up.name).name
            path.write_bytes(up.getbuffer())
            with st.spinner(f"Indexing {up.name}..."):
                try:
                    n = index_pdf(path, store, embedder, settings)
                    st.success(f"{up.name}: {n} chunks indexed")
                except ValueError as exc:
                    st.error(str(exc))

    sources = store.list_sources()
    if sources:
        st.markdown("**Indexed documents**")
        for s in sources:
            cols = st.columns([5, 1])
            cols[0].write(s)
            if cols[1].button("🗑", key=f"del-{s}", help=f"Remove {s}"):
                store.delete_source(s)
                st.rerun()
    else:
        st.info("No documents yet. Upload a PDF and click Index.")

    st.header("2. Answer settings")
    marks = st.selectbox(
        "Exam marks",
        MARK_OPTIONS,
        format_func=lambda m: "Just answer" if m == 0 else f"{m} marks",
        help="Shapes the length and structure of the answer like an exam paper.",
    )
    language = st.selectbox("Answer language", LANGUAGES)
    source_filter = st.selectbox("Search in", ["All documents"] + sources)
    source = None if source_filter == "All documents" else source_filter
    units = store.list_units(source)
    unit_filter = st.selectbox("Unit / chapter", ["All"] + units)
    unit = None if unit_filter == "All" else unit_filter

    st.divider()
    st.caption(
        f"LLM: **{settings.llm_provider}** · Embeddings: `{settings.embedding_model}`"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(hits):
    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            unit_txt = f" · {h.unit}" if h.unit else ""
            st.markdown(
                f"**[{i}]** {h.source} · page {h.page}{unit_txt} "
                f"· similarity {h.score:.2f}"
            )
            st.caption(h.text[:400] + ("..." if len(h.text) > 400 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"])

question = st.chat_input("Ask a question about your documents (English, தமிழ், हिन्दी)")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if store.count() == 0:
            reply = "Upload and index at least one PDF first (left sidebar)."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            try:
                llm = get_llm(settings)
                with st.spinner("Searching your documents..."):
                    result = answer_question(
                        question,
                        store,
                        embedder,
                        llm,
                        settings,
                        marks=marks,
                        language=language,
                        source=source,
                        unit=unit,
                    )
                st.markdown(result.text)
                if result.found and result.hits:
                    render_sources(result.hits)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.text,
                        "hits": result.hits if result.found else [],
                    }
                )
            except RuntimeError as exc:
                st.error(str(exc))

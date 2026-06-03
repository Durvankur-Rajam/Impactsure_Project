
import os
import sys

# Current file: Month_3/Week_11/app/app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Month_3/Week_11
week11_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Month_3/Week_10
week10_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "Week_10"))

# Add folders to Python path
sys.path.insert(0, week11_dir)
sys.path.insert(0, week10_dir)

import streamlit as st
import tempfile
import time
import chromadb

from utils.pdf_chunker import process_pdf

from rag.embedder import (
    get_embedder,
    embed_and_store,
    get_collection
)

from rag.retriever import (
    retrieve,
    build_context
)

from rag.generator import (
    load_model,
    generate_baseline,
    generate_rag
)


# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Week_10'))

# import streamlit as st
# import tempfile
# import time
# import chromadb

# from utils.pdf_chunker import process_pdf
# from rag.embedder import get_embedder, embed_and_store, get_collection
# from rag.retriever import retrieve, build_context
# from rag.generator import load_model, generate_baseline, generate_rag


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "HalluciCheck",
    page_icon  = "🔍",
    layout     = "wide"
)

st.title("HalluciCheck")
st.caption("Compare Baseline LLM vs RAG answers — see hallucinations in real time")
st.divider()


# ── Load model and embedder (cached so they only load once) ──────────────────
@st.cache_resource
def load_resources():
    model, tokenizer = load_model(device="auto")
    embedder         = get_embedder()
    return model, tokenizer, embedder

with st.spinner("Loading model and embedder... (first run takes 2-3 mins)"):
    model, tokenizer, embedder = load_resources()

st.success("Model and embedder ready!")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k       = st.slider("Chunks to retrieve (top-k)", 1, 10, 3)
    max_tokens  = st.slider("Max answer tokens", 50, 300, 150)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Ask a question")
    st.markdown("3. See Baseline vs RAG answer")
    st.divider()
    st.markdown("**Model:** TinyLlama-1.1B")
    st.markdown("**Embedder:** all-MiniLM-L6-v2")
    st.markdown("**Vector store:** ChromaDB")


# ── PDF Upload ────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF — extracting and embedding chunks..."):
        chunks = process_pdf(tmp_path)
        client = chromadb.Client()
        collection, client = get_collection(client)
        embed_and_store(chunks, uploaded_file.name, embedder, collection, client)

    st.success(f"PDF processed — {len(chunks)} chunks embedded into ChromaDB")
    os.unlink(tmp_path)

    # ── Question Input ────────────────────────────────────────────────────────
    st.subheader("Step 2 — Ask a Question")
    question = st.text_input("Enter your question about the PDF:")

    if question:
        st.subheader("Step 3 — Answers")

        left_col, right_col = st.columns(2)

        # ── Left panel: Baseline ──────────────────────────────────────────────
        with left_col:
            st.markdown("### Baseline LLM")
            st.caption("No retrieval — answer from model memory only")
            with st.spinner("Generating baseline answer..."):
                baseline_answer, baseline_latency = generate_baseline(
                    question, model, tokenizer, max_new_tokens=max_tokens
                )
            st.info(baseline_answer)
            st.caption(f"Latency: {baseline_latency}s")

        # ── Right panel: RAG ──────────────────────────────────────────────────
        with right_col:
            st.markdown("### RAG Answer")
            st.caption("Retrieval-augmented — grounded in your PDF")
            with st.spinner("Retrieving chunks and generating RAG answer..."):
                docs, sources = retrieve(question, collection, embedder, top_k=top_k)
                context       = build_context(docs)
                rag_answer, rag_latency = generate_rag(
                    question, context, model, tokenizer, max_new_tokens=max_tokens
                )
            st.success(rag_answer)
            st.caption(f"Latency: {rag_latency}s")

        # ── Grounding Score ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Grounding Score")

        rag_words     = set(rag_answer.lower().split())
        context_words = set(context.lower().split())
        if rag_words:
            grounding = len(rag_words & context_words) / len(rag_words) * 100
        else:
            grounding = 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Grounding Score",  f"{grounding:.1f}%")
        col2.metric("Baseline Latency", f"{baseline_latency}s")
        col3.metric("RAG Latency",      f"{rag_latency}s")

        st.progress(int(grounding))
        if grounding >= 50:
            st.success("High grounding — RAG answer is well supported by the PDF")
        elif grounding >= 25:
            st.warning("Medium grounding — RAG answer is partially supported")
        else:
            st.error("Low grounding — answer may not be well supported by the PDF")

        # ── Source Chunks ─────────────────────────────────────────────────────
        st.divider()
        with st.expander("View retrieved source chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i + 1}** — source: `{sources[i]}`")
                st.text(doc[:400])
                st.divider()
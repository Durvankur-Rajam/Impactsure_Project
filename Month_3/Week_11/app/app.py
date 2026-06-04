import os
import sys

# Current file: Month_3/Week_11/app/app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
week11_dir  = os.path.abspath(os.path.join(current_dir, ".."))
week10_dir  = os.path.abspath(os.path.join(current_dir, "..", "..", "Week_10"))

sys.path.insert(0, week11_dir)
sys.path.insert(0, week10_dir)

import streamlit as st
import tempfile
import chromadb
import pandas as pd

from utils.pdf_chunker import process_pdf
from rag.embedder import get_embedder, embed_and_store, get_collection
from rag.retriever import retrieve, build_context
from rag.generator import load_model, generate_baseline, generate_rag


st.set_page_config(
    page_title = "HalluciCheck",
    page_icon  = "🔍",
    layout     = "wide"
)

st.title("HalluciCheck")
st.caption("Compare Baseline LLM vs RAG answers — see hallucinations in real time")
st.divider()


@st.cache_resource
def load_resources():
    model, tokenizer = load_model(device="auto")
    embedder         = get_embedder()
    return model, tokenizer, embedder

with st.spinner("Loading model and embedder... (first run takes 2-3 mins)"):
    model, tokenizer, embedder = load_resources()

st.success("Model and embedder ready!")


with st.sidebar:
    st.header("Settings")
    top_k      = st.slider("Chunks to retrieve (top-k)", 1, 10, 3)
    max_tokens = st.slider("Max answer tokens", 50, 300, 150)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Ask a question")
    st.markdown("3. See Baseline vs RAG answer")
    st.divider()
    st.markdown("**Model:** TinyLlama-1.1B")
    st.markdown("**Embedder:** all-MiniLM-L6-v2")
    st.markdown("**Vector store:** ChromaDB")
    st.divider()
    if st.button("Clear cache and reset"):
        st.cache_resource.clear()
        st.rerun()

if "test_results" not in st.session_state:
    st.session_state.test_results = []

if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.client     = None


st.subheader("Step 1 — Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 50:
        st.error("File too large — please upload a PDF under 50MB")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF — extracting and embedding chunks..."):
        try:
            chunks = process_pdf(tmp_path)
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            os.unlink(tmp_path)
            st.stop()

    if not chunks:
        st.error("No text could be extracted. Try a different PDF.")
        os.unlink(tmp_path)
        st.stop()

    client     = chromadb.Client()
    collection, client = get_collection(client)
    st.session_state.collection = collection
    st.session_state.client     = client

    with st.spinner(f"Embedding {len(chunks)} chunks into ChromaDB..."):
        embed_and_store(chunks, uploaded_file.name, embedder, collection, client)

    st.success(f"PDF processed — {len(chunks)} chunks embedded into ChromaDB")
    st.info(f"File: {uploaded_file.name}  |  Size: {file_size_mb:.1f}MB  |  Chunks: {len(chunks)}")
    os.unlink(tmp_path)

    st.subheader("Step 2 — Ask a Question")
    question = st.text_input("Enter your question about the PDF:")

    # Bug fix 5: short question check
    if question and len(question.strip()) < 5:
        st.warning("Please enter a more specific question.")
        st.stop()

    if question:
        st.subheader("Step 3 — Answers")

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Baseline LLM")
            st.caption("No retrieval — answer from model memory only")
            with st.spinner("Generating baseline answer..."):
                try:
                    baseline_answer, baseline_latency = generate_baseline(
                        question, model, tokenizer, max_new_tokens=max_tokens
                    )
                except Exception as e:
                    st.error(f"Baseline generation failed: {e}")
                    st.stop()
            st.info(baseline_answer)
            st.caption(f"Latency: {baseline_latency}s")

        with right_col:
            st.markdown("### RAG Answer")
            st.caption("Retrieval-augmented — grounded in your PDF")
            with st.spinner("Retrieving chunks and generating RAG answer..."):
                try:
                    docs, sources = retrieve(
                        question,
                        st.session_state.collection,
                        embedder,
                        top_k=top_k
                    )
                    context       = build_context(docs)
                    rag_answer, rag_latency = generate_rag(
                        question, context, model, tokenizer, max_new_tokens=max_tokens
                    )
                except Exception as e:
                    st.error(f"RAG generation failed: {e}")
                    st.stop()
            st.success(rag_answer)
            st.caption(f"Latency: {rag_latency}s")

        st.divider()
        st.subheader("Grounding Score")

        rag_words     = set(rag_answer.lower().split())
        context_words = set(context.lower().split())
        grounding     = len(rag_words & context_words) / len(rag_words) * 100 if rag_words else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Grounding Score",  f"{grounding:.1f}%")
        col2.metric("Baseline Latency", f"{baseline_latency}s")
        col3.metric("RAG Latency",      f"{rag_latency}s")

        st.progress(int(min(grounding, 100)))
        if grounding >= 50:
            st.success("High grounding — RAG answer is well supported by the PDF")
        elif grounding >= 25:
            st.warning("Medium grounding — RAG answer is partially supported")
        else:
            st.error("Low grounding — answer may not be well supported by the PDF")

        st.divider()
        with st.expander("View retrieved source chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i + 1}** — source: `{sources[i]}`")
                st.text(doc[:400])
                st.divider()

        st.session_state.test_results.append({
            "pdf"              : uploaded_file.name,
            "question"         : question,
            "baseline_answer"  : baseline_answer[:150],
            "rag_answer"       : rag_answer[:150],
            "grounding_score"  : round(grounding, 1),
            "baseline_latency" : baseline_latency,
            "rag_latency"      : rag_latency,
        })

        os.makedirs(
            os.path.join(current_dir, "..", "..", "..", "Week12", "evaluation"),
            exist_ok=True
        )
        eval_path = os.path.join(
            current_dir, "..", "..", "..", "Week12", "evaluation", "eval_results.csv"
        )
        pd.DataFrame(st.session_state.test_results).to_csv(eval_path, index=False)

if st.session_state.test_results:
    st.divider()
    st.subheader("Test Results Across All PDFs")
    df = pd.DataFrame(st.session_state.test_results)
    st.dataframe(df)
    col1, col2 = st.columns(2)
    col1.metric("Total Tests Run",     len(df))
    col2.metric("Avg Grounding Score", f"{df['grounding_score'].mean():.1f}%")
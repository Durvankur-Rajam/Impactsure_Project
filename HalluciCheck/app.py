import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import streamlit as st
import tempfile
import chromadb
import pandas as pd
import time
import torch

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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # GPU/CPU detection
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device_option = st.radio(
            "Inference device",
            ["GPU", "CPU"],
            index = 0,
            help  = "GPU is faster. CPU is slower but works without a GPU."
        )
        device = "auto" if device_option == "GPU" else "cpu"
        st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("No GPU detected — running on CPU")
        device_option = "CPU"
        device        = "cpu"

    st.divider()
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


# ── Load model and embedder ───────────────────────────────────────────────────
@st.cache_resource
def load_resources(device):
    load_start       = time.time()
    model, tokenizer = load_model(device=device)
    embedder         = get_embedder()
    load_time        = round(time.time() - load_start, 1)
    return model, tokenizer, embedder, load_time

with st.spinner(f"Loading model on {device_option}... (first run takes 2-3 mins)"):
    model, tokenizer, embedder, load_time = load_resources(device)

st.success(f"Model ready on {device_option} — loaded in {load_time}s")


# ── Session state ─────────────────────────────────────────────────────────────
if "test_results" not in st.session_state:
    st.session_state.test_results = []

if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.client     = None


# ── PDF Upload ────────────────────────────────────────────────────────────────
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


    # ── Question Input ────────────────────────────────────────────────────────
    st.subheader("Step 2 — Ask a Question")
    question = st.text_input("Enter your question about the PDF:")

    if question and len(question.strip()) < 5:
        st.warning("Please enter a more specific question.")
        st.stop()

    if question:
        st.subheader("Step 3 — Answers")

        left_col, right_col = st.columns(2)

        # ── Left panel: Baseline ──────────────────────────────────────────────
        with left_col:
            st.markdown("### Baseline LLM")
            st.caption(f"No retrieval — model memory only ({device_option})")
            with st.spinner("Generating baseline answer..."):
                try:
                    baseline_answer, baseline_latency = generate_baseline(
                        question, model, tokenizer, max_new_tokens=max_tokens
                    )
                except Exception as e:
                    st.error(f"Baseline generation failed: {e}")
                    st.stop()
            st.info(baseline_answer)
            st.metric("Baseline Latency", f"{baseline_latency}s")

        # ── Right panel: RAG ──────────────────────────────────────────────────
        with right_col:
            st.markdown("### RAG Answer")
            st.caption(f"Retrieval-augmented — grounded in PDF ({device_option})")
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
            st.metric("RAG Latency", f"{rag_latency}s")

        # Latency comparison
        if baseline_latency > 0 and rag_latency > 0:
            faster = "Baseline" if baseline_latency < rag_latency else "RAG"
            diff   = abs(baseline_latency - rag_latency)
            st.caption(f"{faster} was faster by {diff:.2f}s on {device_option}")

        # ── Grounding Score ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Grounding Score")

        rag_words     = set(rag_answer.lower().split())
        context_words = set(context.lower().split())
        grounding     = len(rag_words & context_words) / len(rag_words) * 100 if rag_words else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Grounding Score",  f"{grounding:.1f}%")
        col2.metric("Baseline Latency", f"{baseline_latency}s")
        col3.metric("RAG Latency",      f"{rag_latency}s")
        col4.metric("Device",           device_option)

        st.progress(int(min(grounding, 100)))
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

        # ── Save results ──────────────────────────────────────────────────────
        st.session_state.test_results.append({
            "pdf"              : uploaded_file.name,
            "question"         : question,
            "baseline_answer"  : baseline_answer[:150],
            "rag_answer"       : rag_answer[:150],
            "grounding_score"  : round(grounding, 1),
            "baseline_latency" : baseline_latency,
            "rag_latency"      : rag_latency,
            "device"           : device_option,
        })

        eval_path = os.path.join(current_dir, "evaluation", "eval_results.csv")
        os.makedirs(os.path.join(current_dir, "evaluation"), exist_ok=True)
        pd.DataFrame(st.session_state.test_results).to_csv(eval_path, index=False)


# ── Test results table ────────────────────────────────────────────────────────
if st.session_state.test_results:
    st.divider()
    st.subheader("Test Results Across All PDFs")
    df = pd.DataFrame(st.session_state.test_results)
    st.dataframe(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tests Run",     len(df))
    col2.metric("Avg Grounding Score", f"{df['grounding_score'].mean():.1f}%")
    col3.metric("Avg RAG Latency",     f"{df['rag_latency'].mean():.2f}s")

    if len(df['device'].unique()) > 1:
        st.subheader("GPU vs CPU Latency Comparison")
        device_summary = df.groupby('device')[['baseline_latency', 'rag_latency']].mean().round(2)
        st.dataframe(device_summary)
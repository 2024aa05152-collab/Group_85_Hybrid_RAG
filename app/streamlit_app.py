import streamlit as st
import time
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_generator import ResponseGenerator

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wikipedia Hybrid RAG System",
    page_icon="🔍",
    layout="wide"
)

# ---------------- SYSTEM INITIALIZATION ----------------
@st.cache_resource(show_spinner="Initializing Hybrid RAG system...")
def load_system():
    # Load corpus
    with open("data/corpus_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Dense retriever
    dense = DenseRetriever()
    if os.path.exists("data/dense.index"):
        dense.load("data", chunks)
    else:
        dense.build_index(chunks)
        dense.save("data")

    # Sparse retriever
    sparse = SparseRetriever()
    if os.path.exists("data/sparse.pkl"):
        sparse.load("data", chunks)
    else:
        sparse.build_index(chunks)
        sparse.save("data")

    hybrid = HybridRetriever(dense, sparse)
    generator = ResponseGenerator()

    return hybrid, generator

hybrid, generator = load_system()

# ---------------- UI HEADER ----------------
st.title("🤖 Wikipedia Hybrid RAG System")
st.caption("Dense (Embedding) + Sparse (BM25) Retrieval with LLM-based Answer Generation")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of sources to retrieve", 1, 5, 3)
    st.divider()
    st.info(
        "This system uses **Hybrid Retrieval**:\n\n"
        "- Dense search (semantic embeddings)\n"
        "- Sparse search (keyword-based BM25)\n\n"
        "The retrieved chunks are fused and passed to an LLM for answer generation."
    )

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 View retrieved sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**Chunk ID:** `{src['chunk_id']}`  \n"
                        f"**Hybrid Score (RRF):** `{src['rrf_score']:.4f}`"
                    )
                    st.caption(src["text"])

# ---------------- USER INPUT ----------------
if query := st.chat_input("Ask a question about the dataset..."):
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):
        start_time = time.time()

        with st.spinner("Retrieving context and generating answer..."):
            retrieved_chunks = hybrid.retrieve(query, top_k=top_k)
            answer = generator.generate(query, retrieved_chunks)

        st.markdown(answer)
        st.caption(f"⏱️ Response generated in {time.time() - start_time:.2f} seconds")

        with st.expander("📚 Evidence (Top Retrieved Chunks)"):
            for r in retrieved_chunks:
                st.markdown(f"**Chunk {r['chunk_id']}**")
                st.info(r["text"])

                cols = st.columns(3)
                cols[0].metric("Dense Score", f"{r.get('dense_score', 0):.3f}")
                cols[1].metric("Sparse Score", f"{r.get('sparse_score', 0):.3f}")
                cols[2].metric("RRF Score", f"{r.get('rrf_score', 0):.3f}")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks
        }
    )

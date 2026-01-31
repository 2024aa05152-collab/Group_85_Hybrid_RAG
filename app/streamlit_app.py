import streamlit as st
import time
import json
import os
import sys
from pathlib import Path

# Fix Path
root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path: sys.path.insert(0, root)

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_generator import ResponseGenerator

st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide", page_icon="🔍")

# --- SYSTEM INITIALIZATION ---
@st.cache_resource(show_spinner="Loading AI Engines...")
def get_system():
    # 1. Load Chunks
    with open('data/corpus_chunks.json', 'r') as f:
        chunks = json.load(f)
    
    # 2. Dense Setup
    dense = DenseRetriever()
    if os.path.exists("data/dense.index"):
        dense.load("data", chunks)
    else:
        dense.build_index(chunks)
        dense.save("data")
            
    # 3. Sparse Setup
    sparse = SparseRetriever()
    if os.path.exists("data/sparse.pkl"):
        sparse.load("data", chunks)
    else:
        sparse.build_index(chunks)
        sparse.save("data")
            
    return HybridRetriever(dense, sparse), ResponseGenerator()

# Load the system
hybrid, generator = get_system()

# --- UI LAYOUT ---
st.title("🤖 Wikipedia Hybrid RAG")

# Sidebar for metadata and settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of sources to retrieve:", 1, 5, 3)
    st.divider()
    st.info("This system uses a Hybrid search (Dense + Sparse) to find the most accurate context from Wikipedia.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View retrieved sources for this answer"):
                for res in message["sources"]:
                    st.write(f"**Chunk {res['chunk_id']}** (RRF: {res['rrf_score']:.4f})")
                    st.caption(res['text'])

# Chat Input
if query := st.chat_input("Ask a question about the dataset..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner("Searching and Thinking..."):
            # Retrieval
            results = hybrid.retrieve(query, top_k=top_k)
            # Generation
            answer = generator.generate(query, results)
        
        # Display response
        st.markdown(answer)
        st.caption(f"Generated in {time.time()-t0:.2f}s")
        
        # Show sources in an expander
        with st.expander("🔍 Evidence (Top Sources)"):
            for res in results:
                st.write(f"**Chunk {res['chunk_id']}** (RRF: {res['rrf_score']:.4f})")
                st.info(res['text'])
                cols = st.columns(3)
                cols[0].metric("Dense Score", f"{res.get('dense_score', 0):.3f}")
                cols[1].metric("Sparse Score", f"{res.get('sparse_score', 0):.3f}")
                cols[2].metric("Hybrid Rank", res.get('dense_rank', 'N/A'))

    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": results
    })
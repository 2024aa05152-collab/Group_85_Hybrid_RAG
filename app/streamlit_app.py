#!/usr/bin/env python3
"""
Streamlit Web Interface for Hybrid RAG System
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_CONFIG, RETRIEVAL_CONFIG, FILE_PATHS
from src.indexing.dense_index import DenseIndexer
from src.indexing.sparse_index import SparseIndexer
from src.indexing.hybrid_rrf import HybridRetriever
from src.generation.llm_generator import LLMGenerator

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chunk-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .answer-box {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Load the Hybrid RAG system components"""
    # Load indices
    dense_retriever = DenseIndexer(MODEL_CONFIG["embedding_model"])
    dense_retriever.load_index(
        FILE_PATHS["faiss_index"],
        FILE_PATHS["metadata"]
    )

    sparse_retriever = SparseIndexer()
    sparse_retriever.load_index(
        FILE_PATHS["bm25_index"],
        FILE_PATHS["metadata"]
    )

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        k=RETRIEVAL_CONFIG["rrf_k"]
    )

    # Create LLM generator
    llm_generator = LLMGenerator(model_name=MODEL_CONFIG["llm_model"])

    return hybrid_retriever, llm_generator


def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">🔍 Hybrid RAG System</h1>',
                unsafe_allow_html=True)
    st.markdown("### Combining Dense + Sparse Retrieval with Reciprocal Rank Fusion")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        retrieval_k = st.slider(
            "Retrieval K (top chunks per method)",
            min_value=5,
            max_value=50,
            value=RETRIEVAL_CONFIG["top_k"],
            help="Number of chunks to retrieve from each method"
        )

        fusion_n = st.slider(
            "Fusion N (final chunks after RRF)",
            min_value=3,
            max_value=10,
            value=RETRIEVAL_CONFIG["top_n"],
            help="Number of final chunks to use for answer generation"
        )

        rrf_k = st.slider(
            "RRF constant (k)",
            min_value=10,
            max_value=100,
            value=RETRIEVAL_CONFIG["rrf_k"],
            help="Constant in RRF formula: 1/(k + rank)"
        )

        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.write(f"**Embedding Model:** {MODEL_CONFIG['embedding_model']}")
        st.write(f"**LLM Model:** {MODEL_CONFIG['llm_model']}")
        st.write(f"**Chunk Size:** {RETRIEVAL_CONFIG['chunk_size']} tokens")
        st.write(f"**Chunk Overlap:** {RETRIEVAL_CONFIG['chunk_overlap']} tokens")

        # Load system
        with st.spinner("Loading RAG system..."):
            hybrid_retriever, llm_generator = load_system()
        st.success("System loaded successfully!")

    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Evaluation", "ℹ️ About"])

    with tab1:
        st.markdown('<h2 class="sub-header">Ask a Question</h2>',
                    unsafe_allow_html=True)

        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is artificial intelligence and how does it work?"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            dense_only = st.checkbox("Dense Only", value=False)
        with col2:
            sparse_only = st.checkbox("Sparse Only", value=False)
        with col3:
            show_details = st.checkbox("Show Retrieval Details", value=True)

        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if query:
                with st.spinner("Processing your query..."):
                    start_time = time.time()

                    try:
                        # Retrieve chunks
                        if dense_only:
                            results = hybrid_retriever.dense_retriever.search(query, k=retrieval_k)
                            method = "Dense Only"
                        elif sparse_only:
                            results = hybrid_retriever.sparse_retriever.search(query, k=retrieval_k)
                            method = "Sparse Only"
                        else:
                            results = hybrid_retriever.retrieve(
                                query,
                                top_k=retrieval_k,
                                top_n=fusion_n
                            )
                            method = "Hybrid (RRF)"

                        retrieval_time = time.time() - start_time

                        # Generate answer
                        context_chunks = results[:fusion_n]
                        answer_response = llm_generator.generate_answer(query, context_chunks)
                        generation_time = time.time() - start_time - retrieval_time

                        total_time = time.time() - start_time

                        # Display answer
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.markdown("### 💡 Answer")
                        st.write(answer_response.get("answer", "No answer generated."))

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieval Method", method)
                        with col2:
                            st.metric("Total Time", f"{total_time:.2f}s")
                        with col3:
                            st.metric("Chunks Used", len(context_chunks))

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Show retrieval details if requested
                        if show_details and results:
                            st.markdown('<h3 class="sub-header">📋 Retrieved Chunks</h3>',
                                        unsafe_allow_html=True)

                            for i, chunk in enumerate(results[:5], 1):
                                with st.expander(f"Chunk {i}: {chunk.get('title', 'No title')}"):
                                    st.write(f"**Source:** {chunk.get('url', 'No URL')}")
                                    st.write(f"**Score:** {chunk.get('score', 'N/A'):.4f}")
                                    if 'rrf_score' in chunk:
                                        st.write(f"**RRF Score:** {chunk.get('rrf_score', 'N/A'):.4f}")
                                    st.write(f"**Text:**")
                                    st.write(chunk.get('text', '')[:300] + "...")

                        # Display metrics in columns
                        if not dense_only and not sparse_only and 'rrf_score' in results[0]:
                            st.markdown('<h3 class="sub-header">📊 Fusion Scores</h3>',
                                        unsafe_allow_html=True)

                            # Create score comparison
                            score_data = []
                            for i, chunk in enumerate(results[:fusion_n], 1):
                                score_data.append({
                                    "Rank": i,
                                    "Title": chunk.get('title', 'N/A')[:50],
                                    "Dense Score": chunk.get('dense_score', 0),
                                    "Sparse Score": chunk.get('sparse_score', 0),
                                    "RRF Score": chunk.get('rrf_score', 0)
                                })

                            df_scores = pd.DataFrame(score_data)
                            st.dataframe(df_scores, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a question.")

    with tab2:
        st.markdown('<h2 class="sub-header">System Evaluation</h2>',
                    unsafe_allow_html=True)

        # Load evaluation results if available
        eval_path = FILE_PATHS["evaluation_results"]

        if eval_path.exists():
            try:
                # Load evaluation data
                with open(eval_path.with_suffix('.json'), 'r') as f:
                    eval_data = json.load(f)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "MRR Score",
                        f"{eval_data['mrr']['mean_mrr']:.4f}",
                        help="Mean Reciprocal Rank at URL level"
                    )

                with col2:
                    st.metric(
                        "BLEU Score",
                        f"{eval_data['bleu']['mean_bleu']:.4f}",
                        help="Answer quality compared to ground truth"
                    )

                with col3:
                    st.metric(
                        "Context Relevance",
                        f"{eval_data['custom_metrics']['context_relevance']['mean']:.4f}",
                        help="How relevant retrieved context is to query"
                    )

                with col4:
                    st.metric(
                        "Success Rate",
                        f"{eval_data['successful_answers'] / eval_data['total_questions'] * 100:.1f}%",
                        help="Percentage of successful answer generations"
                    )

                # Show detailed results
                st.markdown("### 📈 Detailed Results")

                # Convert to dataframe for display
                details_df = pd.DataFrame(eval_data['details'])
                if not details_df.empty:
                    # Select columns to display
                    display_cols = ['question_id', 'question', 'answer_success']
                    available_cols = [c for c in display_cols if c in details_df.columns]

                    st.dataframe(
                        details_df[available_cols].head(20),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Show distribution of scores
                    st.markdown("### 📊 Score Distributions")

                    col1, col2 = st.columns(2)

                    with col1:
                        if 'mrr' in eval_data:
                            mrr_scores = eval_data['mrr']['mrr_scores']
                            st.bar_chart(pd.DataFrame({'MRR': mrr_scores}))

                    with col2:
                        if 'bleu' in eval_data:
                            bleu_scores = eval_data['bleu']['bleu_scores']
                            st.bar_chart(pd.DataFrame({'BLEU': bleu_scores}))

            except Exception as e:
                st.error(f"Error loading evaluation data: {e}")
        else:
            st.info("No evaluation results found. Run the evaluation pipeline first.")

            if st.button("🔄 Run Evaluation", type="secondary"):
                with st.spinner("Running evaluation..."):
                    # This would run the evaluation pipeline
                    st.info("Evaluation would run here. Use the pipeline.py script for full evaluation.")

    with tab3:
        st.markdown('<h2 class="sub-header">About This System</h2>',
                    unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Objective
        This Hybrid RAG system combines multiple retrieval methods to provide accurate answers from Wikipedia content.

        ### 🔧 Technical Components

        #### 1. **Data Ingestion**
        - 500 Wikipedia articles (200 fixed + 300 random)
        - Text cleaning and chunking (200-400 tokens with 50-token overlap)

        #### 2. **Retrieval Methods**
        - **Dense Retrieval**: Sentence transformers + FAISS index
        - **Sparse Retrieval**: BM25 algorithm
        - **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)

        #### 3. **Answer Generation**
        - Open-source LLM (FLAN-T5)
        - Context-aware prompting

        #### 4. **Evaluation Framework**
        - 100 generated question-answer pairs
        - Multiple metrics: MRR, BLEU, Context Relevance, etc.
        - Automated evaluation pipeline

        ### 📊 Key Metrics
        - **MRR**: Measures how quickly correct source is found
        - **BLEU**: Measures answer quality vs ground truth
        - **Context Relevance**: Measures retrieval quality
        - **Answer Specificity**: Measures answer grounding in context

        ### 🚀 How to Use
        1. Enter a question in the Query tab
        2. Choose retrieval method (Hybrid, Dense-only, or Sparse-only)
        3. View answer and retrieval details
        4. Check evaluation results in the Evaluation tab
        """)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Streamlit Web Interface for Hybrid RAG System - Robust Version
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #F59E0B;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)


def get_default_config():
    """Get default configuration"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "google/flan-t5-base",
        "chunk_size": 400,
        "chunk_overlap": 50,
        "top_k": 10,
        "top_n": 5,
        "rrf_k": 60,
        "file_paths": {
            "faiss_index": DATA_DIR / "faiss_index.bin",
            "bm25_index": DATA_DIR / "bm25_index.pkl",
            "metadata": DATA_DIR / "chunk_metadata.json",
            "corpus_chunks": DATA_DIR / "corpus_chunks.json",
            "evaluation_results": DATA_DIR / "evaluation_results.json",
            "evaluation_csv": DATA_DIR / "evaluation_results.csv"
        }
    }


def check_system_files():
    """Check if required system files exist"""
    config = get_default_config()
    file_paths = config["file_paths"]

    required_files = ["faiss_index", "bm25_index", "metadata"]
    missing_files = []

    for file_key in required_files:
        file_path = file_paths.get(file_key)
        if file_path and not file_path.exists():
            missing_files.append(file_key)

    return missing_files, config


@st.cache_resource
def load_system_components():
    """Load the Hybrid RAG system components"""
    config = get_default_config()

    # Check if indices exist first
    missing_files, _ = check_system_files()
    if missing_files:
        return None, None, config

    try:
        # Import components
        from src.indexing.dense_index import DenseIndexer
        from src.indexing.sparse_index import SparseIndexer
        from src.indexing.hybrid_rrf import HybridRetriever
        from src.generation.llm_generator import LLMGenerator

        file_paths = config["file_paths"]

        # Load dense index
        dense_retriever = DenseIndexer(config["embedding_model"])
        dense_retriever.load_index(
            file_paths["faiss_index"],
            file_paths["metadata"]
        )

        # Load sparse index
        sparse_retriever = SparseIndexer()
        sparse_retriever.load_index(
            file_paths["bm25_index"],
            file_paths["metadata"]
        )

        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            k=config["rrf_k"]
        )

        # Load LLM generator
        llm_generator = LLMGenerator(model_name=config["llm_model"])

        return hybrid_retriever, llm_generator, config

    except ImportError as e:
        st.error(f"Import error: {e}")
        return None, None, config
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, config


def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("### 📊 System Status")

    missing_files, config = check_system_files()

    if missing_files:
        st.sidebar.error(f"Missing: {', '.join(missing_files)}")
        st.sidebar.info("Run pipeline first:")
        st.sidebar.code("python src/pipeline.py --mode full")
        return False, config
    else:
        st.sidebar.success("All files found!")
        return True, config


def display_query_tab(hybrid_retriever, llm_generator, config):
    """Display the query tab"""
    st.markdown('<h2 class="sub-header">🔍 Ask a Question</h2>', unsafe_allow_html=True)

    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What is artificial intelligence and how does it work?",
        key="query_input"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["Hybrid (RRF)", "Dense Only", "Sparse Only"],
            index=0
        )
    with col2:
        top_k = st.slider("Top K results", 3, 20, 10)
    with col3:
        top_n = st.slider("Final chunks", 2, 10, 5)

    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if query:
            with st.spinner("Processing your query..."):
                start_time = time.time()

                try:
                    # Retrieve chunks
                    if retrieval_method == "Dense Only":
                        results = hybrid_retriever.dense_retriever.search(query, k=top_k)
                        method = "Dense Only"
                    elif retrieval_method == "Sparse Only":
                        results = hybrid_retriever.sparse_retriever.search(query, k=top_k)
                        method = "Sparse Only"
                    else:
                        results = hybrid_retriever.retrieve(
                            query,
                            top_k=top_k,
                            top_n=top_n
                        )
                        method = "Hybrid (RRF)"

                    retrieval_time = time.time() - start_time

                    if not results:
                        st.warning("No results found for your query.")
                        return

                    # Generate answer
                    context_chunks = results[:top_n]
                    answer_response = llm_generator.generate_answer(query, context_chunks)
                    generation_time = time.time() - start_time - retrieval_time

                    total_time = time.time() - start_time

                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 Answer")
                    answer = answer_response.get("answer", "No answer generated.")
                    st.write(answer)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retrieval Method", method)
                    with col2:
                        st.metric("Total Time", f"{total_time:.2f}s")
                    with col3:
                        st.metric("Chunks Used", len(context_chunks))

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show retrieved chunks
                    with st.expander("📋 View Retrieved Chunks"):
                        for i, chunk in enumerate(results[:5], 1):
                            st.markdown(f"**Chunk {i}:** {chunk.get('title', 'No title')}")
                            st.caption(f"Source: {chunk.get('url', 'No URL')}")
                            if 'score' in chunk:
                                st.caption(f"Score: {chunk.get('score', 'N/A'):.4f}")
                            if 'rrf_score' in chunk:
                                st.caption(f"RRF Score: {chunk.get('rrf_score', 'N/A'):.4f}")
                            st.write(chunk.get('text', '')[:200] + "...")
                            st.divider()

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question.")


def display_evaluation_tab(config):
    """Display the evaluation tab"""
    if config is None:
        config = get_default_config()

    st.markdown('<h2 class="sub-header">📊 System Evaluation</h2>', unsafe_allow_html=True)

    file_paths = config["file_paths"]

    # Check if evaluation results exist
    eval_files_exist = False
    eval_data = None

    # Try JSON first
    if file_paths["evaluation_results"].exists():
        try:
            with open(file_paths["evaluation_results"], 'r') as f:
                eval_data = json.load(f)
            eval_files_exist = True
        except:
            pass

    # Try CSV if JSON doesn't exist or failed
    if not eval_files_exist and file_paths["evaluation_csv"].exists():
        try:
            import pandas as pd
            df = pd.read_csv(file_paths["evaluation_csv"])
            eval_data = df.to_dict('records')
            eval_files_exist = True
        except:
            pass

    if eval_files_exist and eval_data:
        try:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if isinstance(eval_data, list):
                    total = len(eval_data)
                elif isinstance(eval_data, dict) and 'details' in eval_data:
                    total = len(eval_data['details'])
                else:
                    total = 1
                st.metric("Total Questions", total)

            with col2:
                if isinstance(eval_data, list):
                    successful = sum(1 for r in eval_data if isinstance(r, dict) and r.get("success", False))
                elif isinstance(eval_data, dict) and 'successful_answers' in eval_data:
                    successful = eval_data['successful_answers']
                else:
                    successful = 0
                st.metric("Successful Answers", successful)

            with col3:
                if total > 0:
                    success_rate = (successful / total) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                else:
                    st.metric("Success Rate", "0%")

            with col4:
                if isinstance(eval_data, dict) and 'mrr' in eval_data and 'mean_mrr' in eval_data['mrr']:
                    mrr = eval_data['mrr']['mean_mrr']
                    st.metric("MRR Score", f"{mrr:.4f}")
                else:
                    st.metric("MRR Score", "N/A")

            # Show detailed results
            st.markdown("### 📈 Detailed Results")

            if isinstance(eval_data, list):
                # Convert to dataframe for display
                try:
                    import pandas as pd
                    display_data = []
                    for r in eval_data[:20]:  # Show first 20
                        if isinstance(r, dict):
                            display_data.append({
                                "Question": r.get("question", "")[:50] + "..." if len(
                                    r.get("question", "")) > 50 else r.get("question", ""),
                                "Answer": r.get("generated_answer", "")[:50] + "..." if len(
                                    r.get("generated_answer", "")) > 50 else r.get("generated_answer", ""),
                                "Success": "✓" if r.get("success", False) else "✗"
                            })

                    if display_data:
                        df = pd.DataFrame(display_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                except:
                    # Fallback to simple display
                    for i, r in enumerate(eval_data[:10]):
                        if isinstance(r, dict):
                            st.write(f"**Q{i + 1}:** {r.get('question', '')[:100]}...")
                            st.write(f"**A:** {r.get('generated_answer', '')[:100]}...")
                            st.write(f"**Success:** {'✓' if r.get('success') else '✗'}")
                            st.divider()

        except Exception as e:
            st.error(f"Error displaying evaluation data: {e}")
    else:
        st.info("No evaluation results found. Run the evaluation pipeline first.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Run Quick Test", type="secondary"):
                st.info("Use the pipeline script to run evaluation:")
                st.code("python src/pipeline.py --mode evaluate --questions 10")

        with col2:
            if st.button("📋 Check Files", type="secondary"):
                missing, _ = check_system_files()
                if missing:
                    st.error(f"Missing files: {', '.join(missing)}")
                else:
                    st.success("All system files found!")


def display_about_tab():
    """Display the about tab"""
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Objective
    This Hybrid RAG system combines multiple retrieval methods to provide accurate answers from Wikipedia content.

    ### 🔧 Technical Components

    #### 1. **Data Ingestion**
    - Wikipedia articles (200 fixed + 300 random)
    - Text cleaning and chunking
    - Token-based chunking with overlap

    #### 2. **Retrieval Methods**
    - **Dense Retrieval**: Sentence transformers + FAISS index
    - **Sparse Retrieval**: BM25 algorithm
    - **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)

    #### 3. **Answer Generation**
    - Open-source LLM (FLAN-T5)
    - Context-aware prompting

    #### 4. **Evaluation Framework**
    - Generated question-answer pairs
    - Multiple evaluation metrics

    ### 🚀 Quick Start Guide

    1. **Set up the system:**
    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Run full pipeline
    python src/pipeline.py --mode full --questions 20
    ```

    2. **Use the web interface:** (you're here!)

    3. **Run evaluations:**
    ```bash
    python src/pipeline.py --mode evaluate --questions 50
    ```
    """)

    # Quick diagnostics
    st.markdown("### 🔍 System Diagnostics")

    missing_files, config = check_system_files()

    if missing_files:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ System not fully set up")
        st.write(f"Missing files: {', '.join(missing_files)}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("**To fix:**")
        st.code("python src/pipeline.py --mode full")
    else:
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.success("✅ System files found")
        st.markdown('</div>', unsafe_allow_html=True)

        # Check if chunks exist
        if config["file_paths"]["corpus_chunks"].exists():
            try:
                with open(config["file_paths"]["corpus_chunks"], 'r') as f:
                    chunks = json.load(f)
                st.info(f"📊 Corpus: {len(chunks)} chunks loaded")
            except:
                st.warning("Corpus file exists but could not be read")
        else:
            st.warning("Corpus file not found")


def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">🔍 Hybrid RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Combining Dense + Sparse Retrieval with Reciprocal Rank Fusion")

    # Initialize session state for system status
    if 'system_loaded' not in st.session_state:
        st.session_state.system_loaded = False
    if 'system_config' not in st.session_state:
        st.session_state.system_config = None

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.write("**Models:**")
        st.write("- Embedding: all-MiniLM-L6-v2")
        st.write("- LLM: FLAN-T5-base")

        st.write("**Retrieval:**")
        st.write("- Chunk size: 400 tokens")
        st.write("- Overlap: 50 tokens")
        st.write("- RRF constant: 60")

        st.markdown("---")

        # System status check
        system_ready, config = display_system_status()
        st.session_state.system_config = config

        if system_ready:
            # Load system components
            with st.spinner("Loading system components..."):
                hybrid_retriever, llm_generator, config = load_system_components()

                if hybrid_retriever and llm_generator:
                    st.session_state.system_loaded = True
                    st.session_state.hybrid_retriever = hybrid_retriever
                    st.session_state.llm_generator = llm_generator
                    st.success("System loaded!")
                else:
                    st.session_state.system_loaded = False
                    st.warning("Could not load system components")
        else:
            st.session_state.system_loaded = False

        st.markdown("---")
        st.markdown("### 🛠️ Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Files", type="secondary"):
                missing, _ = check_system_files()
                if missing:
                    st.error(f"Missing: {', '.join(missing)}")
                else:
                    st.success("All files found!")

        with col2:
            if st.button("Refresh", type="secondary"):
                st.rerun()

    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Evaluation", "ℹ️ About"])

    with tab1:
        if st.session_state.system_loaded:
            display_query_tab(
                st.session_state.hybrid_retriever,
                st.session_state.llm_generator,
                st.session_state.system_config
            )
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("⚠️ System not ready for queries")
            st.markdown("""
            **Please complete these steps first:**

            1. **Run the pipeline:**
            ```bash
            python src/pipeline.py --mode full --questions 20
            ```

            2. **Or run step by step:**
            ```bash
            # Step 1: Fetch data
            python src/pipeline.py --mode ingest

            # Step 2: Build indices  
            python src/pipeline.py --mode index

            # Step 3: Test (optional)
            python src/pipeline.py --mode evaluate --questions 10
            ```

            3. **Refresh this page** after pipeline completes
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        display_evaluation_tab(st.session_state.system_config)

    with tab3:
        display_about_tab()


if __name__ == "__main__":
    main()
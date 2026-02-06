# Group 85: Hybrid RAG System with Automated Evaluation

## 📖 Overview
A hybrid Retrieval-Augmented Generation system combining dense vector retrieval (sentence transformers), sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from 500 Wikipedia articles.

### Key Components:
- **Hybrid Retrieval**: Integrates **FAISS** (Dense) and **BM25** (Sparse) via **Reciprocal Rank Fusion (RRF)**.
- **Local Generation**: Uses **Qwen-2.5-0.5B-Instruct** for private, local-first inference.
- **Automated Benchmarking**: A comprehensive evaluation suite measuring MRR, Hit Rate, and Lexical Overlap.
- **Automated Evaluation**: 100 generated questions with MRR and custom metrics
- **Interactive UI**: Streamlit dashboard
- **One-command Pipeline**: Full automation from indexing to evaluation

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- 8GB RAM (Minimum) | 16GB (Recommended)
- [Optional] CUDA-enabled GPU for faster inference.

### 2. Environment Setup instructions
1) git clone -b develop https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG.git
2) Open a terminal and create virtual environment python -m venv venv 
3) Activate it:
     Windows: .\venv\Scripts\activate 
     Mac/Linux: source venv/bin/activate
4) pip install -r requirements.txt
5) Create a .env file and place in project root in which provide the Hugging face token under HF_TOKEN
6) streamlit run app/streamlit_app.py

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

 ### Important links:
 1) GitHub Repository: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG
 2) Fixed 200 Wikipedia URLs: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/fixed_urls.json
 3) Random 300 Wikipedia URLs: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/random_urls.json
 4) 100 Questions generated from 500 Wikipedia URLs: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/questions_100.json
 5) Corpus chunks: https://raw.githubusercontent.com/2024aa05152-collab/Group_85_Hybrid_RAG/refs/heads/develop/data/corpus_chunks.json
 6) Evaluation results: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/tree/develop/outputs

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- 8GB RAM (Minimum) | 16GB (Recommended)
- ~2GB storage for model weights and indices
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

## requirements
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
html2text>=2020.1.16
wikipedia-api
pandas>=2.0.0
numpy>=1.24.0
tiktoken
tqdm>=4.65.0
nltk>=3.8.0
faiss-cpu>=1.7.0
rank-bm25>=0.2.2
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
transformers>=4.34.0
torch>=2.0.0
accelerate>=0.21.0
huggingface_hub
openai>=0.28.0
python-dotenv
ragas>=0.0.22
datasets>=2.14.0
rouge-score>=0.1.2
evaluate>=0.4.0
streamlit>=1.28.0

## 📊 System Architecture
Embedding Model: all-MiniLM-L6-v2 (Dense)
Keyword Engine: BM25Okapi (Sparse)
Fusion: Reciprocal Rank Fusion (RRF) with k=60
LLM: Qwen2.5-0.5B-Instruct (Local Inference)

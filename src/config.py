import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
APP_DIR = BASE_DIR / "app"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUTS_DIR, REPORTS_DIR, APP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "google/flan-t5-base",
    "generation_max_length": 512,
    "temperature": 0.7,
    "top_p": 0.95,
}

# Retrieval configurations
RETRIEVAL_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 50,
    "top_k": 10,
    "top_n": 5,
    "rrf_k": 60,
}

# Evaluation configurations
EVALUATION_CONFIG = {
    "num_questions": 100,
    "question_types": ["factual", "comparative", "inferential", "multi-hop"],
    "metrics": ["mrr", "bleu", "context_relevance"],
}

# File paths
FILE_PATHS = {
    "fixed_urls": DATA_DIR / "fixed_urls.json",
    "random_urls": DATA_DIR / "random_urls.json",
    "corpus_chunks": DATA_DIR / "corpus_chunks.json",
    "questions": DATA_DIR / "questions_100.json",
    "evaluation_results": DATA_DIR / "evaluation_results.csv",
    "faiss_index": DATA_DIR / "faiss_index.bin",
    "bm25_index": DATA_DIR / "bm25_index.pkl",
    "metadata": DATA_DIR / "chunk_metadata.json",
}

# Export all constants
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'SRC_DIR', 'OUTPUTS_DIR', 'REPORTS_DIR', 'APP_DIR',
    'MODEL_CONFIG', 'RETRIEVAL_CONFIG', 'EVALUATION_CONFIG', 'FILE_PATHS'
]
from rank_bm25 import BM25Okapi
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseIndexer:
    """Create and manage BM25 index"""

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        tokens = word_tokenize(text.lower())
        # Remove very short tokens and punctuation
        tokens = [t for t in tokens if len(t) > 2 and t.isalnum()]
        return tokens

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build BM25 index from chunks"""
        self.chunks = chunks

        # Tokenize all chunks
        self.tokenized_chunks = [self.tokenize(chunk["text"]) for chunk in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        logger.info(f"Built BM25 index with {len(self.chunks)} documents")

    def save_index(self, index_path: Path, metadata_path: Path):
        """Save BM25 index and metadata"""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Save BM25 index
        with open(index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_chunks': self.tokenized_chunks
            }, f)

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.chunks, f, indent=2)

        logger.info(f"Saved BM25 index to {index_path}")

    def load_index(self, index_path: Path, metadata_path: Path):
        """Load BM25 index and metadata"""
        # Load BM25 index
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_chunks = data['tokenized_chunks']

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.chunks = json.load(f)

        logger.info(f"Loaded BM25 index with {len(self.chunks)} documents")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search index with query"""
        if self.bm25 is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        # Tokenize query
        tokenized_query = self.tokenize(query)

        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = scores.argsort()[::-1][:k]

        # Prepare results
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "url": chunk["url"],
                "title": chunk["title"],
                "score": float(scores[idx]),  # BM25 score
                "rank": rank
            })

        return results
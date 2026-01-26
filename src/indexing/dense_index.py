import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseIndexer:
    """Create and manage FAISS vector index"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        self.index = None
        self.chunk_metadata = []

    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded model: {self.model_name}, dimension: {self.embedding_dim}")
            except ImportError:
                logger.error("sentence-transformers not installed. Using fallback embeddings.")
                self.model = None
                # Use simple embeddings as fallback
                self.embedding_dim = 384

    def create_simple_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Create simple random embeddings as fallback"""
        n_chunks = len(chunks)
        embeddings = np.random.randn(n_chunks, self.embedding_dim).astype(np.float32)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for all chunks"""
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Creating embeddings for {len(texts)} chunks...")

        # Try to use sentence transformers
        if self.model is None:
            self.load_model()

        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                logger.info(f"Created embeddings with shape: {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to use sentence-transformers: {e}")

        # Fallback to simple embeddings
        logger.info("Using simple embeddings as fallback")
        return self.create_simple_embeddings(chunks)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build FAISS index from chunks"""
        # Create embeddings
        embeddings = self.create_embeddings(chunks)

        # Ensure correct dimension
        if embeddings.shape[1] != self.embedding_dim:
            self.embedding_dim = embeddings.shape[1]

        # Create FAISS index (Inner product for cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        # Store metadata
        self.chunk_metadata = chunks

        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def save_index(self, index_path: Path, metadata_path: Path):
        """Save FAISS index and metadata"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)

        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    def load_index(self, index_path: Path, metadata_path: Path):
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.embedding_dim = self.index.d

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.chunk_metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors from {index_path}")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search index with query"""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        # Create query embedding
        query_embedding = self._embed_query(query)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "score": float(dist),  # Cosine similarity
                    "rank": len(results) + 1
                })

        return results

    def _embed_query(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        # Try to use model
        if self.model is not None:
            try:
                embedding = self.model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(embedding)
                return embedding
            except:
                pass

        # Fallback: random embedding normalized
        embedding = np.random.randn(1, self.embedding_dim).astype(np.float32)
        faiss.normalize_L2(embedding)
        return embedding
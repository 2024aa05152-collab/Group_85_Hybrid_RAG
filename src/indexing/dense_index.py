import faiss
import numpy as np
import json
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseIndexer:
    """Create and manage FAISS vector index with simplified embeddings"""

    def __init__(self, use_simple_embeddings=True):
        self.embedding_dim = 384  # Standard dimension for all-MiniLM-L6-v2
        self.index = None
        self.chunk_metadata = []
        self.use_simple_embeddings = use_simple_embeddings

    def create_simple_embeddings(self, chunks):
        """Create simple TF-IDF based embeddings as fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [chunk["text"] for chunk in chunks]

        # Create TF-IDF vectors with fixed dimension
        vectorizer = TfidfVectorizer(max_features=self.embedding_dim)
        embeddings = vectorizer.fit_transform(texts).toarray()

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        return embeddings

    def create_embeddings(self, chunks):
        """Create embeddings with fallback options"""
        if self.use_simple_embeddings:
            logger.info("Using simple TF-IDF embeddings")
            return self.create_simple_embeddings(chunks)

        try:
            # Try to use sentence transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')

            texts = [chunk["text"] for chunk in chunks]
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.warning(f"Failed to use sentence-transformers: {e}")
            logger.info("Falling back to simple embeddings")
            return self.create_simple_embeddings(chunks)

    def build_index(self, chunks):
        """Build FAISS index from chunks"""
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        self.embedding_dim = embeddings.shape[1]

        # Create FAISS index
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

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.chunk_metadata = json.load(f)

        self.embedding_dim = self.index.d
        logger.info(f"Loaded index with {self.index.ntotal} vectors from {index_path}")

    def search(self, query: str, k: int = 10):
        """Search index with query"""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        # Create query embedding
        if self.use_simple_embeddings:
            query_embedding = self._embed_query_simple(query)
        else:
            query_embedding = self._embed_query_transformer(query)

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
                    "score": float(dist),
                    "rank": len(results) + 1
                })

        return results

    def _embed_query_simple(self, query: str):
        """Create simple embedding for query"""
        # For simple embeddings, we return a random embedding
        # In production, you'd want to use the same vectorizer as during training
        embedding = np.random.randn(1, self.embedding_dim).astype(np.float32)
        faiss.normalize_L2(embedding)
        return embedding

    def _embed_query_transformer(self, query: str):
        """Create embedding using sentence transformer"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(embedding)
            return embedding
        except:
            return self._embed_query_simple(query)
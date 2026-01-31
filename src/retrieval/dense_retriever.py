import faiss
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, chunks: List[Dict]):
        self.metadata = chunks
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def save(self, path: str):
        if not os.path.exists(path): os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, "dense.index"))

    def load(self, path: str, chunks: List[Dict]):
        self.metadata = chunks
        self.index = faiss.read_index(os.path.join(path, "dense.index"))

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vec = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1:
                hit = self.metadata[idx].copy()
                hit.update({"dense_score": float(score), "dense_rank": rank + 1})
                results.append(hit)
        return results
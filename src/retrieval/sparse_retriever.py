import pickle
import os
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.metadata = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def build_index(self, chunks: List[Dict]):
        self.metadata = chunks
        tokenized_corpus = [self._tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, path: str):
        if not os.path.exists(path): os.makedirs(path)
        with open(os.path.join(path, "sparse.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self, path: str, chunks: List[Dict]):
        self.metadata = chunks
        with open(os.path.join(path, "sparse.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                hit = self.metadata[idx].copy()
                hit.update({"sparse_score": float(scores[idx]), "sparse_rank": rank + 1})
                results.append(hit)
        return results
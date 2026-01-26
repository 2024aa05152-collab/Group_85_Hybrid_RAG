import json, numpy as np
from rank_bm25 import BM25Okapi
from src.config import CORPUS_FILE, BM25_INDEX, TOP_K

def sparse_retrieve(query):
    corpus = json.load(open(CORPUS_FILE))
    tokenized = json.load(open(BM25_INDEX))
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(query.lower().split())
    idxs = np.argsort(scores)[::-1][:TOP_K]

    results = {}
    for rank, i in enumerate(idxs):
        results[corpus[i]["chunk_id"]] = {
            "rank": rank + 1,
            "score": float(scores[i]),
            "chunk": corpus[i]
        }
    return results

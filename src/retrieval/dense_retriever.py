import json, faiss
from sentence_transformers import SentenceTransformer
from src.config import CORPUS_FILE, FAISS_INDEX, EMBED_MODEL, TOP_K

def dense_retrieve(query):
    corpus = json.load(open(CORPUS_FILE))
    index = faiss.read_index(FAISS_INDEX)

    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True)

    scores, idxs = index.search(q_emb, TOP_K)

    results = {}
    for rank, i in enumerate(idxs[0]):
        results[corpus[i]["chunk_id"]] = {
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "chunk": corpus[i]
        }
    return results

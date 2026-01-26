from src.retrieval.dense_retriever import dense_retrieve
from src.retrieval.sparse_retriever import sparse_retrieve
from src.indexing.hybrid_rrf import rrf_fusion
from src.config import RRF_K, TOP_N

def hybrid_retrieve(query):
    dense = dense_retrieve(query)
    sparse = sparse_retrieve(query)

    fused = rrf_fusion(dense, sparse, RRF_K, TOP_N)

    final_chunks = []
    for cid, score in fused:
        chunk = dense.get(cid) or sparse.get(cid)
        chunk["rrf_score"] = score
        final_chunks.append(chunk)

    return final_chunks

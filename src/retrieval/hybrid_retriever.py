from typing import List, Dict

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever

    def retrieve(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict]:
        # Get results from both (retrieve more to ensure overlap)
        dense_hits = self.dense.retrieve(query, top_k=top_k * 2)
        sparse_hits = self.sparse.retrieve(query, top_k=top_k * 2)
        
        fused_scores = {} # key: chunk_id, value: {metadata + scores}

        # Process Dense
        for hit in dense_hits:
            cid = hit['chunk_id']
            fused_scores[cid] = hit
            fused_scores[cid]['rrf_score'] = 1.0 / (rrf_k + hit['dense_rank'])
            # Ensure sparse keys exist even if not found in sparse
            fused_scores[cid].setdefault('sparse_score', 0.0)

        # Process Sparse
        for hit in sparse_hits:
            cid = hit['chunk_id']
            if cid in fused_scores:
                fused_scores[cid]['rrf_score'] += 1.0 / (rrf_k + hit['sparse_rank'])
                fused_scores[cid]['sparse_score'] = hit['sparse_score']
            else:
                fused_scores[cid] = hit
                fused_scores[cid]['rrf_score'] = 1.0 / (rrf_k + hit['sparse_rank'])
                fused_scores[cid].setdefault('dense_score', 0.0)

        # Sort by RRF
        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x['rrf_score'], 
            reverse=True
        )
        return sorted_results[:top_k]
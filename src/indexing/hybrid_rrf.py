from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense and sparse retrieval using Reciprocal Rank Fusion (RRF)"""

    def __init__(self, dense_retriever, sparse_retriever, k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = k

    def reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]],
                               sparse_results: List[Dict[str, Any]],
                               top_n: int = 5) -> List[Dict[str, Any]]:
        """Combine results using RRF"""

        # Create mapping from chunk_id to RRF score
        rrf_scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "dense_score": result["score"],
                    "sparse_score": 0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "rrf_score": rrf_score,
                    "metadata": {
                        "text": result["text"],
                        "url": result["url"],
                        "title": result["title"]
                    }
                }
            else:
                rrf_scores[chunk_id]["dense_score"] = result["score"]
                rrf_scores[chunk_id]["dense_rank"] = rank
                rrf_scores[chunk_id]["rrf_score"] += rrf_score

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "dense_score": 0,
                    "sparse_score": result["score"],
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "rrf_score": rrf_score,
                    "metadata": {
                        "text": result["text"],
                        "url": result["url"],
                        "title": result["title"]
                    }
                }
            else:
                rrf_scores[chunk_id]["sparse_score"] = result["score"]
                rrf_scores[chunk_id]["sparse_rank"] = rank
                rrf_scores[chunk_id]["rrf_score"] += rrf_score

        # Convert to list and sort by RRF score
        combined_results = []
        for chunk_id, scores in rrf_scores.items():
            combined_results.append({
                "chunk_id": chunk_id,
                "text": scores["metadata"]["text"],
                "url": scores["metadata"]["url"],
                "title": scores["metadata"]["title"],
                "dense_score": scores["dense_score"],
                "sparse_score": scores["sparse_score"],
                "dense_rank": scores["dense_rank"],
                "sparse_rank": scores["sparse_rank"],
                "rrf_score": scores["rrf_score"]
            })

        # Sort by RRF score (descending)
        combined_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Return top N results
        return combined_results[:top_n]

    def retrieve(self, query: str, top_k: int = 10, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using hybrid approach"""
        logger.info(f"Retrieving for query: {query}")

        # Get dense results
        dense_results = self.dense_retriever.search(query, k=top_k)
        logger.info(f"Dense retrieval: {len(dense_results)} results")

        # Get sparse results
        sparse_results = self.sparse_retriever.search(query, k=top_k)
        logger.info(f"Sparse retrieval: {len(sparse_results)} results")

        # Combine using RRF
        hybrid_results = self.reciprocal_rank_fusion(dense_results, sparse_results, top_n)
        logger.info(f"Hybrid retrieval: {len(hybrid_results)} results after RRF")

        return hybrid_results
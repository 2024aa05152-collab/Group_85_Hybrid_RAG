from typing import List, Dict, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRREvaluator:
    """Calculate Mean Reciprocal Rank at URL level"""

    def __init__(self):
        pass

    def extract_url_from_chunk_id(self, chunk_id: str) -> str:
        """Extract base URL from chunk ID"""
        # chunk_id format: "url_chunk_index"
        parts = chunk_id.split('_')
        # Reconstruct URL (handling underscores in page titles)
        url_part = '_'.join(parts[:-1])
        return f"https://en.wikipedia.org/wiki/{url_part}"

    def calculate_mrr(self, ground_truth_urls: List[str],
                      retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate MRR at URL level

        Args:
            ground_truth_urls: List of correct Wikipedia URLs
            retrieved_chunks: List of retrieved chunks with metadata

        Returns:
            MRR score (float)
        """
        if not retrieved_chunks:
            return 0.0

        # Extract URLs from retrieved chunks
        retrieved_urls = []
        for chunk in retrieved_chunks:
            if 'url' in chunk:
                retrieved_urls.append(chunk['url'])
            else:
                # Fallback to extracting from chunk_id
                retrieved_urls.append(self.extract_url_from_chunk_id(chunk['chunk_id']))

        # Find the first occurrence of any ground truth URL
        for rank, url in enumerate(retrieved_urls, 1):
            if url in ground_truth_urls:
                return 1.0 / rank

        # No correct URL found
        return 0.0

    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate MRR for a batch of questions"""
        mrr_scores = []
        details = []

        for eval_item in evaluations:
            ground_truth_urls = eval_item.get("ground_truth_urls", [])
            retrieved_chunks = eval_item.get("retrieved_chunks", [])

            mrr_score = self.calculate_mrr(ground_truth_urls, retrieved_chunks)
            mrr_scores.append(mrr_score)

            details.append({
                "question_id": eval_item.get("question_id", ""),
                "question": eval_item.get("question", ""),
                "mrr": mrr_score,
                "ground_truth_urls": ground_truth_urls,
                "retrieved_urls": [chunk.get('url', '') for chunk in retrieved_chunks[:5]],
                "first_correct_rank": 1.0 / mrr_score if mrr_score > 0 else None
            })

        # Calculate statistics
        mean_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        std_mrr = np.std(mrr_scores) if len(mrr_scores) > 1 else 0.0

        result = {
            "mean_mrr": float(mean_mrr),
            "std_mrr": float(std_mrr),
            "mrr_scores": mrr_scores,
            "details": details,
            "summary": {
                "total_questions": len(evaluations),
                "questions_with_correct_url": sum(1 for score in mrr_scores if score > 0),
                "perfect_mrr_questions": sum(1 for score in mrr_scores if score == 1.0)
            }
        }

        logger.info(f"MRR Evaluation: Mean = {mean_mrr:.4f}, Std = {std_mrr:.4f}")
        return result
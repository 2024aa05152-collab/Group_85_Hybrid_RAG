from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BLEUEvaluator:
    """Calculate BLEU score for answer quality"""

    def __init__(self):
        self.smoothing = SmoothingFunction().method4

    def calculate_bleu(self, reference: str, candidate: str,
                       weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> float:
        """
        Calculate BLEU score

        Args:
            reference: Ground truth answer
            candidate: Generated answer
            weights: N-gram weights

        Returns:
            BLEU score (float)
        """
        # Tokenize
        ref_tokens = [word_tokenize(reference.lower())]
        can_tokens = word_tokenize(candidate.lower())

        # Calculate BLEU with smoothing
        try:
            score = sentence_bleu(
                ref_tokens,
                can_tokens,
                weights=weights,
                smoothing_function=self.smoothing
            )
            return score
        except:
            return 0.0

    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate BLEU for a batch of questions"""
        bleu_scores = []
        details = []

        for eval_item in evaluations:
            ground_truth = eval_item.get("ground_truth", "")
            generated_answer = eval_item.get("generated_answer", "")

            bleu_score = self.calculate_bleu(ground_truth, generated_answer)
            bleu_scores.append(bleu_score)

            details.append({
                "question_id": eval_item.get("question_id", ""),
                "question": eval_item.get("question", ""),
                "bleu_score": bleu_score,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer
            })

        # Calculate statistics
        mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        std_bleu = np.std(bleu_scores) if len(bleu_scores) > 1 else 0.0

        result = {
            "mean_bleu": float(mean_bleu),
            "std_bleu": float(std_bleu),
            "bleu_scores": bleu_scores,
            "details": details,
            "summary": {
                "total_questions": len(evaluations),
                "high_bleu": sum(1 for score in bleu_scores if score > 0.5),
                "medium_bleu": sum(1 for score in bleu_scores if 0.2 <= score <= 0.5),
                "low_bleu": sum(1 for score in bleu_scores if score < 0.2)
            }
        }

        logger.info(f"BLEU Evaluation: Mean = {mean_bleu:.4f}, Std = {std_bleu:.4f}")
        return result
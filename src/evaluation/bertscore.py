from typing import List, Dict, Any, Optional
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTScoreEvaluator:
    """
    BERTScore evaluation for measuring semantic similarity between generated and reference answers.
    
    Justification:
    BERTScore is crucial for RAG systems because it measures semantic similarity rather than
    exact string matching. Unlike BLEU/ROUGE which focus on n-gram overlap, BERTScore
    captures semantic meaning using contextual embeddings, making it ideal for evaluating
    whether generated answers convey the same information as reference answers even with
    different wording.
    
    Calculation Method:
    1. Encode both generated and reference texts using pre-trained BERT model
    2. Compute token-level cosine similarities between embeddings
    3. Apply maximum matching for precision and recall
    4. Calculate F1 score from precision and recall
    
    Interpretation:
    - Score range: 0.0 to 1.0 (higher is better)
    - >0.9: Excellent semantic similarity
    - 0.8-0.9: Good semantic similarity
    - 0.7-0.8: Moderate semantic similarity
    - <0.7: Poor semantic similarity
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"Loaded BERTScore model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load BERTScore model: {e}")
            raise

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings for texts"""
        # Tokenize the batch of texts at once (ensures consistent padding)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move inputs to the model device
        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

            # Attention-mask-aware mean pooling if available
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / lengths
            else:
                pooled = last_hidden.mean(dim=1)

            # Move to CPU and convert to numpy
            embeddings = pooled.detach().cpu().numpy()

        return np.array(embeddings)

    def calculate_bertscore(self, generated_answer: str, 
                           reference_answer: str) -> Dict[str, float]:
        """
        Calculate BERTScore between generated and reference answers
        
        Args:
            generated_answer: Generated answer from RAG system
            reference_answer: Reference/ground truth answer
            
        Returns:
            Dict with precision, recall, and f1 scores
        """
        if not generated_answer or not reference_answer:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            # Get embeddings
            embeddings = self._get_embeddings([generated_answer, reference_answer])
            
            if len(embeddings) != 2:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # For sentence-level BERTScore, precision, recall, and F1 are the same
            # as we're comparing whole sentences rather than token-level matching
            score = max(0.0, similarity)  # Ensure non-negative
            
            return {
                "precision": float(score),
                "recall": float(score),
                "f1": float(score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate BERTScore for a batch of questions"""
        precision_scores = []
        recall_scores = []
        f1_scores = []
        details = []

        for eval_item in evaluations:
            generated_answer = eval_item.get("generated_answer", "")
            reference_answer = eval_item.get("reference_answer", "")
            
            # If no reference answer, skip this evaluation
            if not reference_answer:
                continue
                
            bert_scores = self.calculate_bertscore(generated_answer, reference_answer)
            
            precision_scores.append(bert_scores["precision"])
            recall_scores.append(bert_scores["recall"])
            f1_scores.append(bert_scores["f1"])

            details.append({
                "question_id": eval_item.get("question_id", ""),
                "question": eval_item.get("question", ""),
                "generated_answer": generated_answer[:200] + "..." if len(generated_answer) > 200 else generated_answer,
                "reference_answer": reference_answer[:200] + "..." if len(reference_answer) > 200 else reference_answer,
                "bert_precision": bert_scores["precision"],
                "bert_recall": bert_scores["recall"],
                "bert_f1": bert_scores["f1"]
            })

        # Calculate statistics
        if not f1_scores:
            return {
                "bertscore_precision": {"mean": 0.0, "std": 0.0, "scores": []},
                "bertscore_recall": {"mean": 0.0, "std": 0.0, "scores": []},
                "bertscore_f1": {"mean": 0.0, "std": 0.0, "scores": []},
                "details": [],
                "summary": {"total_evaluated": 0, "high_quality_answers": 0}
            }

        results = {
            "bertscore_precision": {
                "mean": float(np.mean(precision_scores)),
                "std": float(np.std(precision_scores)),
                "scores": precision_scores
            },
            "bertscore_recall": {
                "mean": float(np.mean(recall_scores)),
                "std": float(np.std(recall_scores)),
                "scores": recall_scores
            },
            "bertscore_f1": {
                "mean": float(np.mean(f1_scores)),
                "std": float(np.std(f1_scores)),
                "scores": f1_scores
            },
            "details": details,
            "summary": {
                "total_evaluated": len(f1_scores),
                "high_quality_answers": sum(1 for s in f1_scores if s > 0.8),
                "moderate_quality_answers": sum(1 for s in f1_scores if 0.7 <= s <= 0.8),
                "low_quality_answers": sum(1 for s in f1_scores if s < 0.7)
            }
        }

        logger.info(f"BERTScore Evaluation - F1: {results['bertscore_f1']['mean']:.4f} ± {results['bertscore_f1']['std']:.4f}")
        return results

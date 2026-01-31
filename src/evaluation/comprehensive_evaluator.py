from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

from .mrr import MRREvaluator
from .bertscore import BERTScoreEvaluator
from .custom_metrics import CustomMetricsEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation coordinator for RAG system
    
    Combines all evaluation metrics:
    1. MRR (Mean Reciprocal Rank) - URL Level [Mandatory - 2 Marks]
    2. BERTScore for Answer Quality [Custom Metric 1 - 2 Marks] 
    3. Precision@K for Retrieval Quality [Custom Metric 2 - 2 Marks]
    
    Additional metrics for comprehensive analysis:
    - Context Relevance
    - Answer Specificity  
    - Retrieval Diversity
    - Hit Rate@K
    """
    
    def __init__(self):
        self.mrr_evaluator = MRREvaluator()
        try:
            self.bert_evaluator = BERTScoreEvaluator()
        except Exception as e:
            logger.warning(f"BERTScore evaluator failed to initialize: {e}")
            self.bert_evaluator = None
        self.custom_evaluator = CustomMetricsEvaluator()
    
    def evaluate_system(self, evaluations: List[Dict[str, Any]], 
                       save_results: bool = True,
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the RAG system
        
        Args:
            evaluations: List of evaluation items containing:
                - question: The query
                - ground_truth_urls: List of correct Wikipedia URLs
                - retrieved_chunks: List of retrieved document chunks
                - generated_answer: Generated answer from RAG system
                - reference_answer: Reference/ground truth answer (optional)
                - context_chunks: Context chunks used for generation
            save_results: Whether to save results to file
            output_file: Custom output file path
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive RAG evaluation...")
        
        # 1. Mandatory Metric: MRR at URL Level (2 Marks)
        logger.info("Evaluating MRR (Mean Reciprocal Rank) at URL level...")
        mrr_results = self.mrr_evaluator.evaluate_batch(evaluations)
        
        # 2. Custom Metric 1: BERTScore for Answer Quality (2 Marks)
        bert_results = {}
        if self.bert_evaluator:
            logger.info("Evaluating BERTScore for answer quality...")
            bert_results = self.bert_evaluator.evaluate_batch(evaluations)
        else:
            logger.warning("BERTScore evaluation skipped due to initialization failure")
        
        # 3. Custom Metric 2 + Additional: Custom Metrics including Precision@K (2 Marks)
        logger.info("Evaluating custom metrics (Precision@K, Context Relevance, etc.)...")
        custom_results = self.custom_evaluator.evaluate_batch(evaluations)
        
        # Combine all results
        comprehensive_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": len(evaluations),
            "metrics": {
                # Mandatory Metric (2 Marks)
                "mrr_url_level": mrr_results,
                
                # Custom Metrics (4 Marks Total)
                "bertscore_answer_quality": bert_results,  # 2 Marks
                "precision_k_retrieval_quality": {         # 2 Marks
                    "precision_at_5": custom_results.get("precision_at_5", {}),
                    "precision_at_10": custom_results.get("precision_at_10", {}),
                    "hit_rate_at_10": custom_results.get("hit_rate_at_10", {}),
                },
                
                # Additional Analysis Metrics
                "additional_metrics": {
                    "context_relevance": custom_results.get("context_relevance", {}),
                    "answer_specificity": custom_results.get("answer_specificity", {}),
                    "retrieval_diversity": custom_results.get("retrieval_diversity", {}),
                }
            },
            "summary": self._generate_summary(mrr_results, bert_results, custom_results),
            "detailed_results": {
                "mrr_details": mrr_results.get("details", []),
                "bert_details": bert_results.get("details", []),
                "custom_details": custom_results.get("details", [])
            }
        }
        
        # Save results if requested
        if save_results:
            output_path = output_file or f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._save_results(comprehensive_results, output_path)
        
        self._log_summary(comprehensive_results)
        return comprehensive_results
    
    def _generate_summary(self, mrr_results: Dict, bert_results: Dict, 
                         custom_results: Dict) -> Dict[str, Any]:
        """Generate evaluation summary"""
        summary = {
            "overall_performance": "Excellent",  # Will be determined by scores
            "key_metrics": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Extract key metric scores
        mrr_score = mrr_results.get("mean_mrr", 0.0)
        bert_f1 = bert_results.get("bertscore_f1", {}).get("mean", 0.0) if bert_results else 0.0
        precision_5 = custom_results.get("precision_at_5", {}).get("mean", 0.0)
        precision_10 = custom_results.get("precision_at_10", {}).get("mean", 0.0)
        
        summary["key_metrics"] = {
            "mrr_url_level": round(mrr_score, 4),
            "bertscore_f1": round(bert_f1, 4),
            "precision_at_5": round(precision_5, 4),
            "precision_at_10": round(precision_10, 4)
        }
        
        # Determine overall performance
        avg_score = (mrr_score + bert_f1 + precision_5) / 3
        if avg_score >= 0.8:
            summary["overall_performance"] = "Excellent"
        elif avg_score >= 0.6:
            summary["overall_performance"] = "Good"
        elif avg_score >= 0.4:
            summary["overall_performance"] = "Fair"
        else:
            summary["overall_performance"] = "Poor"
        
        # Identify strengths and weaknesses
        if mrr_score >= 0.7:
            summary["strengths"].append("Good URL-level retrieval (high MRR)")
        else:
            summary["weaknesses"].append("Poor URL-level retrieval (low MRR)")
            summary["recommendations"].append("Improve retrieval relevance and ranking")
        
        if bert_f1 >= 0.7:
            summary["strengths"].append("High semantic answer quality (high BERTScore)")
        else:
            summary["weaknesses"].append("Poor semantic answer quality (low BERTScore)")
            summary["recommendations"].append("Improve answer generation quality")
        
        if precision_5 >= 0.6:
            summary["strengths"].append("Good precision in top-5 retrievals")
        else:
            summary["weaknesses"].append("Low precision in top-5 retrievals")
            summary["recommendations"].append("Refine retrieval algorithm to improve precision")
        
        return summary
    
    def _save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
    
    def _log_summary(self, results: Dict):
        """Log evaluation summary"""
        summary = results.get("summary", {})
        metrics = summary.get("key_metrics", {})
        
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE RAG EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Performance: {summary.get('overall_performance', 'Unknown')}")
        logger.info(f"Total Questions Evaluated: {results.get('total_questions', 0)}")
        logger.info("")
        logger.info("Key Metrics:")
        logger.info(f"  • MRR (URL Level):           {metrics.get('mrr_url_level', 0.0):.4f}")
        logger.info(f"  • BERTScore F1:              {metrics.get('bertscore_f1', 0.0):.4f}")
        logger.info(f"  • Precision@5:               {metrics.get('precision_at_5', 0.0):.4f}")
        logger.info(f"  • Precision@10:              {metrics.get('precision_at_10', 0.0):.4f}")
        logger.info("")
        
        strengths = summary.get("strengths", [])
        if strengths:
            logger.info("Strengths:")
            for strength in strengths:
                logger.info(f"  ✓ {strength}")
        
        weaknesses = summary.get("weaknesses", [])
        if weaknesses:
            logger.info("Weaknesses:")
            for weakness in weaknesses:
                logger.info(f"  ✗ {weakness}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            logger.info("Recommendations:")
            for rec in recommendations:
                logger.info(f"  → {rec}")
        
        logger.info("=" * 60)


def create_sample_evaluation():
    """Create sample evaluation data for testing"""
    return [
        {
            "question_id": "q001",
            "question": "What is the capital of France?",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Paris"],
            "retrieved_chunks": [
                {
                    "chunk_id": "Paris_chunk_0",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "text": "Paris is the capital and most populous city of France...",
                    "score": 0.95
                },
                {
                    "chunk_id": "France_chunk_1", 
                    "url": "https://en.wikipedia.org/wiki/France",
                    "text": "France is a country in Western Europe...",
                    "score": 0.87
                }
            ],
            "context_chunks": [
                {
                    "chunk_id": "Paris_chunk_0",
                    "text": "Paris is the capital and most populous city of France...",
                }
            ],
            "generated_answer": "Paris is the capital of France.",
            "reference_answer": "The capital of France is Paris."
        }
    ]


if __name__ == "__main__":
    # Example usage
    evaluator = ComprehensiveEvaluator()
    sample_data = create_sample_evaluation()
    results = evaluator.evaluate_system(sample_data)
    print("Evaluation completed!")
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudy:
    """Perform ablation studies for RAG components"""

    def __init__(self, hybrid_system, dense_only=False, sparse_only=False):
        self.hybrid_system = hybrid_system
        self.results = {}

    def run_study(self, questions: List[Dict[str, Any]],
                  methods: List[str] = ["hybrid", "dense_only", "sparse_only"]) -> Dict[str, Any]:
        """
        Run ablation study comparing different retrieval methods

        Args:
            questions: List of evaluation questions
            methods: Which methods to compare

        Returns:
            Dictionary with comparison results
        """
        all_results = {}

        for method in methods:
            logger.info(f"Running ablation for method: {method}")
            method_results = self._evaluate_method(method, questions)
            all_results[method] = method_results

        # Compare methods
        comparison = self._compare_methods(all_results)

        # Save results
        self.results = {
            "methods": all_results,
            "comparison": comparison,
            "questions_analyzed": len(questions)
        }

        return self.results

    def _evaluate_method(self, method: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a specific method"""
        scores = {
            "mrr": [],
            "bleu": [],
            "context_relevance": [],
            "response_times": []
        }

        for q in questions:
            query = q["question"]

            try:
                # Time the retrieval
                import time
                start_time = time.time()

                if method == "hybrid":
                    results = self.hybrid_system.retrieve(query)
                elif method == "dense_only":
                    results = self.hybrid_system.dense_retriever.search(query, k=5)
                elif method == "sparse_only":
                    results = self.hybrid_system.sparse_retriever.search(query, k=5)
                else:
                    results = []

                response_time = time.time() - start_time

                # Generate answer if we have results
                if results:
                    answer_response = self.hybrid_system.generate_answer(query, results[:3])
                    answer = answer_response.get("answer", "")
                else:
                    answer = ""

                # Calculate metrics (simplified - in practice, use proper evaluators)
                mrr_score = self._calculate_simple_mrr(q.get("ground_truth_urls", []), results)
                bleu_score = self._calculate_simple_bleu(q.get("ground_truth", ""), answer)
                relevance_score = self._calculate_simple_relevance(query, results)

                scores["mrr"].append(mrr_score)
                scores["bleu"].append(bleu_score)
                scores["context_relevance"].append(relevance_score)
                scores["response_times"].append(response_time)

            except Exception as e:
                logger.error(f"Error evaluating {method} for query '{query}': {e}")
                scores["mrr"].append(0.0)
                scores["bleu"].append(0.0)
                scores["context_relevance"].append(0.0)
                scores["response_times"].append(0.0)

        # Calculate statistics
        method_results = {}
        for metric, values in scores.items():
            if values:
                method_results[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }
            else:
                method_results[metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0
                }

        return method_results

    def _calculate_simple_mrr(self, ground_truth_urls: List[str],
                              results: List[Dict[str, Any]]) -> float:
        """Simple MRR calculation for ablation"""
        if not results:
            return 0.0

        for rank, result in enumerate(results, 1):
            if result.get("url", "") in ground_truth_urls:
                return 1.0 / rank

        return 0.0

    def _calculate_simple_bleu(self, reference: str, candidate: str) -> float:
        """Simple BLEU calculation for ablation"""
        if not reference or not candidate:
            return 0.0

        # Simple word overlap
        ref_words = set(reference.lower().split())
        can_words = set(candidate.lower().split())

        if not ref_words or not can_words:
            return 0.0

        overlap = len(ref_words.intersection(can_words))
        precision = overlap / len(can_words) if can_words else 0.0
        recall = overlap / len(ref_words) if ref_words else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _calculate_simple_relevance(self, query: str, results: List[Dict[str, Any]]) -> float:
        """Simple relevance calculation for ablation"""
        if not results:
            return 0.0

        # Count query words in results
        query_words = set(query.lower().split())
        total_matches = 0

        for result in results[:3]:  # Only first 3 results
            text = result.get("text", "").lower()
            for word in query_words:
                if word in text and len(word) > 2:
                    total_matches += 1

        max_possible = len(query_words) * min(3, len(results))
        if max_possible == 0:
            return 0.0

        return total_matches / max_possible

    def _compare_methods(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across methods"""
        comparison = {}

        metrics = ["mrr", "bleu", "context_relevance", "response_times"]

        for metric in metrics:
            metric_comparison = {}
            for method, results in all_results.items():
                if metric in results:
                    metric_comparison[method] = results[metric]["mean"]

            # Find best method for this metric
            if metric_comparison:
                if metric == "response_times":
                    # Lower is better for response time
                    best_method = min(metric_comparison.items(), key=lambda x: x[1])[0]
                    best_value = metric_comparison[best_method]
                else:
                    # Higher is better for other metrics
                    best_method = max(metric_comparison.items(), key=lambda x: x[1])[0]
                    best_value = metric_comparison[best_method]

                comparison[metric] = {
                    "values": metric_comparison,
                    "best_method": best_method,
                    "best_value": best_value,
                    "improvement_over_baseline": {}
                }

                # Calculate improvement over other methods
                baseline = metric_comparison.get("sparse_only", 0.0)
                for method, value in metric_comparison.items():
                    if method != "sparse_only" and baseline > 0:
                        improvement = ((value - baseline) / baseline) * 100
                        comparison[metric]["improvement_over_baseline"][method] = improvement

        return comparison

    def save_results(self, output_path: Path):
        """Save ablation results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Also save as CSV for easier analysis
        csv_data = []
        for method, results in self.results.get("methods", {}).items():
            for metric, values in results.items():
                csv_data.append({
                    "method": method,
                    "metric": metric,
                    "mean": values.get("mean", 0.0),
                    "std": values.get("std", 0.0),
                    "min": values.get("min", 0.0),
                    "max": values.get("max", 0.0)
                })

        df = pd.DataFrame(csv_data)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved ablation results to {output_path} and {csv_path}")
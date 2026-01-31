#!/usr/bin/env python3
"""
Test script for comprehensive RAG evaluation metrics

Usage:
    python test_evaluation_metrics.py

This script tests all three main evaluation metrics:
1. MRR (Mean Reciprocal Rank) - URL Level [2 Marks]
2. BERTScore for Answer Quality [2 Marks] 
3. Precision@K for Retrieval Quality [2 Marks]
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, create_sample_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create more comprehensive test data"""
    return [
        {
            "question_id": "q001",
            "question": "What is the capital of France?",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Paris"],
            "retrieved_chunks": [
                {
                    "chunk_id": "Paris_chunk_0",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "text": "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents in 2019.",
                    "score": 0.95
                },
                {
                    "chunk_id": "France_chunk_1", 
                    "url": "https://en.wikipedia.org/wiki/France",
                    "text": "France, officially the French Republic, is a country primarily located in Western Europe.",
                    "score": 0.87
                },
                {
                    "chunk_id": "Europe_chunk_2",
                    "url": "https://en.wikipedia.org/wiki/Europe", 
                    "text": "Europe is a large peninsula conventionally considered a continent in its own right.",
                    "score": 0.65
                }
            ],
            "context_chunks": [
                {
                    "chunk_id": "Paris_chunk_0",
                    "text": "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents in 2019.",
                },
                {
                    "chunk_id": "France_chunk_1",
                    "text": "France, officially the French Republic, is a country primarily located in Western Europe.",
                }
            ],
            "generated_answer": "Paris is the capital city of France.",
            "reference_answer": "The capital of France is Paris."
        },
        {
            "question_id": "q002", 
            "question": "Who wrote Romeo and Juliet?",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/William_Shakespeare"],
            "retrieved_chunks": [
                {
                    "chunk_id": "Romeo_and_Juliet_chunk_0",
                    "url": "https://en.wikipedia.org/wiki/Romeo_and_Juliet",
                    "text": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
                    "score": 0.92
                },
                {
                    "chunk_id": "William_Shakespeare_chunk_1",
                    "url": "https://en.wikipedia.org/wiki/William_Shakespeare", 
                    "text": "William Shakespeare was an English playwright, poet and actor.",
                    "score": 0.88
                },
                {
                    "chunk_id": "English_literature_chunk_2",
                    "url": "https://en.wikipedia.org/wiki/English_literature",
                    "text": "English literature includes works written in the English language.",
                    "score": 0.45
                }
            ],
            "context_chunks": [
                {
                    "chunk_id": "Romeo_and_Juliet_chunk_0", 
                    "text": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
                },
                {
                    "chunk_id": "William_Shakespeare_chunk_1",
                    "text": "William Shakespeare was an English playwright, poet and actor.",
                }
            ],
            "generated_answer": "William Shakespeare wrote Romeo and Juliet.",
            "reference_answer": "Romeo and Juliet was written by William Shakespeare."
        },
        {
            "question_id": "q003",
            "question": "What is the largest planet in our solar system?",
            "ground_truth_urls": ["https://en.wikipedia.org/wiki/Jupiter"],
            "retrieved_chunks": [
                {
                    "chunk_id": "Solar_System_chunk_0",
                    "url": "https://en.wikipedia.org/wiki/Solar_System",
                    "text": "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.",
                    "score": 0.78
                },
                {
                    "chunk_id": "Planet_chunk_1",
                    "url": "https://en.wikipedia.org/wiki/Planet", 
                    "text": "A planet is a large, rounded astronomical body that is neither a star nor its remnant.",
                    "score": 0.72
                },
                {
                    "chunk_id": "Jupiter_chunk_2",
                    "url": "https://en.wikipedia.org/wiki/Jupiter",
                    "text": "Jupiter is the largest planet in the Solar System and the fifth planet from the Sun.",
                    "score": 0.95
                }
            ],
            "context_chunks": [
                {
                    "chunk_id": "Jupiter_chunk_2",
                    "text": "Jupiter is the largest planet in the Solar System and the fifth planet from the Sun.",
                },
                {
                    "chunk_id": "Solar_System_chunk_0",
                    "text": "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.",
                }
            ],
            "generated_answer": "Jupiter is the largest planet in our solar system.",
            "reference_answer": "The largest planet in our solar system is Jupiter."
        }
    ]


def main():
    """Test the comprehensive evaluation system"""
    print(" Testing RAG Evaluation Metrics")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data()
    print(f"📊 Created {len(test_data)} test questions")
    
    # Initialize evaluator
    try:
        evaluator = ComprehensiveEvaluator()
        print(" Comprehensive evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation
    try:
        print("\nRunning comprehensive evaluation...")
        results = evaluator.evaluate_system(
            test_data, 
            save_results=True,
            output_file="test_evaluation_results.json"
        )
        
        print("Evaluation completed successfully!")
        
        # Display key results
        print("\nKEY METRICS SUMMARY:")
        print("-" * 30)
        
        metrics = results.get("metrics", {})
        
        # MRR Results (2 Marks)
        mrr_data = metrics.get("mrr_url_level", {})
        print(f" MRR (URL Level): {mrr_data.get('mean_mrr', 0.0):.4f}")
        
        # BERTScore Results (2 Marks)  
        bert_data = metrics.get("bertscore_answer_quality", {})
        if bert_data:
            bert_f1 = bert_data.get("bertscore_f1", {}).get("mean", 0.0)
            print(f"🤖 BERTScore F1: {bert_f1:.4f}")
        else:
            print("🤖 BERTScore: Not available (requires transformers)")
        
        # Precision@K Results (2 Marks)
        precision_data = metrics.get("precision_k_retrieval_quality", {})
        p5 = precision_data.get("precision_at_5", {}).get("mean", 0.0)
        p10 = precision_data.get("precision_at_10", {}).get("mean", 0.0)
        hit_rate = precision_data.get("hit_rate_at_10", {}).get("mean", 0.0)
        
        print(f"Precision@5: {p5:.4f}")
        print(f"Precision@10: {p10:.4f}")
        print(f"Hit Rate@10: {hit_rate:.4f}")
        
        # Overall Performance
        summary = results.get("summary", {})
        performance = summary.get("overall_performance", "Unknown")
        print(f"\nOverall Performance: {performance}")
        
        # Detailed breakdown
        print(f"\n DETAILED BREAKDOWN:")
        for i, detail in enumerate(mrr_data.get("details", [])[:3]):
            qid = detail.get("question_id", f"q{i+1}")
            mrr = detail.get("mrr", 0.0)
            question = detail.get("question", "")[:50] + "..."
            print(f"   {qid}: MRR={mrr:.3f} | {question}")
        
        print(f"\n Results saved to: test_evaluation_results.json")
        print(f" See EVALUATION_METRICS_DOCUMENTATION.md for detailed explanations")
        
    except Exception as e:
        logger.error(f" Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
import numpy as np
from typing import List, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomMetricsEvaluator:
    """Custom evaluation metrics for RAG system"""

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

    def calculate_context_relevance(self, query: str,
                                    context_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate context relevance score

        Measures how relevant the retrieved context is to the query

        Returns:
            Relevance score between 0 and 1
        """
        if not context_chunks:
            return 0.0

        # Extract context texts
        context_texts = [chunk["text"] for chunk in context_chunks]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        try:
            # Combine query and contexts
            all_texts = [query] + context_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Calculate cosine similarity between query and each context
            query_vector = tfidf_matrix[0:1]
            context_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vector, context_vectors)[0]

            # Average similarity weighted by chunk rank (higher rank = more weight)
            weights = [1.0 / (i + 1) for i in range(len(similarities))]
            weights = np.array(weights) / np.sum(weights)

            weighted_similarity = np.sum(similarities * weights)

            return float(weighted_similarity)
        except:
            return 0.0

    def calculate_answer_specificity(self, answer: str,
                                     context_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate answer specificity score

        Measures how specific the answer is to the context (vs generic)

        Returns:
            Specificity score between 0 and 1
        """
        if not answer or not context_chunks:
            return 0.0

        # Extract all words from context
        context_words = set()
        for chunk in context_chunks:
            words = word_tokenize(chunk["text"].lower())
            words = [w for w in words if w.isalnum() and w not in self.stop_words]
            context_words.update(words)

        # Extract words from answer
        answer_words = word_tokenize(answer.lower())
        answer_words = [w for w in answer_words if w.isalnum() and w not in self.stop_words]

        if not answer_words:
            return 0.0

        # Calculate percentage of answer words that appear in context
        matching_words = [w for w in answer_words if w in context_words]
        specificity = len(matching_words) / len(answer_words)

        return specificity

    def calculate_retrieval_diversity(self,
                                      retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate retrieval diversity score

        Measures diversity of sources in retrieved chunks

        Returns:
            Diversity score between 0 and 1
        """
        if len(retrieved_chunks) <= 1:
            return 1.0 if retrieved_chunks else 0.0

        # Count unique sources
        unique_sources = set()
        for chunk in retrieved_chunks:
            source = chunk.get("url", chunk.get("title", ""))
            unique_sources.add(source)

        # Calculate diversity as unique sources / total chunks
        diversity = len(unique_sources) / len(retrieved_chunks)

        return diversity

    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate custom metrics for a batch of questions"""
        context_relevance_scores = []
        answer_specificity_scores = []
        retrieval_diversity_scores = []

        details = []

        for eval_item in evaluations:
            query = eval_item.get("question", "")
            answer = eval_item.get("generated_answer", "")
            context_chunks = eval_item.get("context_chunks", [])
            retrieved_chunks = eval_item.get("retrieved_chunks", [])

            # Calculate custom metrics
            relevance = self.calculate_context_relevance(query, context_chunks)
            specificity = self.calculate_answer_specificity(answer, context_chunks)
            diversity = self.calculate_retrieval_diversity(retrieved_chunks)

            context_relevance_scores.append(relevance)
            answer_specificity_scores.append(specificity)
            retrieval_diversity_scores.append(diversity)

            details.append({
                "question_id": eval_item.get("question_id", ""),
                "question": query,
                "context_relevance": relevance,
                "answer_specificity": specificity,
                "retrieval_diversity": diversity,
                "num_context_chunks": len(context_chunks),
                "num_retrieved_chunks": len(retrieved_chunks)
            })

        # Calculate statistics
        results = {
            "context_relevance": {
                "mean": float(np.mean(context_relevance_scores)),
                "std": float(np.std(context_relevance_scores)),
                "scores": context_relevance_scores
            },
            "answer_specificity": {
                "mean": float(np.mean(answer_specificity_scores)),
                "std": float(np.std(answer_specificity_scores)),
                "scores": answer_specificity_scores
            },
            "retrieval_diversity": {
                "mean": float(np.mean(retrieval_diversity_scores)),
                "std": float(np.std(retrieval_diversity_scores)),
                "scores": retrieval_diversity_scores
            },
            "details": details,
            "summary": {
                "total_questions": len(evaluations),
                "high_context_relevance": sum(1 for s in context_relevance_scores if s > 0.7),
                "high_specificity": sum(1 for s in answer_specificity_scores if s > 0.7),
                "high_diversity": sum(1 for s in retrieval_diversity_scores if s > 0.7)
            }
        }

        logger.info(f"Custom Metrics - Context Relevance: {results['context_relevance']['mean']:.4f}")
        logger.info(f"Custom Metrics - Answer Specificity: {results['answer_specificity']['mean']:.4f}")
        logger.info(f"Custom Metrics - Retrieval Diversity: {results['retrieval_diversity']['mean']:.4f}")

        return results
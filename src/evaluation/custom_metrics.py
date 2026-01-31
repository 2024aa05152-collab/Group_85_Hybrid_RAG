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
        precision_at_5_scores = []
        precision_at_10_scores = []
        hit_rate_at_10_scores = []

        details = []

        for eval_item in evaluations:
            query = eval_item.get("question", "")
            answer = eval_item.get("generated_answer", "")
            context_chunks = eval_item.get("context_chunks", [])
            retrieved_chunks = eval_item.get("retrieved_chunks", [])
            ground_truth_urls = eval_item.get("ground_truth_urls", [])

            # Calculate custom metrics
            relevance = self.calculate_context_relevance(query, context_chunks)
            specificity = self.calculate_answer_specificity(answer, context_chunks)
            diversity = self.calculate_retrieval_diversity(retrieved_chunks)
            
            # Calculate retrieval quality metrics
            precision_5 = self.calculate_precision_at_k(ground_truth_urls, retrieved_chunks, k=5)
            precision_10 = self.calculate_precision_at_k(ground_truth_urls, retrieved_chunks, k=10)
            hit_rate_10 = self.calculate_hit_rate_at_k(ground_truth_urls, retrieved_chunks, k=10)

            context_relevance_scores.append(relevance)
            answer_specificity_scores.append(specificity)
            retrieval_diversity_scores.append(diversity)
            precision_at_5_scores.append(precision_5)
            precision_at_10_scores.append(precision_10)
            hit_rate_at_10_scores.append(hit_rate_10)

            details.append({
                "question_id": eval_item.get("question_id", ""),
                "question": query,
                "context_relevance": relevance,
                "answer_specificity": specificity,
                "retrieval_diversity": diversity,
                "precision_at_5": precision_5,
                "precision_at_10": precision_10, 
                "hit_rate_at_10": hit_rate_10,
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
            "precision_at_5": {
                "mean": float(np.mean(precision_at_5_scores)),
                "std": float(np.std(precision_at_5_scores)),
                "scores": precision_at_5_scores
            },
            "precision_at_10": {
                "mean": float(np.mean(precision_at_10_scores)),
                "std": float(np.std(precision_at_10_scores)),
                "scores": precision_at_10_scores
            },
            "hit_rate_at_10": {
                "mean": float(np.mean(hit_rate_at_10_scores)),
                "std": float(np.std(hit_rate_at_10_scores)),
                "scores": hit_rate_at_10_scores
            },
            "details": details,
            "summary": {
                "total_questions": len(evaluations),
                "high_context_relevance": sum(1 for s in context_relevance_scores if s > 0.7),
                "high_specificity": sum(1 for s in answer_specificity_scores if s > 0.7),
                "high_diversity": sum(1 for s in retrieval_diversity_scores if s > 0.7),
                "high_precision_at_5": sum(1 for s in precision_at_5_scores if s > 0.6),
                "high_precision_at_10": sum(1 for s in precision_at_10_scores if s > 0.5),
                "successful_retrievals": sum(1 for s in hit_rate_at_10_scores if s > 0.0)
            }
        }

        logger.info(f"Custom Metrics - Context Relevance: {results['context_relevance']['mean']:.4f}")
        logger.info(f"Custom Metrics - Answer Specificity: {results['answer_specificity']['mean']:.4f}")
        logger.info(f"Custom Metrics - Retrieval Diversity: {results['retrieval_diversity']['mean']:.4f}")
        logger.info(f"Custom Metrics - Precision@5: {results['precision_at_5']['mean']:.4f}")
        logger.info(f"Custom Metrics - Precision@10: {results['precision_at_10']['mean']:.4f}")
        logger.info(f"Custom Metrics - Hit Rate@10: {results['hit_rate_at_10']['mean']:.4f}")

        return results

    def calculate_precision_at_k(self, ground_truth_urls: List[str],
                                retrieved_chunks: List[Dict[str, Any]], 
                                k: int = 5) -> float:
        """
        Calculate Precision@K for retrieval quality evaluation
        
        Justification:
        Precision@K is essential for RAG systems as it measures what fraction of the
        top-K retrieved documents are relevant. This directly impacts answer quality
        since irrelevant retrieved documents can lead to hallucinations or incorrect
        answers. Unlike MRR which only cares about the first relevant document,
        Precision@K evaluates the overall quality of the retrieved set.
        
        Calculation Method:
        Precision@K = (Number of relevant documents in top K) / K
        Where relevance is determined by URL matching with ground truth
        
        Interpretation:
        - Score range: 0.0 to 1.0 (higher is better)
        - 1.0: All top-K documents are relevant
        - 0.5: Half of top-K documents are relevant  
        - 0.0: No relevant documents in top-K
        
        Args:
            ground_truth_urls: List of correct Wikipedia URLs
            retrieved_chunks: List of retrieved chunks with metadata
            k: Number of top documents to evaluate (default: 5)
            
        Returns:
            Precision@K score between 0.0 and 1.0
        """
        if not retrieved_chunks or k <= 0:
            return 0.0
        
        # Take only top-K chunks
        top_k_chunks = retrieved_chunks[:k]
        
        # Extract URLs from retrieved chunks
        retrieved_urls = []
        for chunk in top_k_chunks:
            if 'url' in chunk:
                retrieved_urls.append(chunk['url'])
            else:
                # Fallback to extracting from chunk_id
                chunk_id = chunk.get('chunk_id', '')
                if chunk_id:
                    parts = chunk_id.split('_')
                    url_part = '_'.join(parts[:-1])
                    url = f"https://en.wikipedia.org/wiki/{url_part}"
                    retrieved_urls.append(url)
        
        # Count relevant documents
        relevant_count = sum(1 for url in retrieved_urls if url in ground_truth_urls)
        
        # Calculate precision@k
        precision_k = relevant_count / len(top_k_chunks)
        return float(precision_k)
    
    def calculate_hit_rate_at_k(self, ground_truth_urls: List[str],
                               retrieved_chunks: List[Dict[str, Any]],
                               k: int = 10) -> float:
        """
        Calculate Hit Rate@K (whether any relevant document appears in top-K)
        
        Justification:
        Hit Rate@K measures whether the retrieval system found at least one relevant
        document in the top-K results. This is important for RAG systems because
        having at least one relevant document significantly increases the chance
        of generating a correct answer.
        
        Args:
            ground_truth_urls: List of correct Wikipedia URLs
            retrieved_chunks: List of retrieved chunks with metadata  
            k: Number of top documents to evaluate (default: 10)
            
        Returns:
            1.0 if any relevant document found in top-K, 0.0 otherwise
        """
        if not retrieved_chunks or k <= 0:
            return 0.0
            
        # Take only top-K chunks
        top_k_chunks = retrieved_chunks[:k]
        
        # Check if any URL matches ground truth
        for chunk in top_k_chunks:
            if 'url' in chunk:
                if chunk['url'] in ground_truth_urls:
                    return 1.0
            else:
                # Fallback to extracting from chunk_id
                chunk_id = chunk.get('chunk_id', '')
                if chunk_id:
                    parts = chunk_id.split('_')
                    url_part = '_'.join(parts[:-1])
                    url = f"https://en.wikipedia.org/wiki/{url_part}"
                    if url in ground_truth_urls:
                        return 1.0
        
        return 0.0
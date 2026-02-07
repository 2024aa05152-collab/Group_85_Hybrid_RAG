#!/usr/bin/env python3
"""
Main pipeline for Hybrid RAG System
Run with: python src/pipeline.py --mode full
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import time
from datetime import datetime

# Add parent directory to path to find scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from config
try:
    from config import MODEL_CONFIG, RETRIEVAL_CONFIG, EVALUATION_CONFIG, FILE_PATHS, DATA_DIR, OUTPUTS_DIR, REPORTS_DIR
except ImportError:
    # Fallback: define them here if config import fails
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    REPORTS_DIR = BASE_DIR / "reports"

    MODEL_CONFIG = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "google/flan-t5-base",
        "generation_max_length": 512,
        "temperature": 0.7,
        "top_p": 0.95,
    }

    RETRIEVAL_CONFIG = {
        "chunk_size": 400,
        "chunk_overlap": 50,
        "top_k": 10,
        "top_n": 5,
        "rrf_k": 60,
    }

    EVALUATION_CONFIG = {
        "num_questions": 100,
        "question_types": ["factual", "comparative", "inferential", "multi-hop"],
        "metrics": ["mrr", "bleu", "context_relevance"],
    }

    FILE_PATHS = {
        "fixed_urls": DATA_DIR / "fixed_urls.json",
        "random_urls": DATA_DIR / "random_urls.json",
        "corpus_chunks": DATA_DIR / "corpus_chunks.json",
        "questions": DATA_DIR / "questions_100.json",
        "evaluation_results": DATA_DIR / "evaluation_results.csv",
        "faiss_index": DATA_DIR / "faiss_index.bin",
        "bm25_index": DATA_DIR / "bm25_index.pkl",
        "metadata": DATA_DIR / "chunk_metadata.json",
    }

from ingestion.wikipedia_loader import WikipediaLoader
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import TextChunker
#from indexing.dense_index import DenseIndexer
#from indexing.sparse_index import SparseIndexer
#from indexing.hybrid_rrf import HybridRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid_retriever import HybridRetriever

from generation.llm_generator import LLMGenerator
from evaluation.mrr import MRREvaluator
from evaluation.bleu import BLEUEvaluator
from evaluation.custom_metrics import CustomMetricsEvaluator
from evaluation.run_innovative_eval import run_full_innovative_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Simple question generator for pipeline"""

    def __init__(self):
        self.question_templates = {
            "factual": [
                "What is {entity}?",
                "When was {entity} created?",
                "Who created {entity}?",
                "Where is {entity} located?",
                "How does {entity} work?",
            ],
            "comparative": [
                "What are the differences between {entity1} and {entity2}?",
                "How is {entity1} similar to {entity2}?",
            ],
            "inferential": [
                "Based on the information, what can be inferred about {entity}?",
                "What are the implications of {fact}?",
            ],
            "multi-hop": [
                "What is the relationship between {entity1} and {entity2}?",
                "How did {event1} lead to {event2}?",
            ]
        }

    def extract_entities(self, text: str):
        """Extract entities from text"""
        import re
        entities = []
        # Simple extraction: capitalized words/phrases
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for match in matches:
            if len(match.split()) <= 3 and match not in entities:
                entities.append(match)
        return entities[:5]

    def generate_questions_from_chunks(self, chunks, num_questions=100):
        """Generate questions from chunks"""
        import random
        questions = []

        for i in range(num_questions):
            chunk = random.choice(chunks)
            entities = self.extract_entities(chunk["text"])

            if len(entities) >= 1:
                q_type = random.choice(list(self.question_templates.keys()))
                template = random.choice(self.question_templates[q_type])

                if "{entity}" in template:
                    question = template.replace("{entity}", random.choice(entities))
                elif "{entity1}" in template and "{entity2}" in template:
                    if len(entities) >= 2:
                        entity1, entity2 = random.sample(entities, 2)
                        question = template.replace("{entity1}", entity1).replace("{entity2}", entity2)
                    else:
                        continue
                elif "{fact}" in template:
                    sentences = chunk["text"].split('. ')
                    fact = sentences[0] if sentences else "this topic"
                    question = template.replace("{fact}", fact[:50])
                else:
                    question = template

                # Simple answer extraction
                sentences = chunk["text"].split('. ')
                answer = sentences[0] if sentences else "Information not found"

                questions.append({
                    "id": f"q_{i:03d}",
                    "question": question,
                    "answer": answer,
                    "question_type": q_type,
                    "source_chunk_id": chunk["chunk_id"],
                    "source_url": chunk["url"],
                    "source_urls": [chunk["url"]]
                })

        return questions


class HybridRAGPipeline:
    """Main pipeline for Hybrid RAG system"""

    def __init__(self, config=None):
        if config is None:
            config = {
                "model": MODEL_CONFIG,
                "retrieval": RETRIEVAL_CONFIG,
                "evaluation": EVALUATION_CONFIG
            }

        self.config = config
        self.file_paths = FILE_PATHS

        # Create directories
        for dir_path in [DATA_DIR, OUTPUTS_DIR, REPORTS_DIR]:
            dir_path.mkdir(exist_ok=True)

        # Initialize components
        self.wikipedia_loader = WikipediaLoader(FILE_PATHS)
        self.text_cleaner = TextCleaner()
        self.chunker = TextChunker(
            chunk_size=RETRIEVAL_CONFIG["chunk_size"],
            chunk_overlap=RETRIEVAL_CONFIG["chunk_overlap"]
        )

    def run_full_pipeline(self):
        """Run complete pipeline: ingestion, indexing, evaluation"""
        logger.info("=" * 60)
        logger.info("Starting Full Hybrid RAG Pipeline")
        logger.info("=" * 60)

        start_time = time.time()

        # Step 1: Data Ingestion
        logger.info("\nStep 1: Data Ingestion")
        chunks = self.ingest_and_process_data()

        if not chunks:
            logger.error("No chunks created. Exiting.")
            return

        # Step 2: Indexing
        logger.info("\nStep 2: Indexing")
        dense_retriever, sparse_retriever = self.build_indices(chunks)

        # Step 3: Build Hybrid System
        logger.info("\nStep 3: Building Hybrid System")
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            k=RETRIEVAL_CONFIG["rrf_k"]
        )

        llm_generator = LLMGenerator(model_name=MODEL_CONFIG["llm_model"])

        # Step 4: Generate Questions
        logger.info("\nStep 4: Generating Evaluation Questions")
        questions = self.generate_evaluation_questions(chunks)

        # Step 5: Evaluation
        logger.info("\nStep 5: Running Evaluation")
        evaluation_results = self.run_evaluation(
            hybrid_retriever,
            llm_generator,
            questions
        )

        # Step 6: Generate Report
        logger.info("\nStep 6: Generating Report")
        self.generate_report(evaluation_results)

        # Step 7: Run Innovative Evaluation Pipeline (Assignment Part 2.4)
        logger.info("\nStep 7: Running Innovative Automated Evaluation Pipeline")

        try:
            run_full_innovative_evaluation(
                questions_path=self.file_paths["questions"],
                corpus_path=self.file_paths["corpus_chunks"],
                outputs_dir=OUTPUTS_DIR
            )
            logger.info("Innovative evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Innovative evaluation failed: {e}")

        total_time = time.time() - start_time
        logger.info(f"\nPipeline completed in {total_time:.2f} seconds")
        logger.info("=" * 60)

    def ingest_and_process_data(self):
        """Step 1: Ingest and process Wikipedia data"""
        logger.info("Loading Wikipedia URLs...")

        # Get all URLs
        urls = self.wikipedia_loader.get_all_urls(regenerate_random=True, test_mode=True)  # Add test_mode=True

        logger.info(f"Fetching content for {len(urls)} URLs...")
        pages = self.wikipedia_loader.fetch_all_contents_parallel(urls, max_pages=30)  # Use parallel fetching

        logger.info("Cleaning and processing text...")
        cleaned_pages = self.text_cleaner.process_all_pages(pages)

        logger.info("Chunking text...")
        chunks = self.chunker.process_documents(cleaned_pages)

        # Save chunks
        self.chunker.save_chunks(chunks, self.file_paths["corpus_chunks"])

        logger.info(f"Created {len(chunks)} chunks from {len(cleaned_pages)} pages")
        return chunks

    def build_indices(self, chunks):
        """Step 2: Build dense and sparse indices"""
        logger.info("Building dense index...")
        dense_indexer = DenseRetriever(
            model_name=MODEL_CONFIG["embedding_model"]
        )
        dense_indexer.build_index(chunks)
        dense_indexer.save_index(
            self.file_paths["faiss_index"],
            self.file_paths["metadata"]
        )

        logger.info("Building sparse index...")
        sparse_indexer = SparseRetriever()
        sparse_indexer.build_index(chunks)
        sparse_indexer.save_index(
            self.file_paths["bm25_index"],
            self.file_paths["metadata"]
        )

        # Create retrievers
        dense_retriever = DenseRetriever(MODEL_CONFIG["embedding_model"])
        dense_retriever.load_index(
            self.file_paths["faiss_index"],
            self.file_paths["metadata"]
        )

        sparse_retriever = SparseRetriever()
        sparse_retriever.load_index(
            self.file_paths["bm25_index"],
            self.file_paths["metadata"]
        )

        return dense_retriever, sparse_retriever

    def generate_evaluation_questions(self, chunks):
        """Step 4: Generate evaluation questions"""
        question_generator = QuestionGenerator()

        questions_path = self.file_paths["questions"]

        # Check if questions file exists and is valid
        if questions_path.exists():
            try:
                logger.info("Loading existing questions...")
                with open(questions_path, 'r') as f:
                    questions = json.load(f)

                # Validate loaded questions
                if isinstance(questions, list) and len(questions) > 0:
                    logger.info(f"Loaded {len(questions)} existing questions")
                    return questions
                else:
                    logger.warning("Existing questions file is empty or invalid, generating new ones...")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading questions file: {e}, generating new ones...")

        # Generate new questions
        logger.info("Generating new questions...")
        questions = question_generator.generate_questions_from_chunks(
            chunks,
            num_questions=EVALUATION_CONFIG["num_questions"]
        )

        # Save questions
        with open(questions_path, 'w') as f:
            json.dump(questions, f, indent=2)

        logger.info(f"Generated {len(questions)} new questions")
        return questions

    def run_evaluation(self, hybrid_retriever, llm_generator, questions):
        """Step 5: Run evaluation on all questions"""
        logger.info(f"Evaluating on {len(questions)} questions...")

        # Initialize evaluators
        mrr_evaluator = MRREvaluator()
        bleu_evaluator = BLEUEvaluator()
        custom_evaluator = CustomMetricsEvaluator()

        evaluation_data = []

        for i, q in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {q['question'][:50]}...")

            try:
                # Retrieve
                retrieved_chunks = hybrid_retriever.retrieve(
                    q["question"],
                    top_k=RETRIEVAL_CONFIG["top_k"],
                    top_n=RETRIEVAL_CONFIG["top_n"]
                )

                # Generate answer
                context_chunks = retrieved_chunks[:RETRIEVAL_CONFIG["top_n"]]
                answer_response = llm_generator.generate_answer(q["question"], context_chunks)

                # Prepare evaluation item
                eval_item = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "ground_truth": q["answer"],
                    "ground_truth_urls": q.get("source_urls", []),
                    "generated_answer": answer_response.get("answer", ""),
                    "retrieved_chunks": retrieved_chunks,
                    "context_chunks": context_chunks,
                    "answer_success": answer_response.get("success", False)
                }

                evaluation_data.append(eval_item)

            except Exception as e:
                logger.error(f"Error evaluating question {i}: {e}")
                evaluation_data.append({
                    "question_id": q.get("id", f"q_{i}"),
                    "question": q.get("question", ""),
                    "ground_truth": q.get("answer", ""),
                    "ground_truth_urls": q.get("source_urls", []),
                    "generated_answer": f"Error: {str(e)}",
                    "retrieved_chunks": [],
                    "context_chunks": [],
                    "answer_success": False
                })

        # Calculate metrics
        logger.info("Calculating metrics...")

        mrr_results = mrr_evaluator.evaluate_batch(evaluation_data)
        bleu_results = bleu_evaluator.evaluate_batch(evaluation_data)
        custom_results = custom_evaluator.evaluate_batch(evaluation_data)

        # Combine results
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "successful_answers": sum(1 for item in evaluation_data if item["answer_success"]),
            "mrr": mrr_results,
            "bleu": bleu_results,
            "custom_metrics": custom_results,
            "details": evaluation_data
        }

        # Save results
        self.save_evaluation_results(evaluation_results)

        return evaluation_results

    def save_evaluation_results(self, results):
        """Save evaluation results to files"""
        # Save as JSON
        json_path = self.file_paths["evaluation_results"].with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save as CSV
        try:
            import pandas as pd
            csv_data = []

            for detail in results["details"]:
                csv_data.append({
                    "question_id": detail["question_id"],
                    "question": detail["question"],
                    "ground_truth": detail["ground_truth"],
                    "generated_answer": detail["generated_answer"],
                    "mrr": 0.0,  # Placeholder
                    "bleu": 0.0,  # Placeholder
                    "answer_success": detail["answer_success"]
                })

            df = pd.DataFrame(csv_data)
            df.to_csv(self.file_paths["evaluation_results"], index=False)
            logger.info(f"Saved evaluation results to {self.file_paths['evaluation_results']}")
        except ImportError:
            logger.warning("Pandas not installed, skipping CSV export")

    def generate_report(self, results):
        """Step 6: Generate evaluation report"""
        logger.info("Generating evaluation report...")

        # Create simple text report
        report_path = REPORTS_DIR / "evaluation_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Hybrid RAG System Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Report generated: {results['timestamp']}\n")
            f.write(f"Total questions evaluated: {results['total_questions']}\n")
            f.write(f"Successful answers: {results['successful_answers']}\n")
            if results['total_questions'] > 0:
                f.write(f"Success rate: {results['successful_answers'] / results['total_questions'] * 100:.1f}%\n\n")
            else:
                f.write("Success rate: 0.0%\n\n")

            f.write("Metrics Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"MRR (URL level): {results['mrr']['mean_mrr']:.4f}\n")
            f.write(f"BLEU score: {results['bleu']['mean_bleu']:.4f}\n")
            if 'context_relevance' in results['custom_metrics']:
                f.write(f"Context Relevance: {results['custom_metrics']['context_relevance']['mean']:.4f}\n\n")

            f.write("Performance Highlights:\n")
            f.write("-" * 40 + "\n")
            if 'summary' in results['mrr']:
                f.write(
                    f"Questions with correct URL found: {results['mrr']['summary']['questions_with_correct_url']}\n")
            if 'summary' in results['bleu']:
                f.write(f"High BLEU scores (>0.5): {results['bleu']['summary']['high_bleu']}\n\n")

        logger.info(f"Report saved to {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hybrid RAG Pipeline")
    parser.add_argument("--mode", choices=["full", "ingest", "index", "evaluate"],
                        default="full", help="Pipeline mode")
    parser.add_argument("--questions", type=int, default=100,
                        help="Number of evaluation questions to generate")

    args = parser.parse_args()

    # Update config
    EVALUATION_CONFIG["num_questions"] = args.questions

    # Create and run pipeline
    pipeline = HybridRAGPipeline()

    if args.mode == "full":
        pipeline.run_full_pipeline()
    elif args.mode == "ingest":
        pipeline.ingest_and_process_data()
    elif args.mode == "index":
        if not FILE_PATHS["corpus_chunks"].exists():
            logger.error("Corpus chunks not found. Run ingestion first.")
            return
        chunks = pipeline.chunker.load_chunks(FILE_PATHS["corpus_chunks"])
        pipeline.build_indices(chunks)
    elif args.mode == "evaluate":
        # Check if required files exist
        required_files = [
            FILE_PATHS["corpus_chunks"],
            FILE_PATHS["faiss_index"],
            FILE_PATHS["bm25_index"],
            FILE_PATHS["metadata"]
        ]

        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                logger.error("Run full pipeline or indexing first.")
                return

        # Load existing data and evaluate
        chunks = pipeline.chunker.load_chunks(FILE_PATHS["corpus_chunks"])
        questions = pipeline.generate_evaluation_questions(chunks)

        # Load indices
        dense_retriever = DenseRetriever(MODEL_CONFIG["embedding_model"])
        dense_retriever.load_index(FILE_PATHS["faiss_index"], FILE_PATHS["metadata"])

        sparse_retriever = SparseRetriever()
        sparse_retriever.load_index(FILE_PATHS["bm25_index"], FILE_PATHS["metadata"])

        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        llm_generator = LLMGenerator(model_name=MODEL_CONFIG["llm_model"])

        pipeline.run_evaluation(hybrid_retriever, llm_generator, questions)


if __name__ == "__main__":
    main()
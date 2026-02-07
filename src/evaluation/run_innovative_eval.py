"""
Innovative Evaluation Script (Section 2.3)

Includes:
- Adversarial Testing (hallucination & robustness)
- Uses existing Hybrid RAG pipeline (no core code modification)
- CSV outputs for reporting
"""
import faiss
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict

# -----------------------------
# RAG PIPELINE IMPORTS
# -----------------------------
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.generation.llm_generator import ResponseGenerator


# -----------------------------
# INITIALIZE COMPONENTS (ONCE)
# -----------------------------
print("[INFO] Initializing retrievers and generator...")

dense_retriever = DenseRetriever()
sparse_retriever = SparseRetriever()
hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)

response_generator = ResponseGenerator()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

print("[INFO] Loading corpus chunks and indices...")
chunks_path = DATA_DIR / "corpus_chunks.json"
dense_index_path = DATA_DIR / "dense.index"
sparse_index_path = DATA_DIR / "sparse.pkl"

if not chunks_path.exists():
    raise FileNotFoundError(
        f"Missing chunks file: {chunks_path}. Run ingestion to create it."
    )
if not dense_index_path.exists() or not sparse_index_path.exists():
    raise FileNotFoundError(
        f"Missing index files in {DATA_DIR}. Run indexing to create them."
    )

with open(chunks_path, "r", encoding="utf-8") as f:
    corpus_chunks = json.load(f)

dense_retriever.load(str(DATA_DIR), corpus_chunks)
sparse_retriever.load(str(DATA_DIR), corpus_chunks)

print("[INFO] Initialization complete.")


# -----------------------------
# RAG ORCHESTRATION FUNCTION
# -----------------------------
def run_rag(query: str, top_k: int = 10, rrf_k: int = 60):
    """
    Runs the full Hybrid RAG pipeline:
    1. Hybrid Retrieval (Dense + Sparse + RRF)
    2. LLM Generation using retrieved context
    """
    retrieved_chunks = hybrid_retriever.retrieve(
        query=query,
        top_k=top_k,
        rrf_k=rrf_k
    )

    answer = response_generator.generate(
        query=query,
        context_chunks=retrieved_chunks
    )

    return answer, retrieved_chunks


# -----------------------------
# ADVERSARIAL TESTING (2.3)
# -----------------------------
def run_adversarial_tests(
    questions_path: str = "src/evaluation/adversarial_questions.json",
    output_path: str = "src/evaluation/adversarial_results.csv"
):
    """
    Runs adversarial questions through the RAG pipeline
    and detects hallucinations for unanswerable queries.
    """

    print("[INFO] Loading adversarial questions...")
    with open(questions_path, "r") as f:
        adversarial_questions = json.load(f)

    results: List[Dict] = []

    print(f"[INFO] Running {len(adversarial_questions)} adversarial tests...")

    for q in adversarial_questions:
        print(f"[QUERY] {q['id']} | {q['type']}")

        answer, retrieved_chunks = run_rag(q["question"])

        # Simple hallucination heuristic:
        # If question is unanswerable but model still produces a long answer
        hallucinated = False
        if not q.get("answerable", True) and len(answer.strip()) > 20:
            hallucinated = True

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answerable": q.get("answerable", True),
            "generated_answer": answer,
            "hallucinated": hallucinated,
            "num_retrieved_chunks": len(retrieved_chunks)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    hallucination_rate = (
        df[df["answerable"] == False]["hallucinated"].mean()
        if not df[df["answerable"] == False].empty
        else 0.0
    )

    print("[INFO] Adversarial evaluation complete.")
    print(f"[METRIC] Hallucination Rate: {hallucination_rate:.2f}")
    print(f"[SAVED] Results written to {output_path}")

# -----------------------------
# FULL INNOVATIVE EVALUATION
# -----------------------------
def run_full_innovative_evaluation(
    questions_path: str,
    corpus_path: str,
    outputs_dir: str
):
    """
    Wrapper function required for Assignment Part 2.4.

    Runs:
    - Adversarial testing
    - Saves structured outputs inside outputs_dir
    """

    print("\n[INFO] Starting FULL Innovative Evaluation Pipeline (Part 2.4)")

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(exist_ok=True, parents=True)

    # Output file for adversarial results
    adversarial_output = outputs_dir / "adversarial_results.csv"

    # Run adversarial evaluation (Section 2.3)
    run_adversarial_tests(
        questions_path=questions_path,
        output_path=str(adversarial_output)
    )

    print("[INFO] Innovative evaluation finished successfully.")
    print(f"[SAVED] Adversarial results → {adversarial_output}")

# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_adversarial_tests()

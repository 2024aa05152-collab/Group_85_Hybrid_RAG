"""
Ablation Study for Hybrid RAG (Section 2.3 – Innovative Evaluation)

Compares:
- Dense-only retrieval
- Sparse-only retrieval
- Hybrid retrieval (Dense + Sparse + RRF)

Metric:
- Mean Reciprocal Rank (MRR) at URL level

Inputs (from data/):
- questions_100.json
- corpus_chunks.json
- dense.index
- sparse.pkl

Outputs (to outputs/):
- ablation_results.csv
- ablation_mrr.png
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

# -----------------------------
# PATHS (MATCH YOUR DATA DIR)
# -----------------------------
DATA_DIR = "data"
QUESTIONS_FILE = os.path.join(DATA_DIR, "questions_100.json")
CHUNKS_FILE = os.path.join(DATA_DIR, "corpus_chunks.json")

OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "ablation_results.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "ablation_mrr.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD QUESTIONS
# -----------------------------
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"[INFO] Loaded {len(questions)} questions")

# -----------------------------
# LOAD CHUNKS (METADATA)
# -----------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# INITIALIZE RETRIEVERS
# -----------------------------
# IMPORTANT:
# Both retrievers expect DIRECTORY paths, not file paths

dense = DenseRetriever()
dense.load(DATA_DIR, chunks)     # loads data/dense.index

sparse = SparseRetriever()
sparse.load(DATA_DIR, chunks)    # loads data/sparse.pkl

hybrid = HybridRetriever(dense, sparse)

# -----------------------------
# MRR CALCULATION (URL LEVEL)
# -----------------------------
def compute_mrr(results, gt_url):
    if not gt_url:
        return 0.0
    for rank, r in enumerate(results, start=1):
        if r.get("url") == gt_url:
            return 1.0 / rank
    return 0.0

# -----------------------------
# RUN ABLATION
# -----------------------------
rows = []

for q in questions:
    query = q["question"]
    gt_url = (
        q.get("source_url")
        or q.get("url")
        or q.get("ground_truth_url")
    )

    dense_results = dense.retrieve(query, top_k=10)
    sparse_results = sparse.retrieve(query, top_k=10)
    hybrid_results = hybrid.retrieve(query, top_k=10)

    rows.append({
        "question_id": q.get("id"),
        "dense_mrr": compute_mrr(dense_results, gt_url),
        "sparse_mrr": compute_mrr(sparse_results, gt_url),
        "hybrid_mrr": compute_mrr(hybrid_results, gt_url)
    })

df = pd.DataFrame(rows)

# -----------------------------
# SAVE CSV
# -----------------------------
df.to_csv(CSV_PATH, index=False)
print("[SAVED]", CSV_PATH)

# -----------------------------
# PLOT RESULTS
# -----------------------------
mean_scores = df[["dense_mrr", "sparse_mrr", "hybrid_mrr"]].mean()

plt.figure(figsize=(6, 4))
mean_scores.plot(kind="bar")
plt.title("Ablation Study: Dense vs Sparse vs Hybrid")
plt.ylabel("Mean Reciprocal Rank (MRR)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("[SAVED]", PLOT_PATH)

print("[DONE] Ablation study completed successfully")

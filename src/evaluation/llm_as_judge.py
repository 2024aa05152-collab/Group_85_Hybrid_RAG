"""
LLM-as-Judge Evaluation (Section 2.3)

Evaluates generated answers on:
- Factual Accuracy
- Completeness
- Relevance
- Coherence

Outputs:
- outputs/llm_judge_scores.csv
"""

import os
import json
import pandas as pd
from src.generation.llm_generator import ResponseGenerator

# -----------------------------
# PATHS
# -----------------------------
INPUT_FILE = "src/evaluation/adversarial_results.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llm_judge_scores.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE)

judge_llm = ResponseGenerator()

# -----------------------------
# JUDGE PROMPT
# -----------------------------
def build_prompt(question, answer):
    return f"""
You are an expert evaluator.

Evaluate the following answer on a scale of 0 to 5 for each criterion:
- Factual Accuracy
- Completeness
- Relevance
- Coherence

Question:
{question}

Answer:
{answer}

Return your response strictly as JSON in this format:
{{
  "accuracy": <int>,
  "completeness": <int>,
  "relevance": <int>,
  "coherence": <int>
}}
"""

# -----------------------------
# RUN JUDGING
# -----------------------------
records = []

for _, row in df.iterrows():
    prompt = build_prompt(row["question"], row["generated_answer"])

    try:
        response = judge_llm.generate(prompt, [])
        scores = json.loads(response)

        records.append({
            "question": row["question"],
            "accuracy": scores.get("accuracy", 0),
            "completeness": scores.get("completeness", 0),
            "relevance": scores.get("relevance", 0),
            "coherence": scores.get("coherence", 0),
            "average_score": sum(scores.values()) / 4
        })

    except Exception:
        records.append({
            "question": row["question"],
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "coherence": 0,
            "average_score": 0
        })

# -----------------------------
# SAVE RESULTS
# -----------------------------
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_FILE, index=False)

print("[SAVED]", OUTPUT_FILE)
print("[DONE] LLM-as-Judge evaluation complete")

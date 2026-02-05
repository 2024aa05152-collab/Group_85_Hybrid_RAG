"""
Confidence Calibration for Hybrid RAG (Section 2.3)

Confidence source:
- Normalized LLM-as-Judge average score

Outputs:
- outputs/calibration_curve.png
- outputs/confidence_calibration.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# -----------------------------
# PATHS
# -----------------------------
LLM_JUDGE_FILE = "outputs/llm_judge_scores.csv"
ADVERSARIAL_FILE = "src/evaluation/adversarial_results.csv"

OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "confidence_calibration.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "calibration_curve.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
judge_df = pd.read_csv(LLM_JUDGE_FILE)
adv_df = pd.read_csv(ADVERSARIAL_FILE)

# Align rows (by question text)
df = judge_df.merge(
    adv_df[["question", "hallucinated"]],
    on="question",
    how="inner"
)

# -----------------------------
# DEFINE CONFIDENCE & CORRECTNESS
# -----------------------------
df["confidence"] = df["average_score"] / 5.0          # normalize to [0,1]
df["correct"] = (~df["hallucinated"]).astype(int)     # 1 = correct

# -----------------------------
# CALIBRATION CURVE
# -----------------------------
prob_true, prob_pred = calibration_curve(
    df["correct"],
    df["confidence"],
    n_bins=5,
    strategy="uniform"
)

# -----------------------------
# SAVE CSV
# -----------------------------
df[["question", "confidence", "correct"]].to_csv(CSV_PATH, index=False)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Model Calibration")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.xlabel("Predicted Confidence")
plt.ylabel("Observed Accuracy")
plt.title("Confidence Calibration Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("[SAVED]", CSV_PATH)
print("[SAVED]", PLOT_PATH)
print("[DONE] Confidence calibration completed")

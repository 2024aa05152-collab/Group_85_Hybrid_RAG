"""
Error Analysis for Hybrid RAG (Section 2.3)

Outputs:
- outputs/error_analysis.csv
- outputs/error_type_distribution.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PATHS (ALIGNED WITH YOUR REPO)
# -----------------------------
INPUT_FILE = "src/evaluation/adversarial_results.csv"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "error_analysis.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "error_type_distribution.png")

# -----------------------------
# ENSURE OUTPUT DIRECTORY EXISTS
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Output directory ready:", OUTPUT_DIR)

# -----------------------------
# LOAD DATA
# -----------------------------
print("[INFO] Loading adversarial results...")

df = pd.read_csv(INPUT_FILE)

print(f"[INFO] Loaded {len(df)} rows")

# -----------------------------
# ERROR CLASSIFICATION
# -----------------------------
def classify_error(row):
    if row["answerable"] is False and row["hallucinated"] is True:
        return "hallucination_error"
    if row["answerable"] is True and len(row["generated_answer"].strip()) == 0:
        return "generation_error"
    return "success"

print("[INFO] Classifying errors...")
df["error_type"] = df.apply(classify_error, axis=1)

# -----------------------------
# SAVE CSV
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print("[SAVED]", OUTPUT_CSV)

# -----------------------------
# PLOT DISTRIBUTION
# -----------------------------
print("[INFO] Generating plot...")

counts = df["error_type"].value_counts()

plt.figure(figsize=(6, 4))
counts.plot(kind="bar")
plt.title("Error Type Distribution")
plt.ylabel("Count")
plt.xlabel("Error Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()

print("[SAVED]", OUTPUT_PLOT)

# -----------------------------
# DONE
# -----------------------------
print("[DONE] Error analysis complete")
print(counts)

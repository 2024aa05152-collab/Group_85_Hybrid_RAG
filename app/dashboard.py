import streamlit as st
import pandas as pd

st.set_page_config(page_title="Hybrid RAG Evaluation Dashboard", layout="wide")

st.title("📊 Hybrid RAG Evaluation Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
ablation = pd.read_csv("outputs/ablation_results.csv")
errors = pd.read_csv("outputs/error_analysis.csv")
judge = pd.read_csv("outputs/llm_judge_scores.csv")

# -----------------------------
# SECTIONS
# -----------------------------
st.header("Ablation Study")
st.dataframe(ablation.head())
st.image("outputs/ablation_mrr.png")

st.header("Error Analysis")
st.dataframe(errors.head())
st.image("outputs/error_type_distribution.png")

st.header("LLM-as-Judge Scores")
st.dataframe(judge.head())

st.markdown("✅ This dashboard provides an interactive overview of evaluation results.")

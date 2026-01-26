# Hybrid RAG System with Automated Evaluation

## Overview
A hybrid Retrieval-Augmented Generation system combining dense vector retrieval (sentence transformers), sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from 500 Wikipedia articles.

## Features
- **Hybrid Retrieval**: Dense + Sparse + RRF fusion
- **Automated Evaluation**: 100 generated questions with MRR and custom metrics
- **Interactive UI**: Streamlit dashboard
- **One-command Pipeline**: Full automation from indexing to evaluation

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
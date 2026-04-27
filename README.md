# 🎯 AI-Powered Talent Sourcing and Ranking Pipeline

> An end-to-end machine learning pipeline designed to automate candidate discovery, ranking, and re-ranking for recruitment workflows using NLP, Learning-to-Rank algorithms, and feedback-driven optimization.
---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem & Goals](#business-problem--goals)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Bias Mitigation Strategy](#bias-mitigation-strategy)
- [Business Impact](#business-impact)
- [Future Improvements](#future-improvements)

---

## Project Overview

Recruitment teams often rely on keyword-based searches and manual screening to identify candidates. This approach is:

* Time-consuming
* Inefficient at scale
* Prone to inconsistencies and human bias

This project addresses these challenges by building a data-driven, adaptive ranking system that:

* Understands job roles using semantic NLP techniques
* Ranks candidates based on multi-dimensional features
* Learns continuously from recruiter feedback (⭐ starring)
* Reduces manual effort while improving candidate quality

---

## Business Problem & Goals

**The Problem:** The existing manual sourcing process was labor-intensive, time-consuming, and susceptible to unconscious human bias, making it difficult to consistently identify ideal candidates at scale.

**The Goals:**

| Goal | Description |
|------|-------------|
| Predict Candidate Fit | Generate a probability score (0–1) reflecting how well a candidate matches a role |
| Rank Candidates | Produce an ordered list of candidates by fitness score |
| Re-rank Candidates | Dynamically update rankings based on recruiter feedback (e.g., "starring" a candidate) |
| Filter Candidates | Remove low-relevance candidates using a data-driven cutoff strategy |
| Reduce Bias | Replace subjective manual evaluation with objective, data-driven scoring |

---

## Pipeline Architecture
<img width="443" height="569" alt="image" src="https://github.com/user-attachments/assets/127d8628-04bb-46c6-b8ff-d35ee349387d" />

## Features

- **Semantic Candidate Search** — Uses SBERT embeddings to understand job title meaning, not just keyword overlap
- **Multi-Feature Scoring** — Combines job title relevance, location, and professional connections into a single weighted fitness score
- **Adaptive Re-Ranking** — Supports recruiter feedback loops; starring a candidate updates model rankings dynamically
- **Smart Filtering** — Percentile-based cutoff trims the candidate pool to only the highest-potential profiles
- **Bias-Aware Design** — Data-driven scoring reduces reliance on subjective human judgment

---

## Tech Stack

| Category | Tools / Libraries |
|----------|-------------------|
| Language | Python |
| NLP & Embeddings | `gensim` (Word2Vec, FastText), `sentence-transformers` (SBERT — `all-MiniLM-L6-v2`) |
| Machine Learning | `xgboost` (XGBRanker), `lightgbm` (LGBMRanker) |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Text Preprocessing | `nltk` (tokenization, stopwords, lemmatization) |
| Evaluation | `scikit-learn` (NDCG, MAP) |
| Data Source | Google Sheets (CSV export) |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/talent-ranking-pipeline.git
cd talent-ranking-pipeline
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

### Step 1 — Load and Preprocess Data

```python
from pipeline.preprocessing import load_data, preprocess_titles

df = load_data("data/candidates.csv")
df["clean_title"] = preprocess_titles(df["job_title"])
```

### Step 2 — Generate Embeddings

```python
from pipeline.embeddings import generate_sbert_embeddings

title_embeddings = generate_sbert_embeddings(df["clean_title"].tolist())
location_embeddings = generate_sbert_embeddings(df["location"].tolist())
```

### Step 3 — Calculate Fit Scores

```python
from pipeline.scoring import compute_fit_scores, compute_combined_score

query = "Aspiring Human Resources"
df["fit_title"] = compute_fit_scores(query, title_embeddings)
df["fit_location"] = compute_fit_scores("New York", location_embeddings)
df["fit_combined"] = compute_combined_score(
    df, title_weight=0.6, location_weight=0.3, connections_weight=0.1
)
```

### Step 4 — Filter Candidates

```python
from pipeline.filtering import apply_percentile_filter

df_filtered = apply_percentile_filter(df, column="fit_combined", percentile=70)
print(f"Candidates retained: {len(df_filtered)} / {len(df)}")
```

### Step 5 — Train Ranking Models

```python
from pipeline.ranker import train_xgb_ranker, train_lgbm_ranker

xgb_model, xgb_rankings = train_xgb_ranker(df_filtered)
lgbm_model, lgbm_rankings = train_lgbm_ranker(df_filtered)
```

### Step 6 — Re-rank with Starred Candidate

```python
from pipeline.reranker import apply_star_signal, rerank

df_updated = apply_star_signal(df_filtered, candidate_id=42, boost=0.95)
updated_rankings = rerank(lgbm_model, df_updated)
```

---

## Model Details

### Embedding Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Word2Vec** | Dense vectors from word co-occurrence context | Standard title matching |
| **FastText** | Sub-word (character n-gram) embeddings | Rare words, abbreviations, misspellings |
| **SBERT** (`all-MiniLM-L6-v2`) | Contextual sentence-level embeddings | Nuanced semantic understanding of full job titles |

SBERT was the primary embedding method used due to its superior performance on semantic similarity tasks.

### Ranking Models

**XGBRanker**
- Objective: `rank:pairwise`
- Learns to differentiate candidate pairs by relevance
- Strong performance on structured tabular features

**LGBMRanker**
- Objective: `lambdarank`
- Relevance labels: discretized integer scores from `fit_combined`
- Parameter `min_child_samples=1` applied for small filtered datasets

### Weighted Fit Score Formula

```
fit_combined = (0.60 × title_similarity)
             + (0.30 × location_similarity)
             + (0.10 × normalized_connections)
```

These weights reflect business priorities and can be tuned per role or client.

---

## Evaluation Metrics

| Metric | Description | Score Achieved |
|--------|-------------|----------------|
| **NDCG** (Normalized Discounted Cumulative Gain) | Measures ranking quality, rewarding highly relevant candidates placed at the top | ~0.99 |
| **MAP** (Mean Average Precision) | Evaluates precision across recall levels for ranked lists | ~0.99 |

High NDCG and MAP scores confirm the pipeline successfully surfaces the most relevant candidates at the top of ranked lists.

---

## Bias Mitigation Strategy

The pipeline incorporates a multi-layered approach to reducing human bias:

### 1. Data-Driven Scoring
Replaces subjective recruiter judgment with objective, reproducible model scores during initial screening.

### 2. Enhanced Feature Set *(Roadmap)*
Future versions will incorporate skills, years of experience, and project portfolios — reducing over-reliance on job titles and location as proxies for candidate quality.

### 3. Algorithmic Fairness Metrics *(Roadmap)*
Planned integration of fairness constraints (e.g., demographic parity, equalized odds) to monitor and correct for disparate impact across candidate groups.

### 4. Improved Human-in-the-Loop Feedback *(Roadmap)*
The current "star" signal will be augmented with structured recruiter explanations (e.g., reason for starring or rejection), enabling the model to learn from human intent rather than blindly amplifying human preferences.

---

## Business Impact

| Outcome | Details |
|---------|---------|
| **Reduced Manual Labor** | Automates initial screening, freeing recruiter time for high-value evaluation |
| **Smaller, Higher-Quality Pools** | Percentile filtering cut candidate lists by ~70% (e.g., 104 → 31), focusing review effort |
| **Better Candidate-Role Fit** | Semantic embeddings surface genuinely relevant candidates missed by keyword search |
| **Adaptive Rankings** | System learns from recruiter feedback and improves over time |
| **Scalability** | Handles growing candidate volumes without proportional increases in manual effort |
| **Competitive Advantage** | Positions the company as a data-driven leader in talent acquisition |

---

## Future Improvements

- [ ] Integrate skills, certifications, and experience-year features for richer candidate profiles
- [ ] Add fairness auditing layer with demographic parity and equalized opportunity metrics
- [ ] Implement structured feedback UI for recruiters (reason codes for stars/rejections)
- [ ] Experiment with cross-encoder re-ranking using fine-tuned SBERT models
- [ ] Build a real-time inference API (FastAPI or Flask) for live candidate scoring
- [ ] Expand to non-tech roles with role-specific embedding fine-tuning
- [ ] Develop a recruiter dashboard for interactive ranking and filter management

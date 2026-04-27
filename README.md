# 🎯 AI-Powered Talent Sourcing and Ranking Pipeline

> An end-to-end machine learning pipeline designed to automate candidate discovery, ranking, and re-ranking for recruitment workflows using NLP, Learning-to-Rank algorithms, and feedback-driven optimization.
---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem & Goals](#business-problem--goals)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
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
| Evaluation | `scikit-learn` (NDCG, MAP) 

---

## Usage

### Step 1: Data Loading & Exploratory Analysis

The project began by loading candidate data and performing exploratory data analysis.

Key Activities:
- Checked missing values
- Reviewed job title distribution
- Examined location patterns
- Standardized connection values
- Validated dataset quality

### Step 2: Text Preprocessing

The job_title field was cleaned and standardized using NLP preprocessing.

Techniques Used:
* Lowercasing
* Tokenization
* Stop-word removal
* Lemmatization
* Keyword normalization

### tep 3: Embedding Generation

Multiple text embedding methods were tested.

* Word2Vec: Captured contextual word relationships.

* FastText: Handled sub-word information and misspellings.

* SBERT (Best Performance): Used pre-trained transformer embeddings to capture semantic meaning.

### Step 4: Initial Candidate Fit Score

Candidate relevance was first estimated using cosine similarity between:
- Query embedding
- Candidate job title embedding

### Step 5 : Candidate Filtering
-  A percentile-based cutoff (specifically the 70th percentile of the fit_combined score) was introduced to filter out less relevant candidates from the dataset. 

### Step 6 : Learning-to-Rank Models

To improve ranking quality beyond similarity scores, I implemented advanced ranking algorithms.

Models Built:
🔹 XGBRanker
- Pairwise ranking objective
- Strong baseline performance
🔹 LGBMRanker
- LambdaRank objective
- Excellent ranking adaptability
### Step 7 : Re-Ranking with Recruiter Feedback

One major business requirement was adapting rankings based on recruiter actions.

Example:

If recruiter stars candidate ranked #7: Candidate #7 becomes preferred candidate
What I built:
- Increased candidate relevance label
- Retrained ranking model
- Updated rankings dynamically
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

### Weighted Fit Score Formula (optional)

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
**Enhance Feature Set for Bias Mitigation**: To reduce bias in the model, we should include more meaningful and less biased features such as skills, years of experience, and project work instead of relying only on job titles, location, and connections. This helps the model focus more on a candidate’s actual abilities.

**Implement Bias Detection and Mitigation Strategies**: we can apply different techniques to detect and reduce bias in the system. This can include improving the data, adjusting how the model learns, or modifying the final rankings to make them more balanced and fair.

**Refine the Human-in-the-Loop Feedback Mechanism**:The feedback system should also be improved by collecting more detailed information, such as why a recruiter starred or rejected a candidate. This helps the model learn better and reduces the risk of blindly copying human bias.

---

## Business Impact

| Outcome | Details |
|---------|---------|
| **Reduced Manual Labor** | Automates initial screening, freeing recruiter time for high-value evaluation |
| **Smaller, Higher-Quality Pools** | Percentile filtering cut candidate lists by ~70% (e.g., 104 → 31), focusing review effort |
| **Better Candidate-Role Fit** | Semantic embeddings surface genuinely relevant candidates missed by keyword search |
| **Adaptive Rankings** | System learns from recruiter feedback and improves over time |
| **Scalability** | Handles growing candidate volumes without proportional increases in manual effort |

---

## Future Improvements
To further improve the ranking system, advanced feature engineering can be added. Instead of relying only on job title relevance, the model can also use location similarity, connection strength, skills match, years of experience, certifications, and project relevance. This would improve ranking accuracy and better reflect real recruiter decisions.

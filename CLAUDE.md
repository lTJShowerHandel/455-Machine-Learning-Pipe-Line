# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **fraud detection ML pipeline project** built for the IS455 course, following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework. The repo contains:

- `context.md` — Project specification, requirements, and expected deliverables
- `shop.db` — SQLite database (~2.2 MB) with transaction data for fraud detection
- `Textbook_Chapters/` — 28 educational markdown chapters covering the full ML curriculum
- `fraud_detection.ipynb` — The main student implementation notebook (may or may not exist)
- `model.sav` — Serialized trained model (pickle format, produced after modeling phase)

## Development Environment

**Stack:** Python + Jupyter Notebook, SQLite, Supabase (backend), Vercel (deployment)

**Core dependencies:**
```
pandas, scikit-learn, numpy, seaborn, matplotlib, statsmodels, sqlite3, pickle
```

**Working with the database:**
```python
import sqlite3
conn = sqlite3.connect('shop.db')
df = pd.read_sql_query("SELECT * FROM <table>", conn)
```

**Loading/saving the model:**
```python
import pickle
# Save
pickle.dump(model, open('model.sav', 'wb'))
# Load
model = pickle.load(open('model.sav', 'rb'))
```

## CRISP-DM Pipeline Phases

All work in `fraud_detection.ipynb` must follow these six phases in order:

1. **Business Understanding** — Define fraud detection goals; primary metrics are Recall and F1-score
2. **Data Understanding** — Explore `shop.db`, assess quality, distributions, and correlations
3. **Data Preparation** — Clean data, handle class imbalance (fraud is rare), encode/scale features
4. **Modeling** — Train classifiers (logistic regression, decision trees, random forest, gradient boosting) using scikit-learn pipelines
5. **Evaluation** — Cross-validate, compare models on Recall/F1/AUC; select final model
6. **Deployment** — Export `model.sav`, integrate with Supabase + Vercel API for real-time predictions

## Key Constraints

- **Evaluation metric priority**: Recall and F1-score matter most for fraud detection (false negatives are costly); accuracy alone is misleading due to class imbalance
- **Class imbalance**: Fraud cases are rare — use stratified splits, SMOTE, class weights, or threshold tuning
- **Pipeline discipline**: Use `sklearn.pipeline.Pipeline` to prevent data leakage between train/test sets
- **CRISP-DM compliance**: Each notebook section must map to a named CRISP-DM phase

## Textbook Reference

The `Textbook_Chapters/` folder is the authoritative reference for techniques used in this project:

| Topic | Chapter(s) |
|-------|-----------|
| CRISP-DM methodology | Chapter 1 |
| Pandas / data wrangling | Chapters 2–4 |
| EDA & visualization | Chapters 5–9 |
| Classification models | Chapters 13–14 |
| Model evaluation & tuning | Chapter 15 |
| Ensemble methods | Chapters 16–17 |
| Deployment | Chapter 26–27 |

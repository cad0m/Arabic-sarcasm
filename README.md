# Arabic Sarcasm Detection (ArSarcasm-v2)

A modular, reproducible machine learning and NLP project for detecting sarcasm in Arabic tweets using the **ArSarcasm-v2** dataset.

---

## Project Structure & File Guide

```text
tp2-arabic-sarcasm/
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis (class balance, dialects, text lengths)
│   └── 02_baseline.ipynb            # Baseline models (TF-IDF, classifiers, evaluations)
│
├── src/
│   ├── data.py                      # Data loading, train/val/test splitting, deduplication & leakage cleaning
│   ├── preprocess.py                # Text normalization pipeline (diacritics, emojis, dialects, stemming)
│   ├── train.py                     # Training utilities, class balancing (oversampling), threshold tuning, seeds
│   ├── evaluate.py                  # Evaluation metrics (F1, PR-AUC), McNemar statistical test, plotting & saving
│   └── resources/
│       ├── dialect_to_msa.json      # Mapping dictionary: dialectal Arabic terms -> Modern Standard Arabic (MSA)
│       └── hypernyms.json           # Semantic abstraction mapping: specific entities -> general categories
│
├── results/
│   └── results.csv                  # Experiment tracking table (F1 score, PR-AUC, optimal threshold, latency)
│
├── predictions/                     # Output directory for test/val predictions (git-ignored)
├── pyproject.toml                   # Project dependencies and environment specification
├── uv.lock                          # Locked dependency tree for deterministic reproduction
└── main.py                          # General entrypoint script
```

---

## File Breakdown

### Notebooks (`notebooks/`)
- **[`01_EDA.ipynb`](file:///notebooks/01_EDA.ipynb)**: Analyzes label distribution, dialect variety, tweet lengths, and vocabulary patterns in ArSarcasm-v2.
- **[`02_baseline.ipynb`](file:///notebooks/02_baseline.ipynb)**: End-to-end baseline experiments comparing classical classifiers (Logistic Regression, SVM, Random Forest, etc.) with threshold optimization.

### Source Code (`src/`)
- **[`src/data.py`](file:///src/data.py)**: Downloads dataset splits directly from GitHub, performs stratified splitting (90/10 train/val), generates cleaned text keys, and filters out cross-split leakage, label conflicts, and exact duplicates.
- **[`src/preprocess.py`](file:///src/preprocess.py)**: Composable preprocessing pipeline (`build_pipeline`) supporting:
  - HTML & URL stripping/replacement
  - User mention extraction (`<USER>`) and hashtag segmentation
  - Emoji conversion into Arabic text (`emoji_to_arabic`)
  - Diacritics (`Tashkeel`) and Kashida removal
  - Letter normalization (alif, yaa, taa marbuta) & character elongation reduction
  - Dialect-to-MSA mapping and hypernym replacement
  - Light stemming (Tashaphyne) and root extraction (ISRI stemmer)
- **[`src/train.py`](file:///src/train.py)**: Reproducibility utilities (`set_seed`), optimal decision threshold search over Precision-Recall curves, and training set oversampling for minority sarcasm class.
- **[`src/evaluate.py`](file:///src/evaluate.py)**: Performance scoring (F1, PR-AUC, Macro-F1, Precision, Recall), bootstrap confidence intervals, McNemar test for model comparison, inference latency measurement, bar plot generation, and prediction exports.
- **[`src/resources/`](file:///src/resources/)**: Static JSON dictionaries for dialect-to-MSA normalization and hypernym abstraction.

---

## Quickstart

### 1. Installation
This project uses [`uv`](https://github.com/astral-sh/uv) for fast and deterministic package management:

```bash
# Sync all dependencies
uv sync
```

### 2. Run Experiments
Open the notebooks with your preferred Jupyter environment:
```bash
uv run jupyter lab
```
Or execute a script:
```bash
uv run python main.py
```
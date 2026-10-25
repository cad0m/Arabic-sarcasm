# TP2 ArSarcasm-v2 — Parts B and D (add-on package)

Notebooks and helper modules for **Part B** (PLM used directly) and **Part D** (frozen PLM as feature extractor),
including the questions and Part F analyses that depend on them. Nothing here was executed by the author
(no GPU / HF access); only syntax was checked. Expect to fix small errors on the first run.

## Contents
```
src/plm.py          frozen-encoder feature extraction ([CLS]/mean pooled, token states) + disk cache
src/models.py       AttentionPooling + BiLSTMAttention (head of C2, reused by D2)
src/analysis.py     Part F helpers: cue removers/detectors, cue ablation (F4), subgroup scores (F3),
                    prior shift + F1(threshold) plot (F1), error sampling (F6)
notebooks/B_plm_direct.ipynb   B1 zero-shot NLI, B2 sentiment heuristic, Q4, Q5
notebooks/D_frozen_plm.ipynb   D1, D2, attention figure, final models, F1-F6 for D models, Q8 table
```

## Installation into your repo
Copy the files into your existing project, keeping your own `src/data.py`, `preprocess.py`, `train.py`, `evaluate.py`:
```
<repo>/src/plm.py  <repo>/src/models.py  <repo>/src/analysis.py
<repo>/notebooks/B_plm_direct.ipynb  <repo>/notebooks/D_frozen_plm.ipynb
```
`src/models.py` is new: if you already wrote a C2 head, you can swap the import in D.

## Requirements
```
pip install torch transformers scikit-learn pandas numpy scipy matplotlib seaborn emoji nltk tashaphyne
# optional
pip install arabert arabic-reshaper python-bidi
```
A T4 GPU (Colab/Kaggle) is enough. Results are cached in `results/cache/` so reruns are fast.

## Running
Open each notebook and "Run all" (from `notebooks/` or the repo root; the first cell switches to the repo root).
B and D are independent. D's F-section builds baseline predictions only if
`predictions/tfidf_lr_seed42.csv` is missing (set `BASELINE_NAME` in the config cell if yours has another name).

Switches in D's config cell: `RUN_MARBERT` (frozen MARBERT comparison), `RUN_D2_RETRAINS` (F2/F3 retrains of the D2 head).

## Protocol
- All choices (preprocessing, pooling, layers, C, head hyper-parameters, epochs, thresholds) are made on val.
- Test is scored only for the retained models (B1, B2, D1, D2 x3 seeds); Part F is analysis, never selection.
- Rows are logged to `results/results.csv` (val + test, train time, trained/total params, CPU latency);
  predictions go to `predictions/<model>_seed<k>.csv`.

## Max sequence length (data-driven)
Each notebook measures token lengths on **train** (per tokenizer x preprocessing), prints mean / median / Q3 / Tukey fence /
p99 / p99.5 / max, applies `LENGTH_RULE` (default `"p99.5"`; options `tukey`, `p99`, `mean+3std`, `max`), rounds up to a multiple of 8,
plots the distribution with the chosen cap, and shows the % of truncated tweets on train/val/test. D2 additionally caps at `RAM_CAP_TOK`.
Change `LENGTH_RULE` in the cell and re-run that cell (delete cached files in `results/cache/` only if the cap changes and you want fresh features;
the cache key already contains the cap).

## Outputs
- `results/figures/`: `token_length_distribution.png`, `b1_token_length_distribution.png`, `b2_token_length_distribution.png`, `b1_template_sensitivity.png`, `b2_sentiment_confusion.png`, `d2_attention_5_tweets.png`
  (report figure), `f1_threshold_test_D.png` (report figure)
- `results/error_analysis_<D1|D2>.csv`: fill the `category` column by hand, then rerun the last F6 cell.

## Things to verify
- `analysis.is_quoted` assumes the CSV artefact from audit 4.1 is surrounding double quotes.
- Laughter / repeated-punctuation regexes in `analysis.py` are heuristics; check against your audit.
- ArabertPreprocessor arguments and DeBERTa attention output vary by library version (both wrapped to degrade gracefully).
- Picking the best of many configurations on one val set is slightly optimistic; mention it in the report.

## Mapping to the TP
B1/Q4: B notebook section 2 | B2/Q5: section 3 | D1: D section 2 | D2/Q7: D section 3 |
final models, latency, logging: D section 4 | F1-F6 (D models): D section 5 | Q8, Q10, Q11 inputs: D section 6 and first cells.

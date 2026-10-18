import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "results.csv"
PREDICTIONS = ROOT / "predictions"


def metrics(y_true, proba, threshold):
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def f1_by_threshold(y_true, proba, thresholds=np.linspace(0.05, 0.95, 91)):
    proba = np.asarray(proba)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in thresholds]
    return pd.Series(scores, index=thresholds)


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    return tuple(np.percentile(scores, [2.5, 97.5]))


def mcnemar(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    if only_a + only_b == 0:
        return 1.0
    return binomtest(only_a, only_a + only_b, 0.5).pvalue


def latency_per_tweet(predict, texts):
    start = time.time()
    predict(texts)
    return (time.time() - start) / len(texts)


def log_result(key_cols=("model", "preprocessing", "seed"), **row):
    RESULTS.parent.mkdir(exist_ok=True)
    new_row = pd.DataFrame([row])
    if RESULTS.exists() and RESULTS.stat().st_size > 0:
        table = pd.read_csv(RESULTS)
        # Determine matching subset columns that exist in both table and new row
        match_cols = [c for c in key_cols if c in table.columns and c in new_row.columns]
        if match_cols:
            mask = np.ones(len(table), dtype=bool)
            for col in match_cols:
                mask &= (table[col].astype(str) == str(row[col]))
            table = table[~mask]
        table = pd.concat([table, new_row], ignore_index=True)
    else:
        table = new_row
    table.to_csv(RESULTS, index=False)

def show(name, scores, threshold, seconds):
    print(f"{name:<28} f1={scores['f1']:.4f}  pr_auc={scores['pr_auc']:.4f}  "
          f"threshold={threshold:.2f}  {seconds:.0f}s")

def plot_results(table, metrics=("f1", "pr_auc")):
    data = table.sort_values(metrics[0])

    sns.set_theme(style="whitegrid")
    ax = data[list(metrics)].plot.barh(
        figsize=(9, 0.5 * len(data) + 1),
        color=["#4C78A8", "#F2A541"],
        width=0.7,
    )

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    ax.set_xlim(0, data[list(metrics)].max().max() * 1.12)
    ax.set_xlabel("Validation Score")
    ax.set_ylabel("")
    ax.legend(title="Metric", frameon=False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def save_predictions(name, seed, ids, y_true, proba, threshold):
    PREDICTIONS.mkdir(exist_ok=True)
    proba = np.asarray(proba)
    pd.DataFrame({
        "id": ids,
        "y_true": y_true,
        "y_pred": (proba >= threshold).astype(int),
        "proba": proba,
    }).to_csv(PREDICTIONS / f"{name}_seed{seed}.csv", index=False)
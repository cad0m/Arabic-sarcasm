"""Generalisation / robustness helpers (Part F) - model agnostic, they only need predictions
or a predict(list_of_texts) -> proba function."""
import re
from pathlib import Path

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.evaluate import PREDICTIONS, f1_by_threshold
from src.preprocess import URL_PATTERN

MENTION = re.compile(r"@\w+")
LAUGH = re.compile(r"(?<!\w)(?:ه{3,}|(?:ها){2,}ه*|خ{3,}|(?:[Hh][Aa]){2,}[Hh]?|[Ll][Oo][Ll]+)(?!\w)")
REPEAT_PUNCT = re.compile(r"([!?؟.،…*~])\1+|[!?؟]{2,}")


def _spaces(t):
    return " ".join(t.split())


# ---- detectors
def has_url(t): return bool(URL_PATTERN.search(t))
def has_mention(t): return bool(MENTION.search(t))
def has_emoji(t): return emoji.emoji_count(t) > 0
def has_laugh(t): return bool(LAUGH.search(t))
def has_repeat_punct(t): return bool(REPEAT_PUNCT.search(t))
def is_quoted(t):
    s = t.strip()
    return len(s) > 1 and s[0] == s[-1] == '"'   # CSV-export artefact (see strip_quotes)


# ---- removers
def remove_emojis(t): return _spaces(emoji.replace_emoji(t, " "))
def remove_laughter(t): return _spaces(LAUGH.sub(" ", t))
def remove_repeat_punct(t): return _spaces(REPEAT_PUNCT.sub(" ", t))
def remove_urls_mentions(t): return _spaces(MENTION.sub(" ", URL_PATTERN.sub(" ", t)))
def remove_all_cues(t): return remove_repeat_punct(remove_laughter(remove_emojis(t)))


CUES = {
    "emoji": (has_emoji, remove_emojis),
    "laughter": (has_laugh, remove_laughter),
    "repeated_punct": (has_repeat_punct, remove_repeat_punct),
    "all_cues": (lambda t: has_emoji(t) or has_laugh(t) or has_repeat_punct(t), remove_all_cues),
}


def drop_random_words(t, n, rng):
    words = t.split()
    if n <= 0 or len(words) <= n:
        return t
    drop = set(rng.choice(len(words), n, replace=False).tolist())
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def cue_ablation(texts, y_true, y_pred, predict_proba, threshold, seed=42):
    """F4. Among sarcastic tweets that the model detects, remove a cue and measure how many
    stop being detected. Control: remove the same number of random words instead."""
    texts = pd.Series(list(texts)).reset_index(drop=True)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    detected = (y_true == 1) & (y_pred == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for name, (has, remove) in CUES.items():
        sel = np.where(detected & texts.map(has).values)[0]
        if len(sel) == 0:
            continue
        original = texts.iloc[sel].tolist()
        stripped = [remove(t) for t in original]
        removed = [max(1, len(a.split()) - len(b.split())) for a, b in zip(original, stripped)]
        control = [drop_random_words(a, n, rng) for a, n in zip(original, removed)]
        kept_cue = (predict_proba(stripped) >= threshold).mean()
        kept_ctl = (predict_proba(control) >= threshold).mean()
        rows.append({"cue": name, "n_detected_with_cue": len(sel),
                     "still_detected": kept_cue, "drop": 1 - kept_cue,
                     "control_drop (random words)": 1 - kept_ctl,
                     "lost_tweets": int(round((1 - kept_cue) * len(sel))),
                     "share_of_all_detected": (1 - kept_cue) * len(sel) / max(1, detected.sum())})
    return pd.DataFrame(rows).set_index("cue")


def subgroup_scores(df, groups):
    """df needs y_true, y_pred, proba. groups: name -> boolean mask."""
    rows = []
    for name, mask in groups.items():
        sub = df[np.asarray(mask)]
        if len(sub) == 0:
            continue
        y, p = sub["y_true"].values, sub["y_pred"].values
        rows.append({"subset": name, "n": len(sub), "sarcasm_rate": y.mean(),
                     "f1": f1_score(y, p, zero_division=0),
                     "precision": precision_score(y, p, zero_division=0),
                     "recall": recall_score(y, p, zero_division=0),
                     "pr_auc": average_precision_score(y, sub["proba"]) if y.sum() > 0 else np.nan})
    return pd.DataFrame(rows).set_index("subset")


def prior_shift(y_val, y_test, proba_test, val_threshold):
    """F1. Test F1 at the val threshold vs the a-posteriori optimum (analysis only)."""
    curve = f1_by_threshold(y_test, proba_test)
    at_val = f1_score(y_test, np.asarray(proba_test) >= val_threshold, zero_division=0)
    return {"val_rate": float(np.mean(y_val)), "test_rate": float(np.mean(y_test)),
            "val_threshold": float(val_threshold), "test_f1_at_val_threshold": at_val,
            "best_test_threshold": float(curve.idxmax()), "best_test_f1": float(curve.max()),
            "threshold_gap": float(curve.idxmax() - val_threshold), "f1_gap": float(curve.max() - at_val)}


def plot_f1_threshold(curves, val_thresholds, path=None):
    """curves: name -> pd.Series(f1 by threshold) ; val_thresholds: name -> threshold chosen on val."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, s) in enumerate(curves.items()):
        line, = ax.plot(s.index, s.values, label=name)
        ax.axvline(val_thresholds[name], color=line.get_color(), ls=":", alpha=.8)
        ax.scatter([s.idxmax()], [s.max()], marker="*", s=120, color=line.get_color(), zorder=3)
    ax.set_xlabel("decision threshold"); ax.set_ylabel("F1 (sarcastic class) on test")
    ax.set_title("F1 = f(threshold) on test  (dotted = threshold chosen on val, star = a-posteriori optimum)")
    ax.legend(frameon=False); ax.grid(alpha=.3)
    plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.show()


def error_sample(df, n=15, seed=42):
    """F6. n false negatives + n false positives (df needs y_true, y_pred)."""
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
    out = pd.concat([fn.sample(min(n, len(fn)), random_state=seed).assign(error="FN"),
                     fp.sample(min(n, len(fp)), random_state=seed).assign(error="FP")])
    return out


ERROR_CATEGORIES = ["world_knowledge_or_context", "irony_without_marker", "misread_marker",
                    "dialect", "annotation_noise"]


def load_predictions(name, seed=42):
    path = PREDICTIONS / f"{name}_seed{seed}.csv"
    return pd.read_csv(path) if path.exists() else None

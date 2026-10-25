"""Helpers to use a (frozen) pre-trained encoder as a feature extractor."""
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(name, device=None):
    """Tokenizer + frozen encoder in eval mode."""
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return tok, enc


def token_lengths(tok, texts, max_length=512):
    ids = tok(list(texts), truncation=True, max_length=max_length)["input_ids"]
    return np.array([len(x) for x in ids])


def _batches(texts, batch_size):
    order = np.argsort([len(t) for t in texts])  # similar lengths together = less padding
    for i in range(0, len(order), batch_size):
        yield order[i:i + batch_size]


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def extract_pooled(texts, tok, enc, max_length=128, batch_size=64):
    """[CLS] and attention-mask-weighted mean of tokens, from the last layer and from the
    average of the last 4 layers. Returns a dict of (n, hidden) float32 arrays + 'seconds'."""
    texts = list(texts)
    device = next(enc.parameters()).device
    n, h = len(texts), enc.config.hidden_size
    out = {k: np.zeros((n, h), dtype=np.float32) for k in ("cls_last", "mean_last", "cls_last4", "mean_last4")}
    start = time.time()
    for idx in _batches(texts, batch_size):
        batch = tok([texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        with _autocast(device):
            hidden = enc(**batch, output_hidden_states=True).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1).float()
        views = {"last": hidden[-1].float(), "last4": torch.stack(hidden[-4:]).float().mean(0)}
        for tag, states in views.items():
            out[f"cls_{tag}"][idx] = states[:, 0].cpu().numpy()
            out[f"mean_{tag}"][idx] = ((states * mask).sum(1) / mask.sum(1)).cpu().numpy()
    out["seconds"] = np.array(time.time() - start)
    return out


@torch.no_grad()
def extract_tokens(texts, tok, enc, max_length=96, batch_size=64):
    """Last-layer states of every token (fp16 to save RAM), padding mask and token ids."""
    texts = list(texts)
    device = next(enc.parameters()).device
    n, h = len(texts), enc.config.hidden_size
    states = np.zeros((n, max_length, h), dtype=np.float16)
    masks = np.zeros((n, max_length), dtype=bool)
    ids = np.full((n, max_length), tok.pad_token_id, dtype=np.int64)
    start = time.time()
    for idx in _batches(texts, batch_size):
        batch = tok([texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        with _autocast(device):
            last = enc(**batch).last_hidden_state
        width = last.shape[1]
        states[idx, :width] = last.float().cpu().numpy().astype(np.float16)
        masks[idx, :width] = batch["attention_mask"].bool().cpu().numpy()
        ids[idx, :width] = batch["input_ids"].cpu().numpy()
    return {"states": states, "masks": masks, "ids": ids, "seconds": np.array(time.time() - start)}


def cached_npz(path, fn):
    """Run fn() once, store the dict of arrays, reload on later runs (no need to re-extract)."""
    path = Path(path)
    if path.exists():
        return dict(np.load(path))
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------- max-length selection
def length_stats(lengths):
    """Descriptive stats of token lengths, incl. the Tukey fence (Q3 + 1.5 IQR) and the longest non-outlier."""
    L = np.asarray(lengths)
    q1, q3 = np.percentile(L, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    return {"mean": float(L.mean()), "std": float(L.std()), "median": float(np.median(L)),
            "q1": float(q1), "q3": float(q3), "tukey_fence": float(fence),
            "max_non_outlier": int(L[L <= fence].max()), "p95": float(np.percentile(L, 95)),
            "p99": float(np.percentile(L, 99)), "p99.5": float(np.percentile(L, 99.5)),
            "mean+3std": float(L.mean() + 3 * L.std()), "max": int(L.max())}


LENGTH_RULES = {"tukey": "max_non_outlier", "p99": "p99", "p99.5": "p99.5", "mean+3std": "mean+3std", "max": "max"}


def choose_max_length(stats, rule="p99.5", multiple=8, cap=512):
    """Apply a rule to the stats, round up to a multiple of 8 (GPU friendly), clip to the model limit."""
    value = stats[LENGTH_RULES[rule]]
    return int(min(cap, np.ceil(value / multiple) * multiple))


def truncation_table(lengths_by_split, caps):
    """Share of tweets longer than each candidate cap, for every split. caps: label -> int."""
    import pandas as pd
    return pd.DataFrame({s: {f"{k} = {v}": float((np.asarray(L) > v).mean()) for k, v in caps.items()}
                         for s, L in lengths_by_split.items()})


def plot_lengths(series, chosen=None, path=None, log_y=False, title="Token length distribution (train)"):
    """series: name -> lengths. chosen: name -> final cap (drawn as a thick red line)."""
    import matplotlib.pyplot as plt
    n = len(series)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 5.4), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [4, 1]})
    for k, (name, L) in enumerate(series.items()):
        L = np.asarray(L); s = length_stats(L); ax, bx = axes[0, k], axes[1, k]
        ax.hist(L, bins=int(min(60, L.max() - L.min() + 1)), color="#9ecae1", edgecolor="white")
        marks = [("mean", s["mean"], "#e6550d", "-"), ("median", s["median"], "#31a354", "-"),
                 ("Q3", s["q3"], "#636363", ":"), ("Tukey fence", s["tukey_fence"], "#756bb1", "--"),
                 ("p99", s["p99"], "#d94801", "--"), ("p99.5", s["p99.5"], "#a63603", "--"),
                 ("max", s["max"], "black", "-.")]
        for label, v, c, ls in marks:
            ax.axvline(v, color=c, ls=ls, lw=1.3, label=f"{label} = {v:.0f}")
        if chosen and name in chosen:
            ax.axvline(chosen[name], color="red", lw=3, alpha=.8, label=f"CHOSEN = {chosen[name]}")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(name, fontsize=10); ax.legend(fontsize=7, frameon=False)
        ax.set_ylabel("tweets")
        bx.boxplot(L, vert=False, whis=1.5, flierprops={"markersize": 2, "alpha": .4})
        bx.set_yticks([]); bx.set_xlabel("tokens per tweet (box = IQR, whiskers = 1.5 IQR, dots = outliers)", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.show()

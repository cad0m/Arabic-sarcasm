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

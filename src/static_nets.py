import math
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.evaluate import metrics
from src.preprocess import normalize_letters, remove_diacritics, remove_kashida, replace_mentions, replace_urls
from src.train import best_threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD, UNK = 0, 1

MAX_LEN = 64          # words kept per tweet
EMB_DIM = 100         # same size for the random embedding and for AraVec
MIN_FREQ = 2          # a word must appear at least twice in train
HIDDEN = 128          # LSTM size per direction

# training budget, the same for every model
EPOCHS = 20
PATIENCE = 3          # stop after this many epochs without a better val F1
BATCH_SIZE = 64
LR = 1e-3


# ---------- text, vocabulary, OOV ----------

def tokenize(text):
    """Normalize, then split into words, <URL> / <USER> placeholders, and single punctuation or emoji characters."""
    text = replace_mentions(replace_urls(text))
    text = normalize_letters(remove_kashida(remove_diacritics(text)))
    return re.findall(r"<URL>|<USER>|\w+|[^\w\s]", text)


def build_vocab(token_lists):
    """Words seen at least MIN_FREQ times. Built on train only."""
    counts = Counter(t for tokens in token_lists for t in tokens)
    words = [w for w, c in counts.most_common() if c >= MIN_FREQ]
    return {w: i for i, w in enumerate(["<PAD>", "<UNK>"] + words)}


def encode(tokens, vocab):
    ids = [vocab.get(t, UNK) for t in tokens[:MAX_LEN]]
    return ids or [UNK]          # never an empty sequence


def oov_rate(token_lists, vocab):
    """Share of tokens that are not in the vocabulary."""
    total = sum(len(tokens) for tokens in token_lists)
    unknown = sum(1 for tokens in token_lists for t in tokens if t not in vocab)
    return unknown / total


# ---------- AraVec ----------

def clean_str(text):
    """AraVec's own cleaning, applied before looking a word up in its vocabulary."""
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى",
              "\\", "\n", "\t", "&quot;", "?", "؟", "!"]
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا",
               "", "", "", "ي", "", " ", " ", " ", " ? ", " ؟ ", " ! "]

    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)     # diacritics
    text = re.sub(r"(.)\1+", r"\1\1", text)                      # repeated letters
    text = text.replace("وو", "و").replace("يي", "ي").replace("اا", "ا")
    for s, r in zip(search, replace):
        text = text.replace(s, r)
    return text.strip()


def load_aravec(vocab, path):
    """Embedding matrix (vocab size x EMB_DIM). Returns the matrix and the share of words found in AraVec."""
    from gensim.models import Word2Vec
    wv = Word2Vec.load(str(path)).wv
    assert wv.vector_size == EMB_DIM, f"AraVec has {wv.vector_size} dimensions, EMB_DIM is {EMB_DIM}"

    rng = np.random.RandomState(0)
    matrix = rng.normal(0, 0.1, (len(vocab), EMB_DIM)).astype("float32")
    matrix[PAD] = 0

    found = 0
    for word, i in vocab.items():
        key = clean_str(word)
        if i > UNK and key in wv:
            matrix[i] = wv[key]
            found += 1
    return torch.tensor(matrix), found / (len(vocab) - 2)


# ---------- models ----------

def sinusoidal_positions(length, d_model):
    position = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


def run_lstm(embedding, lstm, ids):
    """Runs a BiLSTM on padded ids, ignoring the padding. Returns (B x T x 2H outputs, B x T mask)."""
    mask = ids != PAD
    lengths = mask.sum(dim=1).cpu()
    packed = pack_padded_sequence(embedding(ids), lengths, batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
    return out, mask


class BiLSTM(nn.Module):
    """C1: BiLSTM + max-pooling over the words."""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.embedding_dim, HIDDEN, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(2 * HIDDEN, 2)

    def forward(self, ids):
        h, mask = run_lstm(self.embedding, self.lstm, ids)
        h = h.masked_fill(~mask.unsqueeze(-1), -1e9)          # padding never wins the max
        return self.out(self.dropout(h.max(dim=1).values))


class AttentionPooling(nn.Module):
    """alpha_i = softmax(v^T tanh(W h_i)),  c = sum_i alpha_i h_i"""

    def __init__(self, dim, size=64):
        super().__init__()
        self.W = nn.Linear(dim, size)
        self.v = nn.Linear(size, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)    # B x T
        scores = scores.masked_fill(~mask, -1e9)              # no attention on padding
        alpha = torch.softmax(scores, dim=1)
        self.alpha = alpha.detach()                           # kept to look at the weights later
        return (alpha.unsqueeze(-1) * h).sum(dim=1)


class BiLSTMAttention(nn.Module):
    """C2: BiLSTM + attention pooling."""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.embedding_dim, HIDDEN, batch_first=True, bidirectional=True)
        self.attention = AttentionPooling(2 * HIDDEN)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(2 * HIDDEN, 2)

    def forward(self, ids):
        h, mask = run_lstm(self.embedding, self.lstm, ids)
        return self.out(self.dropout(self.attention(h, mask)))


class MiniTransformer(nn.Module):
    """C3: 2 encoder layers, d_model 128, 4 heads, sinusoidal positions (optional), padding mask, mean-pooling."""

    def __init__(self, embedding, positions=True, d_model=128, heads=4, layers=2):
        super().__init__()
        self.embedding = embedding
        self.project = nn.Linear(embedding.embedding_dim, d_model)
        self.positions = positions
        self.register_buffer("pe", sinusoidal_positions(MAX_LEN, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(d_model, 2)

    def forward(self, ids):
        mask = ids != PAD
        x = self.project(self.embedding(ids))
        if self.positions:
            x = x + self.pe[:ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.out(self.dropout(pooled))


def build_model(arch, vocab_size, matrix=None, freeze=False, positions=True):
    """arch: "bilstm" (C1), "bilstm_att" (C2) or "transformer" (C3).
    matrix: AraVec matrix, or None for a random embedding. freeze: keep the AraVec vectors fixed."""
    if matrix is None:
        embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD)
    else:
        embedding = nn.Embedding.from_pretrained(matrix.clone(), freeze=freeze, padding_idx=PAD)

    if arch == "bilstm":
        model = BiLSTM(embedding)
    elif arch == "bilstm_att":
        model = BiLSTMAttention(embedding)
    else:
        model = MiniTransformer(embedding, positions)
    return model.to(DEVICE)


# ---------- training and prediction ----------

def class_weights(y):
    """Rare class gets a bigger weight. Computed on train only."""
    return (len(y) / (2 * np.bincount(y))).tolist()


def pad_batch(seqs):
    out = np.zeros((len(seqs), max(len(s) for s in seqs)), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return torch.tensor(out).to(DEVICE)


def predict_proba(model, x):
    """P(sarcastic) for a list of id lists."""
    model.eval()
    probas = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            logits = model(pad_batch(x[i:i + 256]))
            probas.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probas)


def train_model(model, x_train, y_train, x_val, y_val, weights=None):
    """Train with early stopping on val F1 (sarcastic class).
    weights: class weights for the loss, or None for the standard loss.
    The model is left at its best epoch. Returns the training time in seconds."""
    weight_tensor = None if weights is None else torch.tensor(weights, dtype=torch.float).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(x_train))
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            ids = pad_batch([x_train[j] for j in idx])
            y = torch.tensor(y_train[idx]).to(DEVICE)

            loss = F.cross_entropy(model(ids), y, weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check on val after every epoch
        val_proba = predict_proba(model, x_val)
        f1 = metrics(y_val, val_proba, best_threshold(y_val, val_proba))["f1"]
        print(f"epoch {epoch + 1}: val f1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1, bad_epochs = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return time.time() - start


# ---------- checkpoints (for Part F) ----------

def save_checkpoint(path, model, vocab, arch, positions, threshold):
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "vocab": vocab, "arch": arch, "positions": positions, "threshold": threshold}, path)


def load_checkpoint(path):
    """Returns the model, its vocabulary and its validation threshold."""
    ckpt = torch.load(path, map_location=DEVICE)
    model = build_model(ckpt["arch"], len(ckpt["vocab"]), positions=ckpt["positions"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["vocab"], ckpt["threshold"]

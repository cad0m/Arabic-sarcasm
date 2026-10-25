import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, get_linear_schedule_with_warmup

from src.evaluate import metrics
from src.train import best_threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# training settings, the same for every run
LR = 2e-5
BATCH_SIZE = 32
EPOCHS = 4
PATIENCE = 2          # stop after this many epochs without a better val F1
WARMUP = 0.1          # 10% of the steps
WEIGHT_DECAY = 0.01

PLMS = {
    "arabert": {"hf_name": "aubmindlab/bert-base-arabertv02-twitter",
                "max_len": 64, "prep": "arabert_preprocessor"},
    "marbert": {"hf_name": "UBC-NLP/MARBERTv2",
                "max_len": 96, "prep": "whitespace_only"},
}



class SingleTask(nn.Module):
    def __init__(self, hf_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_name, add_pooling_layer=False)
        self.sarcasm = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """Sarcasm logits."""
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.sarcasm(cls)

    def loss(self, batch, y_sarcasm, y_sentiment, weights):
        return F.cross_entropy(self(**batch), y_sarcasm, weight=weights)


class MultiTask(nn.Module):
    def __init__(self, hf_name, lam):
        super().__init__()
        self.lam = lam
        self.encoder = AutoModel.from_pretrained(hf_name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.sarcasm = nn.Linear(hidden, 2)
        self.sentiment = nn.Linear(hidden, 3)

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        return self.sarcasm(self.encode(input_ids, attention_mask))

    def loss(self, batch, y_sarcasm, y_sentiment, weights):
        cls = self.encode(**batch)
        sarcasm_loss = F.cross_entropy(self.sarcasm(cls), y_sarcasm, weight=weights)
        sentiment_loss = F.cross_entropy(self.sentiment(cls), y_sentiment)
        return sarcasm_loss + self.lam * sentiment_loss


# ---------- text ----------

def prepare_text(key, texts):
    if key == "arabert":
        from arabert.preprocess import ArabertPreprocessor
        preprocessor = ArabertPreprocessor(model_name=PLMS[key]["hf_name"])
        return [preprocessor.preprocess(t) for t in texts]
    return [" ".join(t.split()) for t in texts]   # marbert: only clean the spaces


def make_batch(tokenizer, texts, max_len):
    batch = tokenizer(texts, truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt", return_token_type_ids=False)
    return {name: tensor.to(DEVICE) for name, tensor in batch.items()}


# ---------- training and prediction ----------

def class_weights(y):
    """Rare class gets a bigger weight. Computed on train only."""
    counts = np.bincount(y)
    return torch.tensor(len(y) / (2 * counts), dtype=torch.float).to(DEVICE)


def predict_proba(model, tokenizer, texts, max_len):
    """P(sarcastic) for every text."""
    model.eval()
    probas = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = make_batch(tokenizer, texts[i:i + 64], max_len)
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                logits = model(**batch)
            probas.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probas)


def train_model(model, tokenizer, x_train, y_sarcasm, y_sentiment, x_val, y_val, max_len):
    """Fine-tune with early stopping on val F1 (sarcastic class).
    The model is left at its best epoch. Returns the training time in seconds."""
    weights = class_weights(y_sarcasm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = int(np.ceil(len(x_train) / BATCH_SIZE)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(x_train))
        for i in tqdm(range(0, len(order), BATCH_SIZE), desc=f"epoch {epoch + 1}", leave=False):
            idx = order[i:i + BATCH_SIZE]
            batch = make_batch(tokenizer, [x_train[j] for j in idx], max_len)
            y_sar = torch.tensor(y_sarcasm[idx]).to(DEVICE)
            y_sen = torch.tensor(y_sentiment[idx]).to(DEVICE)

            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = model.loss(batch, y_sar, y_sen, weights)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # check on val after every epoch
        val_proba = predict_proba(model, tokenizer, x_val, max_len)
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
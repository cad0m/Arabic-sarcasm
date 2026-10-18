import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from transformers import set_seed as set_transformers_seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_transformers_seed(seed)


def best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def fit_with_threshold(model, X_train, y_train, X_val, y_val):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    threshold = best_threshold(y_val, model.predict_proba(X_val)[:, 1])
    return threshold, train_time


def oversample(train, seed=42):
    minority = train[train["sarcasm"] == 1]
    missing = int((train["sarcasm"] == 0).sum() - len(minority))
    extra = minority.sample(missing, replace=True, random_state=seed)
    return pd.concat([train, extra]).reset_index(drop=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
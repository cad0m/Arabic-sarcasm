from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import normalize_letters, remove_diacritics, remove_kashida

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"
SENTIMENT = {"NEG": 0, "NEU": 1, "POS": 2}


def load_split():
    full = pd.read_csv(BASE + "training_data.csv")
    test = pd.read_csv(BASE + "testing_data.csv")
    for f in (full, test):
        f["sarcasm"] = f["sarcasm"].astype(str).str.lower().map({"true": 1, "false": 0})
        f["sentiment_id"] = f["sentiment"].map(SENTIMENT)
    full["id"], test["id"] = range(len(full)), range(len(test))
    train, val = train_test_split(full, test_size=0.10, random_state=42, stratify=full["sarcasm"])
    return train, val, test


def add_key(frame):
    exact = frame["tweet"].map(lambda t: " ".join(t.split()))
    norm = exact.map(lambda t: normalize_letters(remove_diacritics(remove_kashida(t))))
    return frame.assign(key_exact=exact, key=norm)


def clean_train(train, val, test):
    r = {"start": len(train)}

    for k in ("key_exact", "key"):
        r[f"{k}/dup_train"] = int(train[k].duplicated().sum())
        r[f"{k}/dup_val"] = int(val[k].duplicated().sum())
        r[f"{k}/dup_test"] = int(test[k].duplicated().sum())
        r[f"{k}/train_in_test"] = int(train[k].isin(test[k]).sum())
        r[f"{k}/test_in_train"] = int(test[k].isin(train[k]).sum())

    
    rows = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    conflicts = rows[rows.groupby("key")["sarcasm"].transform("nunique") > 1]
    conflicts = conflicts.sort_values(["key", "split", "id"])
    r["conflict_groups"], r["conflict_rows"] = conflicts["key"].nunique(), len(conflicts)

    
    for name, keys in [("in_test", test["key"]), ("in_val", val["key"]), ("conflicting", conflicts["key"])]:
        bad = train["key"].isin(keys)
        r[f"removed_{name}"] = int(bad.sum())
        train = train[~bad]

    r["removed_duplicates"] = int(train["key"].duplicated().sum())
    train = train.drop_duplicates("key", keep="first")

    r["end"] = len(train)
    return train.reset_index(drop=True), r, conflicts


def get_data():
    train, val, test = (add_key(f) for f in load_split())
    train, report, conflicts = clean_train(train, val, test)
    Path("results").mkdir(exist_ok=True)
    conflicts.to_csv("results/conflicts.csv", index=False)  
    return train, val.reset_index(drop=True), test.reset_index(drop=True), report
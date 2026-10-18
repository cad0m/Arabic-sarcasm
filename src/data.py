import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import build_pipeline

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"
KEY_STEPS = ["html", "urls", "mentions", "diacritics", "kashida", "letters", "elongation"]


def load_split():
    full = pd.read_csv(BASE + "training_data.csv")
    test = pd.read_csv(BASE + "testing_data.csv")
    full["id"], test["id"] = range(len(full)), range(len(test))
    for frame in (full, test):
        frame["sarcasm"] = frame["sarcasm"].astype(str).str.lower().map({"true": 1, "false": 0})
    train, val = train_test_split(full, test_size=0.10, random_state=42, stratify=full["sarcasm"])
    return train, val, test


def add_key(frame):
    prep = build_pipeline(KEY_STEPS)
    key = frame["tweet"].apply(prep).str.replace(r"[^\w\s]", " ", regex=True).str.split().str.join(" ")
    return frame.assign(key=key.where(key != "", frame["tweet"]))


def clean_train(train, val, test):
    report = {"start": len(train)}

    for name, other in [("in_test", test), ("in_val", val)]:
        found = train["key"].isin(other["key"])
        report[name] = int(found.sum())
        train = train[~found]

    conflicting = train.groupby("key")["sarcasm"].transform("nunique") > 1
    report["conflicting"] = int(conflicting.sum())
    train = train[~conflicting]

    report["duplicates"] = int(train["key"].duplicated().sum())
    train = train.drop_duplicates("key")

    report["end"] = len(train)
    return train.reset_index(drop=True), report


def get_data():
    train, val, test = (add_key(f) for f in load_split())
    train, report = clean_train(train, val, test)
    return train, val.reset_index(drop=True), test, report
"""Load, clean and split the ISOT Fake/Real news dataset.

Expected files (download from Kaggle, see README):
    data/raw/Fake.csv
    data/raw/True.csv
Label convention: 0 = FAKE, 1 = REAL
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# The raw dataset leaks the label through source artifacts such as
# "WASHINGTON (Reuters) -" (only in real news) and "Featured image via ...".
# We strip them so the model has to learn from content/style instead.
_REUTERS_PREFIX = re.compile(r"^\s*(?:[A-Z][A-Za-z.,' /-]{0,60})?\(Reuters\)\s*[-–—]?\s*")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PIC = re.compile(r"pic\.twitter\.com/\S+")
_FEATURED = re.compile(r"Featured image via[^.\n]*\.?", re.I)
_LEAKS = re.compile(r"\(Reuters\)|\bReuters\b|21st Century Wire( says)?|Getty Images", re.I)
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text)
    text = _REUTERS_PREFIX.sub("", text, count=1)
    text = _URL.sub(" ", text)
    text = _PIC.sub(" ", text)
    text = _FEATURED.sub(" ", text)
    text = _LEAKS.sub(" ", text)
    return _WS.sub(" ", text).strip()


def build_splits(raw_dir="data/raw", out_dir="data/processed"):
    raw, out = Path(raw_dir), Path(out_dir)
    fake_p, true_p = raw / "Fake.csv", raw / "True.csv"
    if not fake_p.exists() or not true_p.exists():
        raise FileNotFoundError(
            "Put Fake.csv and True.csv in data/raw/ "
            "(Kaggle: clmentbisaillon/fake-and-real-news-dataset).")

    fake = pd.read_csv(fake_p)
    real = pd.read_csv(true_p)
    fake["label"], real["label"] = 0, 1
    df = pd.concat([fake, real], ignore_index=True)

    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean_text)
    df = df[df["text"].str.split().str.len() >= 20]
    df = df.drop_duplicates(subset="text")[["text", "label"]].reset_index(drop=True)

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"(real share: {df['label'].mean():.2%})")
    return train, val, test


def load_splits(raw_dir="data/raw", out_dir="data/processed"):
    out = Path(out_dir)
    if not (out / "train.csv").exists():
        return build_splits(raw_dir, out_dir)
    return tuple(pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test"))


if __name__ == "__main__":
    build_splits()

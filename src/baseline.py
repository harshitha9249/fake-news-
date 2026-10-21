"""TF-IDF + Logistic Regression baseline."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.data import load_splits
from src.utils import compute_metrics, plot_confusion, save_json


def main():
    train, val, test = load_splits()
    pipe = make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                        stop_words="english", sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=4.0),
    )
    pipe.fit(train["text"], train["label"])

    for name, df in [("val", val), ("test", test)]:
        m = compute_metrics(df["label"], pipe.predict(df["text"]))
        print(name, m)
        if name == "test":
            save_json(m, "results/baseline_metrics.json")
            plot_confusion(df["label"], pipe.predict(df["text"]),
                           "results/baseline_confusion.png", "TF-IDF + LogReg")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/tfidf_logreg.joblib")


if __name__ == "__main__":
    main()

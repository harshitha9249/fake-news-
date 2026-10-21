"""Evaluate DistilBERT on the held-out test set and compare with the baseline."""
import json
from pathlib import Path

from src.data import load_splits
from src.predict import predict_proba
from src.utils import compute_metrics, plot_confusion, save_json


def main():
    _, _, test = load_splits()
    probs = predict_proba(test["text"].tolist())
    preds = probs.argmax(axis=1)

    m = compute_metrics(test["label"], preds)
    print("DistilBERT:", m)
    save_json(m, "results/distilbert_metrics.json")
    plot_confusion(test["label"], preds, "results/distilbert_confusion.png", "DistilBERT")

    base_path = Path("results/baseline_metrics.json")
    rows = [("DistilBERT", m)]
    if base_path.exists():
        rows.insert(0, ("TF-IDF + LogReg", json.loads(base_path.read_text())))

    lines = ["| Model | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for name, r in rows:
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} |")
    Path("results/comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

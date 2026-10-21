"""Fine-tune DistilBERT for fake news classification."""
import argparse

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from src.data import SEED, load_splits
from src.utils import compute_metrics

BASE_MODEL = "distilbert-base-uncased"


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = [int(x) for x in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def hf_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_metrics(labels, np.argmax(logits, axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--sample", type=int, default=0,
                    help="Use only N training rows (quick CPU test).")
    ap.add_argument("--out", default="models/distilbert-fake-news")
    args = ap.parse_args()

    set_seed(SEED)
    train, val, test = load_splits()
    if args.sample:
        train = train.sample(args.sample, random_state=SEED)
        val = val.sample(min(args.sample, len(val)), random_state=SEED)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2,
        id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})

    train_ds = NewsDataset(train["text"], train["label"], tok, args.max_len)
    val_ds = NewsDataset(val["text"], val["label"], tok, args.max_len)
    test_ds = NewsDataset(test["text"], test["label"], tok, args.max_len)

    targs = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=hf_metrics)
    trainer.train()
    print("TEST:", trainer.evaluate(test_ds, metric_key_prefix="test"))

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()

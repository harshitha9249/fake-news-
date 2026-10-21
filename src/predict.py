"""Inference helpers shared by evaluation, SHAP and the Gradio app."""
import os
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Local folder or a Hugging Face Hub id (e.g. "your-username/distilbert-fake-news")
MODEL_PATH = os.getenv("MODEL_ID", "models/distilbert-fake-news")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load(model_path=None):
    path = model_path or MODEL_PATH
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
    return tok, model


def predict_proba(texts, batch_size=32, max_len=256, model_path=None):
    """Returns array of shape (n, 2): [P(FAKE), P(REAL)]."""
    tok, model = load(model_path)
    texts = list(texts)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(DEVICE)
            out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out)

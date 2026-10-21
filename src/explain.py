"""SHAP explanations with word-level highlights.

Colour convention (relative to the FAKE class):
    red   -> word pushes the prediction toward FAKE
    green -> word pushes the prediction toward REAL
"""
import argparse
import html
from functools import lru_cache
from pathlib import Path

import numpy as np
import shap
from transformers import pipeline

from src.predict import load, predict_proba

MAX_WORDS = 150  # SHAP cost grows quickly with length; explain the first N words


@lru_cache(maxsize=1)
def get_explainer():
    tok, model = load()
    pipe = pipeline("text-classification", model=model, tokenizer=tok,
                    top_k=None, truncation=True, max_length=256)
    return shap.Explainer(pipe)


def explain(text, max_evals=400):
    """Returns list of (token, shap_value_toward_FAKE)."""
    text = " ".join(text.split()[:MAX_WORDS])
    sv = get_explainer()([text], max_evals=max_evals, batch_size=16, silent=True)
    names = list(sv.output_names)
    idx = names.index("FAKE") if "FAKE" in names else 0
    tokens, values = sv.data[0], sv.values[0][:, idx]
    return [(t, float(v)) for t, v in zip(tokens, values) if str(t).strip()]


def render_html(token_values):
    scale = max((abs(v) for _, v in token_values), default=1e-9) or 1e-9
    spans = []
    for tok, v in token_values:
        a = min(abs(v) / scale, 1.0) * 0.85
        rgb = "220,38,38" if v > 0 else "22,163,74"
        spans.append(
            f'<span title="SHAP {v:+.3f}" style="background:rgba({rgb},{a:.2f});'
            f'border-radius:3px;padding:1px 0">{html.escape(str(tok))}</span>')
    legend = ('<div style="margin-bottom:8px;font-size:13px">'
              '<span style="background:rgba(220,38,38,.6);padding:2px 6px;border-radius:3px">'
              'pushes toward FAKE</span> '
              '<span style="background:rgba(22,163,74,.6);padding:2px 6px;border-radius:3px">'
              'pushes toward REAL</span></div>')
    return (f'<div style="background:#fff;color:#111;padding:14px;border-radius:8px;'
            f'line-height:1.9;white-space:pre-wrap">{legend}{"".join(spans)}</div>')


def top_words(token_values, k=5):
    agg = {}
    for t, v in token_values:
        w = str(t).strip().lower()
        agg[w] = agg.get(w, 0.0) + v
    ranked = sorted(agg.items(), key=lambda x: x[1])
    fake = [w for w, v in reversed(ranked[-k:]) if v > 0]
    real = [w for w, v in ranked[:k] if v < 0]
    return fake, real


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="results/explanation.html")
    a = ap.parse_args()
    probs = predict_proba([a.text])[0]
    print(f"P(FAKE)={probs[0]:.3f}  P(REAL)={probs[1]:.3f}")
    tv = explain(a.text)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(render_html(tv), encoding="utf-8")
    print("Saved", a.out)

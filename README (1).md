# 📰 Fake News Detection with Explainable AI

Fine-tuned **DistilBERT** to classify news articles as real or fake, compared against a
**TF-IDF + Logistic Regression** baseline. Predictions are explained with **SHAP**
(word-level highlights) and served in a live **Gradio** demo.

🔗 **Live demo:** _add your Hugging Face Space link here_

## Results (test set)

_Run `python -m src.evaluate` and paste `results/comparison.md` here._

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + LogReg | – | – | – | – |
| DistilBERT | – | – | – | – |

## Project structure

```
├── app.py                  # Gradio demo
├── requirements.txt
├── src/
│   ├── data.py             # load, clean (leak removal), split
│   ├── baseline.py         # TF-IDF + LogReg
│   ├── train.py            # DistilBERT fine-tuning
│   ├── evaluate.py         # test metrics + comparison table
│   ├── predict.py          # inference helpers
│   ├── explain.py          # SHAP + HTML highlighting
│   └── utils.py
├── scripts/push_to_hub.py
├── data/raw/               # put Fake.csv and True.csv here
└── results/                # metrics, confusion matrices
```

## Quickstart

```bash
git clone https://github.com/<you>/fake-news-xai && cd fake-news-xai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Dataset:** download the
[Fake and Real News Dataset (ISOT)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
from Kaggle and place `Fake.csv` and `True.csv` in `data/raw/`.

**2. Baseline**
```bash
python -m src.baseline
```

**3. Train DistilBERT** (GPU strongly recommended; free Colab T4 takes ~15–25 min)
```bash
python -m src.train --epochs 2
# quick smoke test on CPU:
python -m src.train --sample 500 --epochs 1 --max_len 128
```

**4. Evaluate & compare**
```bash
python -m src.evaluate
```

**5. Explain one prediction (CLI)**
```bash
python -m src.explain --text "Your article text here ..."
```

**6. Run the demo**
```bash
python app.py
```

## Deploy the demo on Hugging Face Spaces

1. `huggingface-cli login`, then `python scripts/push_to_hub.py <username>/distilbert-fake-news`
2. Create a new **Gradio** Space and upload `app.py`, `requirements.txt` and the `src/` folder.
3. In the Space → Settings → Variables, add `MODEL_ID = <username>/distilbert-fake-news`.

## Notes & limitations

- **Dataset leakage:** ISOT real articles contain "(Reuters)" and fake ones contain
  source tags. `src/data.py` strips these and removes duplicates, so scores are lower
  (and more honest) than many published 99%+ numbers.
- **Not a fact-checker:** the model learns writing style and topics from 2016–2017
  US-politics news. It can fail on other domains/time periods.
- SHAP explanations cover the first 150 words of an article to keep latency reasonable.
- Both models score very high on ISOT, so report your actual numbers and, if possible,
  test on a second dataset (e.g. WELFake) to show generalisation.

## License
MIT

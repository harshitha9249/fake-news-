"""Gradio demo: classify a news article and show word-level SHAP highlights."""
import gradio as gr

from src.explain import explain, render_html, top_words
from src.predict import predict_proba

EXAMPLES = [
    ["WASHINGTON - The U.S. Senate on Tuesday passed a bipartisan spending bill "
     "that would keep the federal government funded through the end of the fiscal "
     "year, sending the measure to the president's desk after weeks of negotiations "
     "between Republican and Democratic leaders."],
    ["BREAKING: Shocking leaked video PROVES the media has been lying to you for "
     "years! Wake up people, they don't want you to know the truth about what "
     "happened last night. Share this before it gets deleted by the elites!"],
]


def analyze(text):
    if not text or len(text.split()) < 15:
        raise gr.Error("Please paste at least ~15 words of article text.")
    probs = predict_proba([text])[0]
    label = {"FAKE": float(probs[0]), "REAL": float(probs[1])}
    tv = explain(text)
    fake_w, real_w = top_words(tv)
    summary = (f"**Words pushing toward FAKE:** {', '.join(fake_w) or '-'}  \n"
               f"**Words pushing toward REAL:** {', '.join(real_w) or '-'}")
    return label, render_html(tv), summary


with gr.Blocks(title="Fake News Detector (Explainable)") as demo:
    gr.Markdown(
        "# 📰 Fake News Detection with Explainable AI\n"
        "DistilBERT classifier + SHAP word-level explanations. "
        "This model detects *writing patterns* learned from its training data; "
        "it is **not** a fact-checker.")
    inp = gr.Textbox(lines=8, label="News article text")
    btn = gr.Button("Analyze", variant="primary")
    with gr.Row():
        lbl = gr.Label(label="Prediction", num_top_classes=2)
        summ = gr.Markdown()
    out_html = gr.HTML(label="SHAP highlights")
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(analyze, inp, [lbl, out_html, summ])

if __name__ == "__main__":
    demo.launch()

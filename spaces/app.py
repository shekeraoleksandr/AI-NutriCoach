"""NutriCoach — Hugging Face Space (Gradio).

Self-contained deployment of the capstone: a RAG nutrition chatbot backed by a
domain-fine-tuned retriever, plus a tools tab that calls the Phase-1 models
(calorie regressor, Nutri-Score classifier, Conditional VAE).

All trained artifacts are downloaded from the Hugging Face Hub at startup, so this
file has no dependency on the training repo. Set the model repo id via the
NUTRICOACH_REPO env var (Space → Settings → Variables), e.g. "yourname/nutricoach".
"""
from __future__ import annotations

import json
import os
import re

import gradio as gr
import joblib
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# --- Config ------------------------------------------------------------------
HF_REPO = os.environ.get("NUTRICOACH_REPO", "YOUR_HF_USERNAME/nutricoach")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
NUTRIENT_FEATURES = [
    "proteins_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "salt_100g", "sodium_100g",
]
GRADES = ["a", "b", "c", "d", "e"]
TOP_K = 5
CHUNK_SIZE, CHUNK_OVERLAP = 512, 64

WIKI_TITLES = [
    "Nutrition", "Protein (nutrient)", "Carbohydrate", "Dietary fiber", "Fat",
    "Saturated fat", "Sugar", "Vitamin", "Mineral (nutrient)", "Calorie",
    "Dietary supplement", "Creatine", "Whey protein", "Glycemic index",
    "Nutri-Score", "Healthy diet", "Exercise physiology", "Muscle hypertrophy",
    "Basal metabolic rate", "Micronutrient",
]


# --- Conditional VAE (vendored so the Space needs no training package) --------
class CVAE(nn.Module):
    def __init__(self, n_features=8, n_cond=5, latent_dim=8, hidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Linear(n_features + n_cond, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim + n_cond, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_features))

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=-1))


# --- Load everything from the Hub --------------------------------------------
print("Downloading artifacts from", HF_REPO)
LOCAL = snapshot_download(HF_REPO, repo_type="model")

retriever = SentenceTransformer(os.path.join(LOCAL, "retriever_finetuned"))
regressor = joblib.load(os.path.join(LOCAL, "regressor.joblib"))
classifier = joblib.load(os.path.join(LOCAL, "classifier.joblib"))

cvae = CVAE()
cvae.load_state_dict(torch.load(os.path.join(LOCAL, "cvae.pt"), map_location="cpu"))
cvae.eval()
cvae_scaler = joblib.load(os.path.join(LOCAL, "cvae_scaler.joblib"))

# Optional: nutrient database for the "most similar real product" lookup (nearest neighbour).
try:
    _npz = np.load(os.path.join(LOCAL, "foods.npz"), allow_pickle=True)
    FOOD_X = cvae_scaler.transform(_npz["X"].astype("float32"))   # standardised nutrients
    FOOD_NAMES = _npz["names"]
    print("loaded", len(FOOD_NAMES), "foods for nearest-neighbour lookup")
except Exception as e:  # foods.npz not uploaded yet -> feature simply disabled
    FOOD_X, FOOD_NAMES = None, None
    print("foods.npz not found — nearest-food lookup disabled:", e)


# --- Corpus: load prebuilt chunks if present, else fetch Wikipedia -----------
def _chunk(text, source):
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append({"text": text[i:i + CHUNK_SIZE], "meta": {"source": source}})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return out


def build_corpus():
    prebuilt = os.path.join(LOCAL, "corpus.jsonl")
    if os.path.exists(prebuilt):
        chunks = [json.loads(l) for l in open(prebuilt)]
        print("loaded", len(chunks), "chunks from Hub")
        return chunks
    print("corpus.jsonl not found — fetching Wikipedia (slower cold start)")
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia(user_agent="NutriCoach-space/0.1", language="en")
    chunks = []
    for t in WIKI_TITLES:
        page = wiki.page(t)
        if page.exists() and len(page.text) > 200:
            chunks += _chunk(page.text, f"Wikipedia: {t}")
    print("built", len(chunks), "chunks")
    return chunks


CHUNKS = build_corpus()
CORPUS_EMB = retriever.encode([c["text"] for c in CHUNKS], normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")


def search(query, k=TOP_K):
    q = retriever.encode([query], normalize_embeddings=True).astype("float32")
    scores = (CORPUS_EMB @ q[0])
    idx = np.argsort(-scores)[:k]
    return [{"text": CHUNKS[i]["text"], "source": CHUNKS[i]["meta"]["source"],
             "score": float(scores[i])} for i in idx]


# --- Generator LLM -----------------------------------------------------------
from transformers import pipeline

llm = pipeline("text-generation", model=GENERATOR_MODEL,
               torch_dtype="auto", device_map="auto")


def generate(prompt, max_new_tokens=256):
    out = llm([{"role": "user", "content": prompt}],
              max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"]


ANSWER_PROMPT = (
    "You are NutriCoach, a careful nutrition assistant. Answer the question USING ONLY the "
    "context passages below. Cite sources as [1], [2] in order. If the context is insufficient, "
    "say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


# --- Tools (Phase-1 models) --------------------------------------------------
def _vec(features):
    return np.array([[float(features.get(f, 0.0)) for f in NUTRIENT_FEATURES]])


def predict_calories(features):
    return float(regressor.predict(_vec(features))[0])


def predict_grade(features):
    pred = classifier.predict(_vec(features))[0]
    try:
        return GRADES[int(pred)]
    except (ValueError, TypeError, IndexError):
        return str(pred)


def generate_profile(target_grade="a", n=1):
    c = np.zeros((n, len(GRADES)), dtype="float32")
    c[:, GRADES.index(target_grade.lower())] = 1.0
    z = torch.randn(n, cvae.latent_dim)
    with torch.no_grad():
        x = cvae.decode(z, torch.tensor(c)).numpy()
    x = np.clip(cvae_scaler.inverse_transform(x), 0, None)
    return [dict(zip(NUTRIENT_FEATURES, row.tolist())) for row in x]


def nearest_food(features, k=3):
    """Most chemically similar real products (nearest neighbour over the OFF dataset)."""
    if FOOD_X is None:
        return []
    q = cvae_scaler.transform(_vec(features))
    dist = np.linalg.norm(FOOD_X - q, axis=1)
    out = []
    for i in np.argsort(dist):
        name = str(FOOD_NAMES[i]).strip()
        if name and name.lower() not in ("nan", "none"):
            out.append(name)
        if len(out) >= k:
            break
    return out


# --- Chat handler ------------------------------------------------------------
def chat(message, history):
    hits = search(message, k=TOP_K)
    context = "\n\n".join(f"[{i+1}] {h['text']}  (source: {h['source']})"
                          for i, h in enumerate(hits))
    answer = generate(ANSWER_PROMPT.format(context=context, question=message))
    seen, srcs = set(), []
    for h in hits:
        if h["source"] not in seen:
            seen.add(h["source"]); srcs.append(h["source"])
    return answer + "\n\n---\n**Sources:** " + ", ".join(srcs)


# --- Tools tab handler -------------------------------------------------------
def analyze(proteins, fat, sat_fat, carbs, sugars, fiber, salt, sodium, target_grade):
    feats = dict(zip(NUTRIENT_FEATURES,
                     [proteins, fat, sat_fat, carbs, sugars, fiber, salt, sodium]))
    kcal = predict_calories(feats)
    grade = predict_grade(feats).upper()
    match = nearest_food(feats, k=3)
    food_line = f"~{kcal:.0f} kcal/100g · Nutri-Score {grade}"
    if match:
        food_line += "\nMost similar products: " + "; ".join(match)

    alt = generate_profile(target_grade, n=1)[0]
    alt_kcal = predict_calories(alt)
    alt_grade = predict_grade(alt).upper()
    alt_match = nearest_food(alt, k=3)
    alt_line = f"~{alt_kcal:.0f} kcal/100g · Nutri-Score {alt_grade}"
    if alt_match:
        alt_line += "\nMost similar products: " + "; ".join(alt_match)
    alt_str = ", ".join(f"{k.replace('_100g','')}={v:.1f}" for k, v in alt.items())
    return (food_line, alt_line, alt_str)


# --- UI ----------------------------------------------------------------------
with gr.Blocks(title="NutriCoach") as demo:
    gr.Markdown("# 🥗 NutriCoach — nutrition assistant\n"
                "RAG chatbot with a domain-fine-tuned retriever + trained nutrition models.")

    with gr.Tab("Chat (RAG)"):
        gr.ChatInterface(
            chat,
            cache_examples=False,   # don't run the LLM at startup
            examples=[
                "How much protein should I eat to build muscle, and does creatine help?",
                "What is the glycemic index and why does it matter?",
                "Why is dietary fiber good for you?",
            ],
        )

    with gr.Tab("Nutrition tools"):
        gr.Markdown("Enter a food's nutrients (per 100 g). The models predict its calories and "
                    "Nutri-Score, then the CVAE generates a healthier alternative for a target grade.")
        with gr.Row():
            inputs = [gr.Number(label=f.replace("_100g", ""), value=0)
                      for f in NUTRIENT_FEATURES]
        target = gr.Dropdown(GRADES, value="a", label="Target grade for the alternative")
        btn = gr.Button("Analyze", variant="primary")
        out_food = gr.Textbox(label="Your food")
        out_alt = gr.Textbox(label="Healthier alternative (CVAE)")
        out_prof = gr.Textbox(label="Alternative profile")
        btn.click(analyze, inputs + [target], [out_food, out_alt, out_prof])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

---
title: NutriCoach
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.11"
pinned: false
---

# NutriCoach — nutrition assistant

A RAG chatbot with a domain-fine-tuned retriever, plus trained nutrition models
(calorie regression, Nutri-Score classification, Conditional VAE). Artifacts are
loaded from the Hugging Face Hub at startup.

**Setup:** set the Space variable `NUTRICOACH_REPO` to your model repo id
(e.g. `yourname/nutricoach`). Optionally set `GENERATOR_MODEL`
(default `Qwen/Qwen2.5-1.5B-Instruct`; use `Qwen/Qwen2.5-0.5B-Instruct` for faster CPU).

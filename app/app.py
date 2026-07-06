"""Gradio demo — three tabs, mirroring the reference project's single-interface style.

  Tab 1  Tabular tools   — calorie regression, Nutri-Score grade, CVAE profile generation
  Tab 2  Nutrition chat  — RAG answer with cited sources (fine-tuned retriever + LLM)
  Tab 3  Retriever A/B    — same query through baseline vs fine-tuned retriever, side by side

Run locally:   python app/app.py
Deploy:        Hugging Face Spaces (SDK: gradio)
Models are loaded lazily so the app starts even before every artifact is trained.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from nutricoach import config as C


def _tabular_predict(proteins, fat, sat_fat, carbs, sugars, fiber, salt, sodium, grade):
    from nutricoach.models_tabular import predict_calories, predict_grade
    from nutricoach.cvae import generate_profile

    features = {
        "proteins_100g": proteins, "fat_100g": fat, "saturated_fat_100g": sat_fat,
        "carbohydrates_100g": carbs, "sugars_100g": sugars, "fiber_100g": fiber,
        "salt_100g": salt, "sodium_100g": sodium,
    }
    kcal = predict_calories(features)
    pred_grade = predict_grade(features)
    generated = generate_profile(target_grade=grade, n=1)[0]
    gen_str = ", ".join(f"{k}={v:.1f}" for k, v in generated.items())
    return f"~{kcal:.0f} kcal/100g", pred_grade.upper(), gen_str


def build_ui():
    with gr.Blocks(title="NutriCoach") as demo:
        gr.Markdown("# NutriCoach — nutrition assistant (DL capstone)")

        with gr.Tab("Tabular tools (Phase 1)"):
            inputs = [gr.Number(label=f) for f in C.NUTRIENT_FEATURES]
            grade = gr.Dropdown(C.NUTRISCORE_GRADES, value="a", label="CVAE target grade")
            btn = gr.Button("Predict & generate")
            out = [gr.Textbox(label="Calories"), gr.Textbox(label="Nutri-Score"),
                   gr.Textbox(label="Generated profile")]
            btn.click(_tabular_predict, inputs=inputs + [grade], outputs=out)

        with gr.Tab("Nutrition chat (Phase 2)"):
            gr.Markdown("_Wire up RagPipeline once the retriever + generator are trained._")
            q = gr.Textbox(label="Ask a nutrition question")
            a = gr.Textbox(label="Answer (with sources)")
            # TODO: q.submit(rag_pipeline.answer, ...)

        with gr.Tab("Retriever A/B (results)"):
            gr.Markdown("_Baseline vs fine-tuned retriever for the same query._")
            # TODO: show top-k passages from both retrievers side by side
    return demo


if __name__ == "__main__":
    build_ui().launch()

"""RAG orchestration (Phase 2): retrieve -> (optionally call Phase-1 tools) -> generate.

This is where the two phases connect. The assistant answers general nutrition questions from
the retrieved corpus, and for quantitative questions about a specific food it CALLS the
Phase-1 models (calorie regressor, Nutri-Score classifier, Conditional VAE) as tools.
"""
from __future__ import annotations

from . import config as C
from .retriever import DenseRetriever

# Phase-1 models exposed as tools -------------------------------------------------
from .models_tabular import predict_calories, predict_grade
from .cvae import generate_profile

TOOLS = {
    "predict_calories": predict_calories,   # (features: dict) -> float
    "predict_grade": predict_grade,         # (features: dict) -> str ('a'..'e')
    "generate_profile": generate_profile,   # (target_grade: str, n: int) -> list[dict]
}

ANSWER_PROMPT = (
    "You are NutriCoach, a careful nutrition assistant. Answer the question USING ONLY the "
    "context passages below. Cite sources as [1], [2] matching the passage order. If the "
    "context is insufficient, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


class RagPipeline:
    def __init__(self, retriever: DenseRetriever, generate_fn):
        """generate_fn: callable(prompt:str)->str wrapping the generator LLM."""
        self.retriever = retriever
        self.generate_fn = generate_fn

    def _format_context(self, hits: list[dict]) -> str:
        return "\n\n".join(
            f"[{i+1}] {h['text']}  (source: {h['meta'].get('source', 'n/a')})"
            for i, h in enumerate(hits)
        )

    def answer(self, question: str, k: int = C.TOP_K) -> dict:
        hits = self.retriever.search(question, k=k)
        prompt = ANSWER_PROMPT.format(
            context=self._format_context(hits), question=question
        )
        text = self.generate_fn(prompt)
        return {"answer": text, "sources": hits}

    # --- tool call demonstration for the report / demo ---
    def tool_answer(self, tool_name: str, **kwargs):
        """Directly invoke a Phase-1 model as a tool (used by the Gradio 'tools' tab)."""
        if tool_name not in TOOLS:
            raise KeyError(f"unknown tool {tool_name!r}; available: {list(TOOLS)}")
        return TOOLS[tool_name](**kwargs)

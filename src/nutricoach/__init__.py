"""NutriCoach — two-phase Deep Learning capstone.

Phase 1 (tabular):  calorie regression, Nutri-Score classification, Conditional VAE generation.
Phase 2 (RAG):      a nutrition assistant that retrieves from a knowledge corpus with a
                    domain-fine-tuned retriever AND calls the Phase-1 models as tools.

The public interface is intentionally small so the RAG layer (Phase 2) can import the
Phase-1 models as plain callables without touching notebook internals.
"""

__version__ = "0.1.0"

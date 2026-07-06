"""Corpus ingestion for Phase 2: fetch nutrition text, clean, and chunk.

Suggested OPEN sources (document exactly what you use in the README):
  - Wikipedia articles in nutrition / dietary-supplement / exercise-physiology categories
  - OpenStax 'Nutrition' textbook (CC BY)
  - WHO / USDA dietary guidelines (public domain)
Keep the corpus modest (a few hundred to a few thousand chunks) — plenty for a clear
baseline-vs-finetuned signal and fast to index on Colab.
"""
from __future__ import annotations

import re

from . import config as C


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)          # drop wiki-style [12] refs
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, source: str = "", size: int = C.CHUNK_SIZE,
               overlap: int = C.CHUNK_OVERLAP) -> list[dict]:
    """Sliding-window character chunks. Returns [{'text':..., 'meta': {...}}]."""
    text = clean_text(text)
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size]
        chunks.append({"text": piece, "meta": {"source": source}})
        start += size - overlap
    return chunks


def build_corpus(documents: list[dict]) -> list[dict]:
    """documents: [{'text':..., 'source':...}] -> flat list of chunks ready to index."""
    out = []
    for doc in documents:
        out.extend(chunk_text(doc["text"], source=doc.get("source", "")))
    return out

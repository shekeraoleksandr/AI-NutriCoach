"""Synthetic query generation for retriever fine-tuning (Phase 2).

For each corpus chunk we ask an LLM to write plausible user questions it answers. This gives
(query, positive_passage) training pairs with NO manual labelling — the same idea as GPL /
InPars. Split off a held-out slice as the retrieval test set (gold query -> passage).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config as C

QUESTION_PROMPT = (
    "You are creating training data for a nutrition search engine.\n"
    "Given the passage below, write {n} short, natural questions that a person could ask "
    "and that this passage directly answers. Vary the phrasing. Return one question per line.\n\n"
    "Passage:\n{passage}\n\nQuestions:"
)


def generate_pairs(chunks: list[dict], generate_fn, n_per_chunk: int = 2,
                   out_path: Path | None = None) -> Path:
    """Create (query, positive) pairs.

    chunks: list of {"text": ..., "meta": {...}} (from ingest).
    generate_fn: callable(prompt: str) -> str  (wrap any LLM: local transformers or an API).
    """
    out_path = Path(out_path or C.PAIRS_DIR / "all_pairs.jsonl")
    with open(out_path, "w") as f:
        for ch in chunks:
            prompt = QUESTION_PROMPT.format(n=n_per_chunk, passage=ch["text"])
            raw = generate_fn(prompt)
            questions = [q.strip(" -0123456789.") for q in raw.splitlines() if q.strip()]
            for q in questions[:n_per_chunk]:
                if len(q) > 8:
                    f.write(json.dumps({
                        "query": q, "positive": ch["text"], "meta": ch.get("meta", {}),
                    }) + "\n")
    return out_path


def split_pairs(all_path: Path | None = None, test_frac: float = 0.15,
                seed: int = C.RANDOM_SEED):
    """Split into train (for fine-tuning) and test (gold query->passage for eval)."""
    all_path = Path(all_path or C.PAIRS_DIR / "all_pairs.jsonl")
    rows = [json.loads(l) for l in open(all_path)]
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * (1 - test_frac))
    train, test = rows[:cut], rows[cut:]
    for name, data in [("train_pairs.jsonl", train), ("test_pairs.jsonl", test)]:
        with open(C.PAIRS_DIR / name, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    return len(train), len(test)

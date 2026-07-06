"""Dense retriever (Phase 2): build index, search, and fine-tune on domain pairs.

The headline result of the project lives here: baseline embeddings vs. a domain-fine-tuned
retriever, compared with Recall@k / MRR / nDCG in evaluate.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C


class DenseRetriever:
    """Thin wrapper around a sentence-transformers model + a FAISS index."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or C.BASE_EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.passages: list[str] = []
        self.meta: list[dict] = []       # e.g. {"source": ..., "url": ...} per passage

    # --- indexing ---
    def build_index(self, passages: list[str], meta: list[dict] | None = None,
                    batch_size: int = 64):
        import faiss

        self.passages = passages
        self.meta = meta or [{} for _ in passages]
        emb = self.model.encode(
            passages, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])   # cosine (vectors normalized)
        self.index.add(emb)
        return self

    def save_index(self, path: Path | None = None):
        import faiss

        path = Path(path or C.PROCESSED_DIR / "corpus.faiss")
        faiss.write_index(self.index, str(path))
        with open(path.with_suffix(".jsonl"), "w") as f:
            for p, m in zip(self.passages, self.meta):
                f.write(json.dumps({"text": p, "meta": m}) + "\n")

    def load_index(self, path: Path | None = None):
        import faiss

        path = Path(path or C.PROCESSED_DIR / "corpus.faiss")
        self.index = faiss.read_index(str(path))
        self.passages, self.meta = [], []
        with open(path.with_suffix(".jsonl")) as f:
            for line in f:
                row = json.loads(line)
                self.passages.append(row["text"])
                self.meta.append(row["meta"])
        return self

    # --- search ---
    def search(self, query: str, k: int = C.TOP_K) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return [
            {"text": self.passages[i], "meta": self.meta[i], "score": float(s)}
            for s, i in zip(scores[0], idx[0]) if i != -1
        ]


def finetune(pairs_path: Path | None = None, base_model: str | None = None,
             out_dir: Path | None = None, epochs: int = 1, batch_size: int = 32):
    """Fine-tune the embedding model on (query, positive_passage) pairs.

    Uses MultipleNegativesRankingLoss (in-batch negatives) — the standard, data-efficient
    recipe for sentence-transformers retrieval fine-tuning.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    pairs_path = Path(pairs_path or C.PAIRS_DIR / "train_pairs.jsonl")
    base_model = base_model or C.BASE_EMBEDDING_MODEL
    out_dir = Path(out_dir or C.FINETUNED_MODEL_DIR)

    examples = []
    with open(pairs_path) as f:
        for line in f:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["query"], row["positive"]]))

    model = SentenceTransformer(base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * epochs * 0.1)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=warmup, show_progress_bar=True)
    model.save(str(out_dir))
    return out_dir

"""Evaluation utilities.

Phase 1: regression + classification metrics.
Phase 2: retrieval metrics (Recall@k, MRR, nDCG) — the headline baseline-vs-finetuned table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C


# --- Phase 1 -----------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


# --- Phase 2: retrieval ------------------------------------------------------
def recall_at_k(ranked_ids: list, gold_id, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list, gold_id) -> float:
    for rank, i in enumerate(ranked_ids, start=1):
        if i == gold_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list, gold_id, k: int) -> float:
    for rank, i in enumerate(ranked_ids[:k], start=1):
        if i == gold_id:
            return 1.0 / np.log2(rank + 1)
    return 0.0


def evaluate_retriever(retriever, test_pairs_path: Path | None = None,
                       ks=(1, 3, 5, 10)) -> dict:
    """Score a retriever on gold (query -> passage) pairs.

    Passages are matched by exact text identity against the indexed corpus, so make sure the
    test-pair 'positive' strings are the same chunks that were indexed.
    """
    test_pairs_path = Path(test_pairs_path or C.PAIRS_DIR / "test_pairs.jsonl")
    text_to_id = {t: i for i, t in enumerate(retriever.passages)}

    maxk = max(ks)
    agg = {f"recall@{k}": [] for k in ks}
    agg["mrr"], agg["ndcg@10"] = [], []
    for row in map(json.loads, open(test_pairs_path)):
        gold = text_to_id.get(row["positive"])
        if gold is None:
            continue
        hits = retriever.search(row["query"], k=maxk)
        ranked = [text_to_id.get(h["text"]) for h in hits]
        for k in ks:
            agg[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        agg["mrr"].append(reciprocal_rank(ranked, gold))
        agg["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in agg.items()}


def compare_retrievers(baseline, finetuned, test_pairs_path: Path | None = None) -> dict:
    """Return {'baseline': {...}, 'finetuned': {...}} — the headline results table."""
    return {
        "baseline": evaluate_retriever(baseline, test_pairs_path),
        "finetuned": evaluate_retriever(finetuned, test_pairs_path),
    }

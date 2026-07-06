"""Phase 1 tabular models: calorie regression + Nutri-Score classification.

Exposes small, stable callables so the RAG layer (Phase 2) can use them as *tools*:
    predict_calories(features_dict) -> float
    predict_grade(features_dict)    -> str

Under the hood we compare Random Forest / XGBoost / MLP (mirrors the reference project),
persist the best model, and reload it lazily.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from . import config as C

# --- Registry of candidate models (built lazily to avoid heavy imports at import time) ---


def build_regressors():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    return {
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=C.RANDOM_SEED, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.9, random_state=C.RANDOM_SEED,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=C.RANDOM_SEED
        ),
    }


def build_classifiers(class_weight=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=C.RANDOM_SEED,
            n_jobs=-1, class_weight=class_weight,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.9, random_state=C.RANDOM_SEED,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=C.RANDOM_SEED
        ),
    }


# --- Persistence -------------------------------------------------------------
REG_PATH = C.MODELS_DIR / "regressor.joblib"
CLF_PATH = C.MODELS_DIR / "classifier.joblib"


def save_model(model, path: Path) -> None:
    joblib.dump(model, path)


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found — train it first (notebooks/02_train_tabular.ipynb)"
        )
    return joblib.load(path)


# --- Tool interface used by the RAG layer (Phase 2) --------------------------
_reg_cache = None
_clf_cache = None


def _vectorize(features: dict) -> np.ndarray:
    """Turn a {feature: value} dict into the model's input row (config order)."""
    return np.array([[float(features.get(f, 0.0)) for f in C.NUTRIENT_FEATURES]])


def predict_calories(features: dict) -> float:
    """kcal / 100 g for a described food. Stable tool signature for the RAG agent."""
    global _reg_cache
    if _reg_cache is None:
        _reg_cache = _load(REG_PATH)
    return float(_reg_cache.predict(_vectorize(features))[0])


def predict_grade(features: dict) -> str:
    """Predicted Nutri-Score grade (a..e). Stable tool signature for the RAG agent."""
    global _clf_cache
    if _clf_cache is None:
        _clf_cache = _load(CLF_PATH)
    return str(_clf_cache.predict(_vectorize(features))[0])

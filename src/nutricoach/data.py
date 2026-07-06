"""Data loading, cleaning and splitting for Phase 1 (tabular).

Dataset: Open Food Facts (with Nutriscore & Generic Names) on Kaggle
         slug: paufortiana/open-food-facts-with-nutriscore-and-generic-names

The loader maps Open Food Facts' raw column names (which mix hyphens/underscores) to the
canonical schema in config.py, so downstream code and the RAG tools stay clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


# --- download ----------------------------------------------------------------
def download_raw() -> str:
    """Download the Kaggle dataset into data/raw/ and return the CSV path.

    Requires Kaggle credentials (kagglehub reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME /
    KAGGLE_KEY env vars). On Colab you can also upload kaggle.json.
    Falls back to a clear message if you'd rather download the CSV manually.
    """
    try:
        import kagglehub

        path = kagglehub.dataset_download(C.KAGGLE_DATASET)
        # kagglehub returns a directory; find the (first) csv inside it
        from pathlib import Path

        csvs = sorted(Path(path).rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"no CSV found in downloaded dataset at {path}")
        return str(csvs[0])
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Automatic download failed. Either set Kaggle credentials, or download the CSV "
            f"manually from https://www.kaggle.com/datasets/{C.KAGGLE_DATASET} "
            f"and place it in {C.RAW_DIR}. Original error: {e}"
        )


# --- load & clean ------------------------------------------------------------
def _read_any(path) -> pd.DataFrame:
    """OFF exports are sometimes tab-separated; sniff the delimiter."""
    for sep in (",", "\t", ";"):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {canonical_name: raw_name_found_in_df} using the config candidate lists."""
    resolved = {}
    lower = {c.lower(): c for c in df.columns}
    for canonical, candidates in C.RAW_COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand.lower() in lower:
                resolved[canonical] = lower[cand.lower()]
                break
    return resolved


def load_clean(path: str | None = None) -> pd.DataFrame:
    """Load Open Food Facts data and return a cleaned frame in the canonical schema.

    Cleaning: rename -> numeric coercion -> drop rows missing the regression target ->
    clip to plausible bounds -> drop physically impossible rows -> dedupe.
    """
    src = path or _find_local_csv()
    raw = _read_any(src)

    resolved = _resolve_columns(raw)
    missing = [c for c in [C.REGRESSION_TARGET] if c not in resolved]
    if missing:
        raise KeyError(
            f"Could not find column(s) {missing} in the dataset. Columns present: "
            f"{list(raw.columns)[:40]} ... Update config.RAW_COLUMN_CANDIDATES if names differ."
        )

    df = raw[list(resolved.values())].rename(columns={v: k for k, v in resolved.items()})

    # numeric coercion for nutrient + target columns
    num_cols = [c for c in C.NUTRIENT_FEATURES + [C.REGRESSION_TARGET] if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalise grade text
    if C.CLASSIFICATION_TARGET in df.columns:
        df[C.CLASSIFICATION_TARGET] = (
            df[C.CLASSIFICATION_TARGET].astype(str).str.strip().str.lower()
        )
        df.loc[~df[C.CLASSIFICATION_TARGET].isin(C.NUTRISCORE_GRADES),
               C.CLASSIFICATION_TARGET] = np.nan

    # must have the regression target
    df = df.dropna(subset=[C.REGRESSION_TARGET])

    # clip to plausible bounds, then drop rows still out of range on the target
    for c, (lo, hi) in C.VALUE_BOUNDS.items():
        if c in df.columns:
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    df = df.dropna(subset=[C.REGRESSION_TARGET])

    # fill remaining nutrient NaNs with 0 (missing on OFF usually means "not declared")
    for c in C.NUTRIENT_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = 0.0  # guarantee the column exists for the model input schema

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _find_local_csv():
    """Pick a (non-empty) CSV already sitting in data/raw/ (manual download path)."""
    csvs = sorted(p for p in C.RAW_DIR.glob("*.csv") if p.stat().st_size > 0)
    if not csvs:
        raise FileNotFoundError(
            f"No non-empty CSV in {C.RAW_DIR}. Run data.download_raw() or drop the Kaggle CSV there."
        )
    return csvs[0]


# --- splits ------------------------------------------------------------------
def split_regression(df: pd.DataFrame, test_size: float = 0.2):
    """X_train, X_test, y_train, y_test for calorie regression."""
    X = df[C.NUTRIENT_FEATURES].astype(float).values
    y = df[C.REGRESSION_TARGET].astype(float).values
    return train_test_split(X, y, test_size=test_size, random_state=C.RANDOM_SEED)


def split_classification(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified split for Nutri-Score classification (rows with a grade only)."""
    df = df.dropna(subset=[C.CLASSIFICATION_TARGET])
    X = df[C.NUTRIENT_FEATURES].astype(float).values
    y = df[C.CLASSIFICATION_TARGET].astype(str).values
    return train_test_split(
        X, y, test_size=test_size, random_state=C.RANDOM_SEED, stratify=y
    )


def premium_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a Nutri-Score grade — used to train the Conditional VAE."""
    return df.dropna(subset=[C.CLASSIFICATION_TARGET]).reset_index(drop=True)

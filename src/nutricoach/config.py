"""Central configuration: paths, feature schema, constants.

Keeping these in one place means notebooks, src modules and the Gradio app all agree on
column names, the Nutri-Score grades, and where artifacts live.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]        # repo root: nutricoach-rag/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PAIRS_DIR = DATA_DIR / "pairs"                     # synthetic query-passage pairs (Phase 2)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, PAIRS_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Tabular feature schema (Phase 1) ----------------------------------------
# Physicochemical / nutrient columns per 100 g. Adjust to the exact dataset you download
# (Open Food Facts / USDA FoodData Central). Keep names lowercase + snake_case.
NUTRIENT_FEATURES = [
    "proteins_100g",
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "salt_100g",
    "sodium_100g",
]

# Regression target (Task 1)
REGRESSION_TARGET = "energy_kcal_100g"

# Classification target (Task 2): Nutri-Score grade A (healthiest) .. E (least healthy)
NUTRISCORE_GRADES = ["a", "b", "c", "d", "e"]
CLASSIFICATION_TARGET = "nutriscore_grade"

# Human-readable product name columns (used to make the RAG demo tangible)
NAME_COLUMNS = ["product_name", "generic_name"]

# --- Raw -> canonical column mapping -----------------------------------------
# Open Food Facts uses a mix of hyphens/underscores and sometimes energy in kJ.
# `data.load_clean()` renames the FIRST raw name it finds to our canonical name.
# Candidate lists cover the common OFF export variants so the loader is dataset-robust.
RAW_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "energy_kcal_100g": ["energy-kcal_100g", "energy_kcal_100g", "energy_100g", "energy-kcal_value"],
    "proteins_100g":    ["proteins_100g"],
    "fat_100g":         ["fat_100g"],
    "saturated_fat_100g": ["saturated-fat_100g", "saturated_fat_100g"],
    "carbohydrates_100g": ["carbohydrates_100g"],
    "sugars_100g":      ["sugars_100g"],
    "fiber_100g":       ["fiber_100g", "fibre_100g"],
    "salt_100g":        ["salt_100g"],
    "sodium_100g":      ["sodium_100g"],
    "nutriscore_grade": ["nutriscore_grade", "nutrition_grade_fr", "nutrition-grade_fr"],
    "product_name":     ["product_name"],
    "generic_name":     ["generic_name"],
}

# Plausible-value bounds (per 100 g) for sanity-cleaning OFF's noisy entries.
VALUE_BOUNDS = {
    "energy_kcal_100g": (0, 900),   # >900 kcal/100g is impossible (pure fat ~900)
    "proteins_100g": (0, 100),
    "fat_100g": (0, 100),
    "saturated_fat_100g": (0, 100),
    "carbohydrates_100g": (0, 100),
    "sugars_100g": (0, 100),
    "fiber_100g": (0, 100),
    "salt_100g": (0, 100),
    "sodium_100g": (0, 40),
}

# Kaggle dataset slug (used by kagglehub in notebook 01)
KAGGLE_DATASET = "paufortiana/open-food-facts-with-nutriscore-and-generic-names"

RANDOM_SEED = 42

# --- RAG / retriever (Phase 2) -----------------------------------------------
BASE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"    # baseline retriever
FINETUNED_MODEL_DIR = MODELS_DIR / "retriever_finetuned"
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"     # small instruct LLM for answers
CHUNK_SIZE = 512           # characters per chunk (tune)
CHUNK_OVERLAP = 64
TOP_K = 5                  # retrieved passages passed to the generator

# Hugging Face Hub repo id to push the fine-tuned retriever (fill in your username)
HF_RETRIEVER_REPO = "your-username/nutricoach-retriever"

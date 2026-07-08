### NutriCoach — Nutrition Intelligence & RAG Assistant

**Live demo:** [oshek/nutricoach on Hugging Face Spaces](https://huggingface.co/spaces/oshek/nutricoach)

A deep-learning project that predicts a food's calorie content, grades its nutritional quality (Nutri-Score), generates novel nutrient profiles with a Conditional Variational Autoencoder, and answers nutrition questions through a Retrieval-Augmented Generation (RAG) assistant built on a **domain-fine-tuned retriever**. Built as a capstone project for a Deep Learning course.

---

### Project Overview

The project is split into two phases. **Phase 1** is a self-contained tabular project; **Phase 2** wraps a RAG assistant around it and reuses the Phase-1 models as callable tools.

**Phase 1 — Tabular nutrition analysis**

**Calorie Prediction** — given 8 physicochemical properties (per 100 g) of a food, predict its energy content in kcal/100 g. This is a regression problem.

**Nutri-Score Grading** — classify a food into one of five Nutri-Score grades (A = healthiest … E = least healthy) from the same nutrients. A 5-class problem with class imbalance.

**Nutrient Profile Generation** — a Conditional Variational Autoencoder (CVAE) trained across all grades generates new, chemically valid nutrient profiles **conditioned on a target Nutri-Score grade**. This is the generative component, connecting to the VAE/Diffusion lecture material, and improves on a plain VAE by adding conditioning.

**Phase 2 — RAG nutrition assistant**

**Retriever Fine-Tuning** — a general-purpose sentence encoder is adapted to the nutrition domain on synthetic question–passage pairs, and measured against the baseline with retrieval metrics. This is the project's headline result.

**RAG Assistant with Tool Use** — the assistant answers natural-language nutrition questions from a Wikipedia knowledge corpus (with citations) using the fine-tuned retriever, and for quantitative questions it **calls the Phase-1 models as tools** (calorie regressor, Nutri-Score classifier, CVAE generator).

All components are accessible through a single Gradio web interface with three tabs.

---

### Dataset

Source: [Open Food Facts (with Nutriscore & Generic Names)](https://www.kaggle.com/datasets/paufortiana/open-food-facts-with-nutriscore-and-generic-names) — a cleaned Kaggle subset of the Open Food Facts database.

After cleaning (column normalisation, numeric coercion, dropping impossible values such as >900 kcal/100 g, and filling undeclared nutrients with 0), the dataset contains **87,164 food products**, each described by nutrients per 100 g plus a human-relevant Nutri-Score grade.

**Input features (8 nutrients, per 100 g):**

| Feature | Description |
|---------|-------------|
| proteins_100g | Protein content |
| fat_100g | Total fat |
| saturated_fat_100g | Saturated fat |
| carbohydrates_100g | Total carbohydrates |
| sugars_100g | Sugars (subset of carbohydrates) |
| fiber_100g | Dietary fiber |
| salt_100g | Salt content |
| sodium_100g | Sodium content |

The dataset also carries `product_name` / `generic_name`, used to make the RAG demo tangible.

**Targets:** `energy_kcal_100g` (regression) and `nutriscore_grade` A–E (classification).

**Class distribution (Nutri-Score grade):**

| Grade | Samples |
|-------|---------|
| A (healthiest) | 15,381 |
| B | 13,211 |
| C | 18,636 |
| D | 24,057 |
| E (least healthy) | 15,879 |

The classes are imbalanced (D is nearly 2× larger than B), which is handled with class-weighted training.

---

### Models and Results

### Task 1 — Regression: predict calories (kcal/100 g)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Random Forest | 29.64 | 8.34 | 0.974 |
| **XGBoost** | 27.74 | 8.87 | **0.978** |
| MLP Neural Network | 28.04 | 8.44 | 0.977 |

**Best model: XGBoost** with RMSE ≈ 27.7 kcal and R² ≈ 0.978.

**How to read these numbers:** an R² of 0.978 looks spectacular, but this task is close to a *closed-form relationship*. Calories per 100 g are governed by the **Atwater factors**: `energy ≈ 4·protein + 4·carbohydrate + 9·fat`, and those macronutrients are exactly the input features. So the model is largely rediscovering a known formula. This task is therefore best read as a **sanity check** that the features are clean and informative — not as the project's achievement. All three models perform almost identically, which is itself evidence of the near-deterministic target.

### Task 2 — Classification: Nutri-Score grade (A–E)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Random Forest | 88.4% | 0.879 |
| **XGBoost** | 88.6% | **0.882** |
| MLP Neural Network | 85.9% | 0.853 |

Trained with `class_weight='balanced'` to counter the imbalance; macro-F1 is reported alongside accuracy so minority grades are not hidden.

The confusion matrix shows that misclassifications concentrate almost entirely between **adjacent** grades (A↔B, C↔D, D↔E) — the model rarely confuses distant grades. This is exactly the desirable failure mode for an ordinal scale. Nutri-Score is itself an algorithm computed from the nutrients, so ~88.6% reflects the model learning that scoring logic; it is not ~99% because Open Food Facts frequently lacks the fruit/vegetable/nut % component that the official formula uses.

### Task 3 — Conditional VAE: nutrient profile generation

A Conditional Variational Autoencoder (PyTorch, 8-dimensional latent space) was trained on the full dataset, with the target Nutri-Score grade supplied as a one-hot condition. Sampling the latent space with a chosen grade generates new nutrient profiles for that grade on demand. Training loss converged smoothly (≈ 0.64 → 0.37).

**Generated grade-A ("healthy") profiles consistently show:**

- Low fat and low saturated fat
- Low salt / sodium
- Moderate protein and fiber

**Closed-loop validation:** every generated grade-A profile is independently **confirmed as grade A by the Task-2 classifier**. The generative model (CVAE) and the discriminative model (classifier) agree, which is strong evidence the CVAE learned the true "healthy profile" distribution rather than noise.

### Task 4 — Retriever fine-tuning (headline result)

**Pipeline:** a 905-chunk knowledge corpus (20 nutrition/exercise Wikipedia articles, 512-char chunks) is indexed with FAISS. A small instruct LLM (`Qwen2.5-1.5B-Instruct`) then generates **2,715 synthetic (question, passage) pairs** with no manual labelling (the GPL/InPars idea), split into 2,307 train / 408 held-out test. The base encoder `all-MiniLM-L6-v2` is fine-tuned on the training pairs with **MultipleNegativesRankingLoss** (in-batch negatives, 3 epochs), then evaluated on the gold test pairs.

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|-----------|----------|----------|----------|-----------|-----|---------|
| Baseline (`all-MiniLM-L6-v2`) | 0.503 | 0.750 | 0.797 | 0.897 | 0.641 | 0.703 |
| **Fine-tuned** | **0.613** | **0.821** | **0.868** | **0.941** | **0.728** | **0.780** |

Domain fine-tuning improves **every** metric with no regressions — **Recall@1 +11.0 pp (+22% relative), MRR +8.6 pp, nDCG@10 +7.6 pp**. This is the core, defensible finding of the project: adapting a general encoder to a domain measurably improves retrieval, quantified cleanly with a before/after table.

### Task 5 — RAG assistant with tool use

The fine-tuned retriever feeds the top-5 passages to `Qwen2.5-1.5B-Instruct`, which answers with inline source citations. For quantitative questions the assistant instead invokes the Phase-1 models as tools:

- `predict_calories(features)` → kcal/100 g
- `predict_grade(features)` → Nutri-Score grade A–E
- `generate_profile(target_grade)` → a synthetic nutrient profile (CVAE)

This closes the loop between the two phases: the retrieval/generation stack answers open questions, while the trained tabular/generative models answer numeric ones.

---

### Key Findings

**Fat dominates calorie content.** In EDA, fat has by far the highest correlation with calories (r ≈ 0.80), followed by saturated fat (0.60) and carbohydrates (0.51). This is the Atwater 9-kcal-per-gram-of-fat rule surfacing directly in the data, and explains why the regression is near-deterministic.

**Nutri-Score errors are ordinal, not random.** The classifier's mistakes fall between neighbouring grades, never between A and E. A model that "misses" only by one grade on a subjective, formula-derived label is behaving correctly.

**A general encoder can be measurably specialised with zero labels.** Using an LLM to synthesise training pairs, the retriever gained +22% Recall@1 with no human annotation — a practical, reproducible recipe for domain adaptation.

**Generative and discriminative models agree.** The CVAE's grade-A outputs are confirmed as grade A by the independent classifier, validating that the generator captured the real premium-profile distribution.

**High accuracy is not always impressive.** Both the calorie R² (0.978) and part of the Nutri-Score accuracy come from targets that are algorithmic functions of the inputs. Reporting this honestly — and locating the genuine novelty in Phase 2 — is itself a finding.

---

### Limitations and Future Work

- The knowledge corpus is small (20 articles, 905 chunks); expanding it and using a stronger base encoder would likely widen the retrieval gap further.
- Synthetic question quality depends on the small LLM; a larger generator or light human review of pairs would improve training-signal quality.
- End-to-end answer quality is currently judged qualitatively; adding RAGAS (faithfulness, answer relevancy) would quantify the generation stage.
- A cross-encoder **reranker** on top of the fine-tuned retriever is a natural next step for another accuracy bump.
- The Nutri-Score classifier is missing the fruit/vegetable/nut % input; adding it (where available) would push accuracy higher.
- Deployment on Hugging Face Spaces would make the Gradio demo permanently accessible.

---

### Artifacts

Trained models (calorie regressor, Nutri-Score classifier, CVAE, and the fine-tuned retriever) plus metric tables are published to the Hugging Face Hub: **[oshek/nutricoach](https://huggingface.co/oshek/nutricoach)**. They can be reloaded in a fresh session without retraining.

The interactive chatbot is deployed on Hugging Face Spaces: **[oshek/nutricoach](https://huggingface.co/spaces/oshek/nutricoach)** — ask nutrition questions (RAG with citations) or use the tools tab to predict calories/Nutri-Score and generate a healthier alternative.

---

## Project Structure

```
nutricoach-rag/
├── NutriCoach_Capstone.ipynb   # Single end-to-end notebook (runs the whole pipeline)
├── README.md
├── requirements.txt
├── src/nutricoach/             # Reusable modules
│   ├── config.py                   # paths, feature schema, model names
│   ├── data.py                     # Open Food Facts download + cleaning
│   ├── models_tabular.py           # regression + classification + tool interfaces
│   ├── cvae.py                     # Conditional VAE + generate_profile()
│   ├── ingest.py                   # corpus fetching + chunking
│   ├── pairs.py                    # synthetic query–passage generation
│   ├── retriever.py                # dense retriever, FAISS index, fine-tuning
│   ├── rag.py                      # RAG orchestration + Phase-1 tools
│   └── evaluate.py                 # regression / classification / retrieval metrics
├── app/app.py                  # Gradio demo (3 tabs)
├── notebooks/                  # Per-topic notebooks (00–06) mirroring the steps
└── reports/                    # Metric tables + figures
```

**Run it:** open `NutriCoach_Capstone.ipynb` in Colab (GPU runtime), add a Kaggle token (`KAGGLE_API_TOKEN`) and a Hugging Face token (`HF_TOKEN`) as Colab Secrets, and run top to bottom. The whole pipeline (Phase 1 + Phase 2) executes in a single runtime.

# NutriCoach — Nutrition Assistant (Deep Learning Capstone)

*Predict it, grade it, generate it — then ask it anything.*

A two-phase deep-learning project on food nutrition. **Phase 1** is a self-contained tabular
project (regression + classification + a Conditional VAE). **Phase 2** wraps a Retrieval-Augmented
Generation (RAG) assistant around it, using a **domain-fine-tuned retriever** and calling the
Phase-1 models as tools.

> Built for an 8-week Deep Learning course capstone. Phase 1 is fully submittable on its own;
> Phase 2 is a designed-in extension, not a bolt-on.

---

## Why two phases

Phase 1 gives clean, reproducible results (hard metric tables). Phase 2 adds the modern GenAI
story (Transformers + RAG) and — crucially — **reuses** the Phase-1 models instead of stapling an
unrelated chatbot on top. The trained regressor / classifier / CVAE become *tools* the assistant
calls for quantitative questions.

Course coverage: classical ML baselines, generative modelling (VAE), Transformers (sentence
embeddings + a small instruct LLM), and RAG with a proper retrieval evaluation.

---

## Phase 1 — Tabular nutrition analysis

**Dataset:** open food-nutrition data (Open Food Facts export or USDA FoodData Central), nutrients
per 100 g plus a Nutri-Score grade. *(Document the exact export + filters you use here.)*

| Task | Type | Target |
|------|------|--------|
| 1. Calorie prediction | Regression | `energy_kcal_100g` |
| 2. Nutri-Score grade | Classification (A–E, imbalanced) | `nutriscore_grade` |
| 3. Profile generation | Generative (Conditional VAE) | nutrient vector conditioned on grade |

### Results — Task 1 (regression)

| Model | RMSE | MAE | R² |
|-------|-----:|----:|---:|
| Random Forest | _tbd_ | _tbd_ | _tbd_ |
| XGBoost | _tbd_ | _tbd_ | _tbd_ |
| MLP | _tbd_ | _tbd_ | _tbd_ |

### Results — Task 2 (classification)

| Model | Accuracy | Macro-F1 |
|-------|---------:|---------:|
| Random Forest | _tbd_ | _tbd_ |
| XGBoost | _tbd_ | _tbd_ |
| MLP | _tbd_ | _tbd_ |

### Task 3 — Conditional VAE

Trained to generate nutrient profiles **conditioned on a target Nutri-Score grade**. Report the
chemical signatures of generated grade-A vs grade-E profiles and cross-check them with the Task-2
classifier. *(fill in with generated-sample statistics.)*

---

## Phase 2 — RAG nutrition assistant

Pipeline: **corpus → chunk → embed → retrieve → generate (with citations)**, plus tool calls to
the Phase-1 models.

**Knowledge corpus (open sources):** Wikipedia (nutrition / supplements / exercise physiology),
OpenStax *Nutrition* (CC BY), WHO / USDA dietary guidelines. *(List exactly what you ingest.)*

**The headline experiment — fine-tuned retriever.** Base embeddings often miss domain phrasing.
We generate `(question, passage)` training pairs *synthetically* with an LLM (no manual labels,
à la GPL/InPars), fine-tune the embedding model, and measure the lift on a held-out gold set.

### Results — retrieval (baseline vs fine-tuned)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@10 |
|-----------|---------:|---------:|---------:|----:|--------:|
| Baseline (`bge-small`) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **Fine-tuned** | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

End-to-end answer quality via **RAGAS** (faithfulness, answer relevancy) on held-out questions.

**Tools.** For quantitative questions the assistant calls Phase-1 models:
`predict_calories(features)`, `predict_grade(features)`, `generate_profile(target_grade, n)`.

---

## Repository layout

```
nutricoach-rag/
├── README.md
├── requirements.txt
├── data/            raw / processed / pairs   (gitignored; download instructions above)
├── models/          trained artifacts         (gitignored; push retriever to HF Hub)
├── notebooks/
│   ├── 00_colab_runner.ipynb          clone + install + run on Colab GPU
│   ├── 01_ingest_eda.ipynb            tabular EDA + corpus ingest
│   ├── 02_train_tabular.ipynb         regression + classification
│   ├── 03_cvae.ipynb                  conditional VAE
│   ├── 04_retriever_pairs_finetune.ipynb   synthetic pairs + retriever fine-tuning
│   ├── 05_eval.ipynb                  baseline vs fine-tuned + RAGAS
│   └── 06_rag_demo.ipynb              full RAG + tool calls
├── src/nutricoach/  config, data, models_tabular, cvae, ingest, retriever, pairs, rag, evaluate
├── app/app.py       Gradio demo (3 tabs)
└── reports/figures/ plots for this README
```

---

## Quickstart

**Colab (recommended):** open **`NutriCoach_Capstone.ipynb`** (in Colab: *File → Open notebook →
GitHub →* `shekeraoleksandr/AI-NutriCoach`), set runtime to **GPU**, add a Colab Secret
`KAGGLE_API_TOKEN`, then run top to bottom. It runs the whole pipeline (Phase 1 + Phase 2) in a
single runtime, so all state persists. *(The numbered `notebooks/00–06` are the same steps split
per-topic for reading, but Colab gives each notebook its own VM, so the single notebook is the way
to actually run everything.)*

**Kaggle credentials** (needed to download the dataset): Kaggle → *Settings → API → Create New
Token* gives `kaggle.json`. Locally put it at `~/.kaggle/kaggle.json` (`chmod 600`); on Colab
upload it or use Colab Secrets (`KAGGLE_USERNAME` / `KAGGLE_KEY`). Never commit it — it's gitignored.

**Local:**

```bash
git clone https://github.com/shekeraoleksandr/AI-NutriCoach.git
cd AI-NutriCoach
pip install -r requirements.txt
python app/app.py          # launch the Gradio demo (after training artifacts exist)
```

Models are loaded lazily, so the app and modules import cleanly before everything is trained.

---

## Roadmap

- [ ] Phase 1: dataset download + EDA (`01`)
- [ ] Phase 1: regression & classification results (`02`)
- [ ] Phase 1: Conditional VAE (`03`)
- [ ] Phase 2: corpus ingest + baseline index (`01`, `04`)
- [ ] Phase 2: synthetic pairs + retriever fine-tuning (`04`)
- [ ] Phase 2: retrieval eval + RAGAS (`05`)
- [ ] Phase 2: full RAG demo + tool calls (`06`)
- [ ] Deploy Gradio demo to Hugging Face Spaces

## Limitations & future work

Nutri-Score labels are coarse; the class distribution is imbalanced (addressed with class weights /
SMOTE). The RAG retrieval metric depends on the quality of synthetic pairs — worth spot-checking a
sample by hand. A reranker (cross-encoder) is a natural stretch goal on top of the fine-tuned
retriever.

## License

MIT — see `LICENSE`.

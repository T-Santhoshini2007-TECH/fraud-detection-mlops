# 🔍 Interpretable Fraud Detection with Drift-Aware MLOps

A complete, end-to-end fraud detection system — not a notebook with an accuracy
score, but the full pipeline a real team would build: training → tracking →
serving → explainability → drift monitoring → CI/CD retraining triggers.

**Live demo:** _add your Render URL here after deploying_
**API docs:** _add your Render URL/docs here_

---

## Why this project exists

Most student fraud-detection projects stop at "99.9% accuracy" on a notebook.
That number is almost meaningless here — with fraud at ~0.17% of transactions,
a model that **always predicts "not fraud"** also scores ~99.8% accuracy
while catching zero fraud.

This project is built around the actual hard parts of deploying fraud
detection in production:

1. **Severe class imbalance** — handled with cost-sensitive thresholding,
   not naive resampling, so probability calibration is preserved.
2. **Interpretability over raw performance** — a flagged transaction needs
   a reason a human can act on. The primary model is logistic regression
   with exact SHAP-style explanations, not a black-box ensemble.
3. **Concept drift** — fraud patterns evolve as fraudsters adapt. This
   system doesn't just claim to monitor drift; it demonstrates a real,
   important failure mode of naive drift monitoring (see below) and fixes it.

## The key finding (read this before judging the metrics)

This project deliberately demonstrates something most "drift monitoring"
demos skip: **on a severely imbalanced problem, population-level drift
detection can completely miss a real shift in fraud patterns.**

| Detection method | What it measures | Result on injected drift |
|---|---|---|
| Population-level feature drift (PSI/KS) | Whole transaction population | ❌ Misses it (99.75% of rows are unaffected legitimate transactions) |
| Model score drift | Predicted probability distribution | ❌ Also misses it (same reason — averaged over the population) |
| **Fraud-conditional drift** | Only confirmed-fraud rows | ✅ **Catches it clearly** (PSI > 10, p < 1e-10) |

This isn't a contrived result — it's a documented, real limitation of
production fraud monitoring (rare-class drift is invisible at the population
level), and the fix (conditional monitoring once labels arrive) is exactly
what real fraud teams do. See [`docs/DRIFT_FINDINGS.md`](docs/DRIFT_FINDINGS.md)
for the full writeup with numbers.

## Architecture

```
                 ┌─────────────┐
                 │   Dataset    │  Kaggle creditcard.csv (or synthetic
                 │  (raw CSV)   │  fallback with the same schema + injected drift)
                 └──────┬──────┘
                        │ time-ordered split (not random — prevents leakage)
                        ▼
              ┌──────────────────┐
              │ Feature Pipeline │  scaling fit ONLY on train, hour-of-day
              └────────┬─────────┘  feature, passthrough PCA features
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Training (MLflow tracked)   │  Logistic Regression (primary,
        │  + cost-sensitive threshold   │  interpretable) vs Gradient
        │     selection                 │  Boosted Trees (comparison)
        └───────────────┬───────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   ┌─────────────────┐    ┌──────────────────┐
   │   FastAPI        │    │ Streamlit         │
   │   /predict        │    │ Monitoring         │
   │   + SHAP explain  │    │ Dashboard          │
   └─────────────────┘    └──────────────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
              ┌──────────────────┐
              │  Drift Detection  │  Population / Score / Conditional
              │  (3 methods)      │  PSI + KS-test
              └────────┬─────────┘
                        ▼
              ┌──────────────────┐
              │  GitHub Actions   │  Scheduled drift check →
              │  retrain trigger  │  auto-retrain if significant
              └──────────────────┘
```

## What's actually in this repo

```
src/
├── data/load_data.py       # Time-ordered split (prevents leakage), real
│                            # dataset loader with synthetic fallback that
│                            # injects realistic concept drift
├── features/pipeline.py     # Leakage-safe scaling (fit on train only)
├── models/
│   ├── train.py             # Dual-model training, cost-sensitive threshold
│   │                        # selection, MLflow tracking
│   └── explain.py           # SHAP explanations (exact for linear models)
├── monitoring/drift.py      # PSI + KS-test, 3 complementary drift views
└── api/main.py               # FastAPI serving + live drift endpoint

dashboard/app.py              # Streamlit dashboard (overview, live scoring,
                              # drift comparison with the key finding above)

tests/                        # 27 tests: data integrity, leakage prevention,
                              # drift detection correctness, API contracts

docker/                       # Dockerfile per service + docker-compose
.github/workflows/
├── ci.yml                    # Test + build on every push
└── drift-monitor.yml         # Scheduled drift check + auto-retrain trigger
```

## Running it

### Quickest path (Docker, recommended)
```bash
git clone <your-repo-url>
cd fraud-detection-system
docker compose -f docker/docker-compose.yml up --build
```
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

### Local (no Docker)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Train (generates models/*.joblib + data/processed/train_baseline.csv)
python -m src.models.train

# 2. Serve the API
uvicorn src.api.main:app --reload --port 8000

# 3. (separate terminal) Run the dashboard
streamlit run dashboard/app.py
```

### Run the tests
```bash
pytest tests/ -v
```

### Use the REAL dataset (recommended before deploying)
The synthetic fallback exists so the repo runs without any setup, but for
real results:
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/ulb-machine-learning-group/creditcardfraud)
2. Place it at `data/raw/creditcard.csv`
3. Re-run `python -m src.models.train` — it auto-detects the real file.

## Model performance

Run `python -m src.models.train` and check the console output, or the
MLflow UI (`mlflow ui` from the repo root) for full tracked metrics. The
training script reports precision, recall, F1, ROC-AUC, PR-AUC, and a
**cost-weighted business metric** (assumed costs documented in
`src/models/train.py` — these are illustrative, not a claim about real
fraud economics, and easy to override for your own cost assumptions).

On the synthetic fallback data, results typically look like:

| Model | Precision | Recall | F1 | Business cost* |
|---|---|---|---|---|
| Logistic Regression (primary) | ~0.49 | ~0.96 | ~0.65 | lower — misses almost no fraud |
| Gradient Boosted Trees (comparison) | ~0.88 | ~0.88 | ~0.88 | higher — fewer false alarms but misses more fraud |

*Lower is better; cost = (missed fraud × assumed loss) + (false alarms × review cost).
The "better F1" model isn't necessarily the better business decision —
this tension is the point, and is discussed further in the dashboard.

## Honest limitations

- The synthetic data fallback is a reasonable stand-in for demoing the
  pipeline, but is **not** a substitute for the real Kaggle dataset for
  any claim about real-world fraud detection performance.
- The cost assumptions (`COST_FALSE_NEGATIVE`, `COST_FALSE_POSITIVE` in
  `src/models/train.py`) are illustrative defaults, not researched figures.
- This is a portfolio/demonstration project, not a production-hardened
  system — there's no authentication on the API, no real-time streaming
  ingestion, and the "retraining" CI job doesn't push to a real model
  registry (left as a documented extension point).

## Tech stack

`scikit-learn` · `imbalanced-learn` · `SHAP` · `MLflow` · `FastAPI` ·
`Streamlit` · `Docker` · `GitHub Actions` · `pytest`

## License

MIT — see [LICENSE](LICENSE).

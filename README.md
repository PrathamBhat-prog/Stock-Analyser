# 📈 ML Time-Series Stock Analyser

A **production-grade machine learning pipeline** that fetches 10 years of live market data from **yfinance**, engineers 70+ technical features, trains and compares **7 models** (including XGBoost, LightGBM, and LSTM), benchmarks against industry baselines, and serves **BUY / SELL / HOLD** decisions via FastAPI + Gradio UI.

> **Disclaimer:** Research/education project. Past performance does not guarantee future results. Not financial advice.

---

## 🏗️ Architecture

```
yfinance OHLCV (10 years, 32 tickers — US large-caps + NSE India)
    │
    ▼
Feature Engineering ──► 70+ features:
    │   Lagged returns/prices/volume (1-20d)
    │   Rolling mean/std (5/10/20/60d windows)
    │   Momentum (5/10/20/60d)
    │   RSI (14 & 28 period)
    │   MACD + Signal Line + Histogram
    │   Bollinger Bands (mid/upper/lower/width/pct)
    │   ATR (14d, normalised)
    │   OBV + OBV Rate-of-Change (10d)
    │   Price vs MA20 / MA60 (regime signals)
    │   High-Low range + 5d moving average
    │
    ▼
Binary Label: Close[t+5] > Close[t]  (5-day direction)
    │
    ▼
Chronological split: 70% train / 15% val / 15% test  (no lookahead leakage)
    │
    ▼
Model Comparison ──► Logistic Regression
                 ──► Random Forest
                 ──► Gradient Boosting
                 ──► Hist Gradient Boosting
                 ──► XGBoost       ◄── industry standard
                 ──► LightGBM      ◄── quant-fund favourite
                 ──► LSTM (PyTorch)◄── sequence model
    │
    ▼
Best model selected by validation ROC-AUC
    │
    ▼
Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC,
            MAE, RMSE, MAPE, Directional Accuracy, Sharpe Ratio
    │
    ▼
Benchmark vs literature baselines (random guess → Transformer)
    │
    ▼
Inference ──► BUY / SELL / HOLD + confidence
    │
    ▼
MLflow tracking  +  Epoch plot (docs/training_results/epoch_plot.png)
```

---

## 🧠 Models & Why They Are Used

| Model | Type | Why Production-Grade |
|-------|------|---------------------|
| Logistic Regression | Linear | Interpretable baseline; required by many regulatory frameworks |
| Random Forest | Ensemble | Robust, handles non-linearity, low variance |
| Gradient Boosting | Ensemble | Classic GBM; solid across financial domains |
| Hist Gradient Boosting | Ensemble | Faster GBM; handles NaNs natively |
| **XGBoost** | Gradient Boost | Industry standard in quant finance; winner of most Kaggle finance competitions |
| **LightGBM** | Gradient Boost | Preferred by hedge funds; 10–20× faster than GBM; leaf-wise growth |
| **LSTM** | Deep Learning | Captures long-range temporal patterns in price sequences |

---

## 📊 Evaluation Metrics Explained

All metrics are computed on a **held-out chronological test set** (no data snooping).

| Metric | Category | Why it Matters for Stock Prediction |
|--------|----------|-------------------------------------|
| **ROC-AUC** | Classification | Primary selection metric. Robust to class imbalance in direction labels. 0.5 = random, >0.56 = significant |
| **Accuracy** | Classification | % of correct UP/DOWN calls. Floor is ~50% (random walk) |
| **Precision** | Classification | Of predicted UPs, how many were actually UP — reduces false long signals |
| **Recall** | Classification | Of actual UPs, how many we caught — measures signal completeness |
| **F1** | Classification | Harmonic mean of Precision & Recall; balanced for imbalanced datasets |
| **MAE** | Calibration | Mean absolute error of predicted probability vs true label; measures calibration |
| **RMSE** | Calibration | Root MSE of predicted probability; penalises large confidence errors |
| **MAPE** | Calibration | Mean absolute percentage error; scale-invariant calibration check |
| **Directional Accuracy** | Finance | Fraction of correct directional calls. >52% is economically meaningful at scale |
| **Sharpe Ratio** | Finance | Annualised risk-adjusted return of a naïve long-flat strategy driven by model predictions. >1.0 is production-ready |

---

## 📈 Market Benchmarks (from literature)

Our model is automatically compared against these industry references:

| Baseline | Accuracy | F1 | ROC-AUC | Source |
|----------|----------|----|---------|--------|
| Random Guess | 0.50 | 0.50 | 0.50 | Theoretical floor |
| Buy & Hold Majority | 0.52 | 0.55 | 0.52 | Naive baseline |
| ARIMA Directional Proxy | 0.51 | 0.52 | 0.53 | Classical TS model |
| **XGBoost (Industry median)** | **0.55** | **0.56** | **0.57** | QuantLib survey 2023 |
| LSTM Daily Direction (lit.) | 0.54 | 0.58 | 0.56 | Fischer & Krauss 2018 |
| Transformer Finance (lit.) | 0.56 | 0.60 | 0.58 | Lim et al. 2021 |

---

## 🚀 Quick Start

### 1. Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train models (full 10-year run)

```powershell
# Full training — all 32 tickers, 10 years, all 7 models
python train.py

# Custom tickers
python train.py --tickers AAPL MSFT GOOGL --period 10y

# Specific models only
python train.py --models xgboost lightgbm lstm
```

**Training outputs:**
- `artifacts/models/best_model.joblib` or `best_model.pt` (LSTM)
- `artifacts/models/model_metadata.json` — full metrics & feature list
- `artifacts/models/benchmark_comparison.json` — vs industry baselines
- `artifacts/models/feature_importance.csv` — top features (tree models)
- `docs/training_results/epoch_plot.png` — **LSTM epoch proof plot**
- MLflow experiment runs in `mlruns/`

### 3. View epoch plot (LSTM training proof)

```powershell
# After training completes, open:
Invoke-Item docs\training_results\epoch_plot.png
```

The plot shows:
- **Blue line**: Training BCE loss per epoch (left axis)
- **Orange line**: Validation ROC-AUC per epoch (right axis)
- **Green dashed**: Best epoch (model restored to this checkpoint)
- **Red dotted**: Early-stop epoch (if triggered)

### 4. View MLflow experiments

```powershell
mlflow ui --backend-store-uri mlruns
```
Open http://localhost:5000

### 5. Run inference

**Gradio UI:**
```powershell
python -m src.ui.gradio_app
```
Open http://localhost:7860

**FastAPI:**
```powershell
uvicorn src.main:app --reload
```
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "period": "2y"}'
```

### 6. Docker
```powershell
docker compose up --build
```
- Gradio UI: http://localhost:7860
- MLflow UI: http://localhost:5000

---

## 🧪 Local Testing Commands

Run these in order to verify everything works end-to-end:

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Install / update dependencies (includes xgboost & lightgbm)
pip install -r requirements.txt

# 3. Smoke test — verify pipeline imports & feature engineering
python verify_pipeline.py

# 4. Quick training run (3 tickers, 5y, all models)
#    Produces: artifacts/models/ + docs/training_results/epoch_plot.png
python train.py --tickers AAPL MSFT GOOGL --period 5y

# 5. Open the LSTM epoch proof plot
Invoke-Item docs\training_results\epoch_plot.png

# 6. Full production training (32 tickers, 10 years) — takes 15-45 min
python train.py

# 7. View MLflow results
mlflow ui --backend-store-uri mlruns

# 8. Launch Gradio UI
python -m src.ui.gradio_app
```

---

## 🗂️ Project Structure

```
src/
  config/ml_config.py         # tickers, hyperparameters, paths
  data/
    fetch_data.py             # yfinance OHLCV ingestion
    validate_data.py          # schema + timezone checks
    features.py               # 70+ production-grade features
    labeling.py               # forward-return binary labels
    dataset.py                # multi-ticker dataset builder
    sequences.py              # LSTM sliding-window sequences
  models/
    feature_columns.py        # canonical feature list (shared)
    metrics.py                # finance-grade evaluation metrics
    trainer.py                # multi-model training + comparison
    lstm_model.py             # PyTorch LSTM + epoch plot
    predictor.py              # inference loader (sklearn or LSTM)
  agents/
    ml_agent.py               # ML prediction agent
    decision_agent.py         # probability → BUY/SELL/HOLD
  pipelines/
    training_pipeline.py      # full training workflow
    inference_pipeline.py     # full inference workflow
  ui/gradio_app.py            # web UI
  main.py                     # FastAPI
train.py                      # training CLI
verify_pipeline.py            # smoke test
docs/training_results/        # epoch_plot.png generated here
```

---

## 🔧 Configuration (`src/config/ml_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_TRAIN_TICKERS` | 32 tickers | US large-caps + NSE India |
| `TRAIN_PERIOD` | `"10y"` | yfinance history window |
| `FORECAST_HORIZON_DAYS` | `5` | Days ahead for prediction |
| `SEQUENCE_LENGTH` | `60` | LSTM lookback window (days) |
| `LSTM_HIDDEN_SIZE` | `128` | LSTM hidden state size |
| `LSTM_EPOCHS` | `60` | Max training epochs (early-stop enabled) |
| `LSTM_PATIENCE` | `8` | Early-stop patience |
| `PRIMARY_METRIC` | `"roc_auc"` | Model selection metric |

---

## 🤔 Decision Logic

| Condition | Action |
|-----------|--------|
| P(up) ≥ 0.58 | **BUY** |
| P(up) ≤ 0.42 | **SELL** |
| otherwise | **HOLD** |

---

## ⚠️ Limitations

- yfinance provides **daily OHLCV** — not tick-level or real-time data
- ~25k–80k rows is small vs production quant systems (millions+)
- Directional accuracy near 50–55% is typical for daily equity prediction
- No transaction costs, slippage, or portfolio-level optimisation
- Single-asset inference only (no portfolio allocation)

---

## ☁️ AWS Deployment *(coming soon)*

> AWS account configuration in progress. Deployment guide will cover:
> - **ECR** — Docker image registry
> - **SageMaker** — managed model training & endpoints
> - **Lambda + API Gateway** — serverless inference
> - **S3** — artifact and model storage

---

## 👤 Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)
📧 prathambhat75@gmail.com

---

## 📄 License

MIT

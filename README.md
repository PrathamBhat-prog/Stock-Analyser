# ML Time-Series Stock Analyser

A quant-style machine learning pipeline that fetches live market data from **yfinance**, engineers time-series features, trains multiple models (including **LSTM**), compares them against market baselines, and serves **BUY / SELL / HOLD** decisions via API and Gradio UI.

> **Disclaimer:** This is a research/education project. Past performance does not guarantee future results. Not financial advice.

---

## Architecture

```
yfinance (OHLCV)
    │
    ▼
Data validation + time-series features (lags, rolling stats, momentum)
    │
    ▼
Label: will price be higher in N days? (binary)
    │
    ▼
Chronological train / val / test split (no lookahead leakage)
    │
    ▼
Model comparison ──► Logistic Regression, Random Forest, Gradient Boosting, LSTM
    │
    ▼
Best model selected by validation ROC-AUC
    │
    ▼
Inference ──► BUY / SELL / HOLD + confidence + reasoning
    │
    ▼
MLflow experiment tracking + artifact storage
```

---

## Features

- **32 tickers** (US large caps + India NSE) with **10 years** of history (~25k+ rows)
- **36 time-series features**: lagged returns/prices/volume, rolling statistics, momentum
- **5-day forward direction** prediction (binary classification)
- **Models compared**: Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, **LSTM**
- **Quant-style evaluation**: ROC-AUC primary metric, chronological splits, class balancing
- **MLOps**: MLflow logging, model artifacts, benchmark comparison vs literature baselines
- **Interfaces**: FastAPI (`/analyze`), Gradio UI, Docker Compose

---

## Quick Start

### 1. Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train models

```powershell
python train.py
```

Custom tickers / period:

```powershell
python train.py --tickers AAPL MSFT GOOGL --period 10y
python train.py --models lstm random_forest
```

Training outputs:
- `artifacts/models/best_model.joblib` or `best_model.pt` (LSTM)
- `artifacts/models/model_metadata.json`
- `artifacts/models/benchmark_comparison.json`
- MLflow runs in `mlruns/`

### 3. View experiments

```powershell
mlflow ui --backend-store-uri mlruns
```

Open http://localhost:5000

### 4. Run inference

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
  -d '{"ticker": "AAPL", "period": "1y"}'
```

### 5. Docker

```powershell
docker compose up --build
```

- Gradio UI: http://localhost:7860
- MLflow UI: http://localhost:5000

---

## Project Structure

```
src/
  config/ml_config.py       # tickers, hyperparameters, paths
  data/
    fetch_data.py           # yfinance ingestion
    validate_data.py        # schema checks
    features.py             # time-series feature engineering
    labeling.py             # forward-return labels
    dataset.py              # multi-ticker dataset builder
    sequences.py            # LSTM sequence windows
  models/
    feature_columns.py      # shared feature list
    trainer.py              # multi-model training + comparison
    lstm_model.py           # PyTorch LSTM classifier
    predictor.py            # inference loader (sklearn or LSTM)
  agents/
    ml_agent.py             # ML prediction agent
    decision_agent.py       # probability → BUY/SELL/HOLD
  pipelines/
    training_pipeline.py    # full training workflow
    inference_pipeline.py   # full inference workflow
  ui/gradio_app.py          # web UI
  main.py                   # FastAPI
train.py                    # training CLI
verify_pipeline.py          # smoke test
```

---

## ML Workflow (what a quant engineer would do)

| Step | What we do |
|------|------------|
| Data sourcing | yfinance daily OHLCV, multi-ticker universe |
| Cleaning | Column validation, timezone normalization, sort by date |
| Features | Lags (1–20d), rolling mean/std, momentum, volume ratios |
| Label | Binary: `Close[t+5] > Close[t]` |
| Split | Chronological 70/15/15 — no random shuffle |
| Baselines | 4 sklearn models + LSTM sequence model |
| Selection | Best validation **ROC-AUC** (robust to class imbalance) |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC on held-out test set |
| Benchmarks | Compared vs random guess, ARIMA proxy, LSTM/Transformer literature |
| Tracking | MLflow params, metrics, artifacts |
| Serving | Load best model, predict on latest window, map to action |

---

## Decision Logic

| Condition | Action |
|-----------|--------|
| P(up) ≥ 0.58 | **BUY** |
| P(up) ≤ 0.42 | **SELL** |
| otherwise | **HOLD** |

---

## Configuration

Edit `src/config/ml_config.py`:

- `DEFAULT_TRAIN_TICKERS` — training universe
- `TRAIN_PERIOD` — yfinance history window (`10y`)
- `FORECAST_HORIZON_DAYS` — prediction horizon (default 5)
- `SEQUENCE_LENGTH` — LSTM lookback window (default 30)
- `PRIMARY_METRIC` — model selection metric (`roc_auc`)

---

## Limitations

- yfinance is **not** tick-level real-time data
- ~25k–80k rows is small vs production quant systems (millions+)
- Directional accuracy near 50–55% is typical for daily equity prediction
- No transaction costs, slippage, or portfolio optimization
- Cloud deployment guide coming next

---

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)

---

## License

MIT

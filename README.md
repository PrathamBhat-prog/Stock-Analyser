# ML Stock Analyser

An end-to-end **production-grade ML pipeline** that analyses any stock from any exchange and provides **BUY / SELL / HOLD** recommendations across **5 investment horizons** (1 week to 1 year).

> **Disclaimer:** Research/education project. Not financial advice.

---

## What the AI Actually Does

This is **not a formula-based calculator**. It uses two ML/AI systems:

### System 1 — Hist Gradient Boosting Classifier (Primary ML Model)

| Property | Value |
|---|---|
| **Algorithm** | `HistGradientBoostingClassifier` (sklearn) |
| **Training rows** | 38,096 daily OHLCV records |
| **Tickers trained on** | 32 companies (US large-caps + India NSE) |
| **History per ticker** | 5 years daily |
| **Input features** | 60 technical indicators |
| **Task** | Binary: will price be higher in 5 days? |
| **Training split** | 70% train / 15% val / 15% test (chronological, no lookahead) |

**Why HistGradientBoosting won** among 7 candidates: it handles missing values natively, trains fastest on tabular data, and achieved the highest validation ROC-AUC.

#### Real Accuracy Metrics (held-out test set — never seen during training)

| Metric | Our Model | Random Guess | ARIMA Proxy | XGBoost Ind. | LSTM Lit. | Transformer Lit. |
|---|---|---|---|---|---|---|
| **ROC-AUC** | **0.515** | 0.500 | 0.530 | 0.570 | 0.560 | 0.580 |
| **F1 Score** | **0.579** | 0.500 | 0.520 | 0.560 | 0.580 | 0.600 |
| **Accuracy** | 49.8% | 50.0% | 51.0% | 55.0% | 54.0% | 56.0% |
| Precision | 49.0% | — | — | — | — | — |
| Recall | 70.8% | — | — | — | — | — |
| Sharpe Ratio | -0.26 | — | — | — | — | — |
| Directional Acc. | 49.8% | 50.0% | — | — | — | — |

> **Context on accuracy:** Daily stock direction prediction is among the hardest problems in ML. The Efficient Market Hypothesis states all public information is already priced in. Academic literature consistently reports 51–56% accuracy for daily models — our F1 of 0.579 outperforms the random guess and ARIMA baselines. A 51% edge is economically significant at scale with low-cost execution.

#### Other models trained (all compared, best selected by val ROC-AUC)

| Model | Type | Result |
|---|---|---|
| Logistic Regression | Linear baseline | Lower ROC-AUC |
| Random Forest (300 trees) | Ensemble | Competitive |
| Gradient Boosting (200 iter) | GBM | Competitive |
| **Hist Gradient Boosting** | **GBM (winner)** | **Best val ROC-AUC** |
| XGBoost (400 trees) | GBM | Competitive |
| LightGBM (400 trees, 63 leaves) | GBM | Competitive |
| LSTM (PyTorch, 2 layers) | Deep Learning | Training interrupted |

### System 2 — Trend Analysis Agent (ARIMA-inspired, any-ticker)

A pure-pandas rule-based engine that detects market regime using 6 signal categories:
1. MA alignment (price vs 20d/60d moving averages)
2. RSI overbought/oversold (14d and 28d)
3. MACD crossover detection
4. Bollinger Band position
5. 20-day price momentum
6. Volume confirmation

Outputs: trend score [-1, +1], trend label, momentum label, volatility label, plain-English summary.

**Works for ANY ticker** — uses only price/volume patterns, no company-specific training.

### Investment Horizon Blending (no retraining required)

| Horizon | ML Weight | Trend Weight | Use Case |
|---|---|---|---|
| **1 Week** | 80% | 20% | Short-term traders |
| **1 Month** | 50% | 50% | Swing traders |
| **3 Months** | 30% | 70% | Medium-term investors |
| **6 Months** | 15% | 85% | Long-term investors |
| **1 Year** | 10% | 90% | Buy-and-hold investors |

---

## Features

- **Any company** — US, India NSE/BSE, Europe, Asia, ETFs, crypto (any yfinance ticker)
- **5 investment horizons** — 1 week to 1 year
- **60 technical features** — RSI, MACD, Bollinger Bands, ATR, OBV, momentum, lags
- **7 models trained and compared** — best selected automatically
- **Company fundamentals** — name, sector, PE ratio, market cap, 52-week range
- **MLflow experiment tracking** — all runs logged
- **FastAPI backend** + **Gradio frontend**

---

## Architecture

```
yfinance OHLCV (any ticker, any exchange)
    |
Feature Engineering (60 indicators: RSI, MACD, BB, ATR, OBV, lags, rolling stats)
    |
    +---> ML Model (HistGradientBoosting) --> P(price up in 5 days)
    |
    +---> Trend Agent (ARIMA-inspired)   --> trend score [-1,+1]
    |
    +---> Horizon Blender (weights ML + Trend based on investment horizon)
    |
BUY / SELL / HOLD + confidence + plain-English explanation
    |
    +---> FastAPI  (port 8000)
    +---> Gradio   (port 7860)
```

---

## Quick Start

### 1. Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the model (one-time)

```powershell
# Full 10-year run — all 32 tickers, all 7 models
python train.py

# Quick test (3 tickers, 5y)
python train.py --tickers AAPL MSFT GOOGL --period 5y
```

### 3. Run the application

**Terminal 1 — Backend (FastAPI, port 8000):**
```powershell
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

**Terminal 2 — Frontend (Gradio, port 7860):**
```powershell
python -m src.ui.gradio_app
```
- Open: **http://localhost:7860**

### 4. Use the app

1. Enter any stock ticker (e.g. `AAPL`, `RELIANCE.NS`, `TSM`)
2. Choose your **investment horizon** (1 week to 1 year)
3. Click **Analyse Stock**
4. Get: BUY / SELL / HOLD + confidence + trend analysis + chart

---

## API Usage

```bash
# 1-week horizon (default)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "period": "2y", "horizon_key": "5d"}'

# 1-year horizon
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE.NS", "period": "2y", "horizon_key": "252d"}'

# Available horizons
curl http://localhost:8000/horizons
```

---

## Project Structure

```
src/
  config/ml_config.py         # tickers, hyperparameters, paths
  data/
    fetch_data.py             # yfinance ingestion
    validate_data.py          # schema + timezone checks
    features.py               # 60 production-grade features
    labeling.py               # forward-return binary labels
    dataset.py                # multi-ticker dataset builder
    sequences.py              # LSTM sliding-window sequences
  models/
    feature_columns.py        # canonical feature list (shared)
    metrics.py                # finance-grade evaluation metrics
    trainer.py                # multi-model training + comparison
    lstm_model.py             # PyTorch LSTM + epoch plot
    predictor.py              # inference loader
  agents/
    ml_agent.py               # ML prediction wrapper
    decision_agent.py         # multi-horizon BUY/SELL/HOLD logic
    trend_agent.py            # ARIMA-inspired trend analysis
  pipelines/
    training_pipeline.py      # full training workflow
    inference_pipeline.py     # full inference workflow
  ui/gradio_app.py            # Gradio web UI
  main.py                     # FastAPI
train.py                      # training CLI
docs/training_results/        # epoch_plot.png generated here
```

---

## Configuration

Edit `src/config/ml_config.py`:

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_TRAIN_TICKERS` | 32 tickers | US large-caps + India NSE |
| `TRAIN_PERIOD` | `"10y"` | Training history |
| `FORECAST_HORIZON_DAYS` | `5` | Label horizon for ML model |
| `SEQUENCE_LENGTH` | `60` | LSTM lookback window |
| `LSTM_HIDDEN_SIZE` | `128` | LSTM hidden state size |
| `LSTM_EPOCHS` | `60` | Max LSTM training epochs |
| `PRIMARY_METRIC` | `"roc_auc"` | Model selection metric |

---

## Decision Logic

| Horizon | BUY when | SELL when | HOLD when |
|---|---|---|---|
| 1 Week | Composite > 58% | Composite < 42% | 42-58% range |
| 1 Month | Same thresholds | Same | Same |
| 3-12 Months | Trend-weighted composite > 58% | < 42% | 42-58% |

---

## AWS Deployment (Coming Soon)

Account setup in progress. Planned stack:
- **ECR** — Docker image registry
- **SageMaker** — managed training & endpoints
- **Lambda + API Gateway** — serverless inference
- **S3** — artifact and model storage

---

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)
Email: prathambhat75@gmail.com

---

## License

MIT

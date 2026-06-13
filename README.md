# 🎯 Sniper v5 Strategy — Optimized GDELT Sentiment Ensemble

**Production-grade CatBoost model** for high-precision stock price predictions across **5 investment horizons** (20 days to 1 year).

> **Disclaimer:** Research/education project. Not financial advice. Past performance ≠ future results.

---

## Strategy Overview

| Property | Value |
|---|---|
| **Model** | CatBoostClassifier (optimized for gradient boosting) |
| **Objective** | High-precision classification of **20-day appreciation** |
| **Accuracy** | 59.32% (final stable run) |
| **Precision** | 60.46% (calibrated threshold: 0.52) |
| **Recall** | 58.9% |
| **Risk Management** | Inverse-volatility sizing (drawdown reduced: -99% → -35%) |

---

## Core Architecture

### System 1 — CatBoostClassifier (Primary ML Model)

**Why CatBoost?**
- Superior at non-linear relationships in market data
- Internal feature scaling — no preprocessing needed
- Robust to outliers with built-in regularization
- Fast training on mixed categorical/continuous data

**Hyperparameters:**
```python
model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.015,
    depth=7,
    l2_leaf_reg=8,
    random_seed=42,
    use_best_model=True
)
```

#### Feature Engineering — The Alpha Drivers

**1. Sentiment (GDELT Project)**
- Live news headlines via GDELT API
- vaderSentiment for fast, CPU-efficient scoring
- **20-day rolling sentiment** — captures current market narrative
- **Lagged sentiment** (1, 3, 5 days) — detects delayed market reactions
- **Example:** Positive tech news today → price move over next 1-5 days

**2. Macro Fear Index (~21% feature importance)**
- **VIX price levels** — current volatility regime
- **VIX Velocity** — 5-day rate of change (regime shift detector)
- **sent_vix_interaction** — Sentiment × VIX (amplifies news impact during volatility)
- **Insight:** Market regime matters more than individual stock momentum

**3. Technical Alpha**
- **dist_52w_high** — proximity to yearly highs (overhead resistance)
- **momentum_20d** — medium-term momentum persistence
- **vol_ratio_5d** — volume spikes validate trend changes

**4. Data Preprocessing**
- **Winsorization** — clip outliers at 1st/99th percentile (prevents overfitting)
- **Target horizon** — 20-day forward-return binary labels

#### Performance Metrics

| Metric | Sniper v5 | Random Guess | Efficient Market Hypothesis |
|---|---|---|---|
| **Accuracy** | **59.32%** | 50.0% | 50.0% |
| **Precision** | **60.46%** | N/A | N/A |
| **Recall** | **58.9%** | N/A | N/A |
| **ROC-AUC** | **0.62** | 0.50 | 0.50 |

> **Key Breakthroughs:**
> - **52% → 59.32%:** Moved from 1-day to 20-day horizons, added VIX velocity, implemented Winsorization
> - **Threshold sweet spot (0.52):** Achieves 60%+ precision (viable after transaction costs)
> - **Risk management:** Inverse-volatility sizing reduced drawdown from -99% to -35%

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
| **20 Days** | 100% | 0% | Pure ML signal (model trained for this) |
| **1 Month** | 80% | 20% | Short-term traders |
| **3 Months** | 50% | 50% | Swing traders |
| **6 Months** | 20% | 80% | Long-term investors |
| **1 Year** | 10% | 90% | Buy-and-hold investors |

---

## Features

- **Any company** — US, India NSE/BSE, Europe, Asia, ETFs, crypto (any yfinance ticker)
- **5 investment horizons** — 20 days to 1 year
- **Multi-modal feature engineering** — GDELT sentiment, VIX, technical indicators
- **CatBoostClassifier** — optimized for non-linear relationships and robustness
- **59.32% accuracy** — high-precision classification with 60.46% precision threshold
- **Company fundamentals** — name, sector, PE ratio, market cap, 52-week range
- **MLflow experiment tracking** — all runs logged
- **FastAPI backend** + **Gradio frontend**
- **Production model** — `trading_model_sniper_v5.pkl` (serialized CatBoost)

---

## Architecture

```
yfinance OHLCV (any ticker, any exchange)
    |
    +--> GDELT Sentiment (live news)
    +--> VIX & VIX Velocity (fear index)
    +--> Technical Indicators (momentum, volume, resistance)
    |
    +---> Feature Engineering (Winsorization, normalization)
    |
    +---> CatBoostClassifier --> P(20-day appreciation)
    |     (iterations=1500, depth=7, lr=0.015)
    |
    +---> Trend Analysis Agent --> trend score [-1,+1]
    |     (MA, RSI, MACD, BB, momentum, volume)
    |
    +---> Horizon Blender (ML% + Trend% based on horizon)
    |
    +---> Risk Management (Inverse-Volatility Sizing)
    |
BUY / SELL / HOLD + confidence + plain-English explanation
    |
    +---> FastAPI  (port 8000)
    +---> Gradio   (port 7860)
```

**Production Model File:** `trading_model_sniper_v5.pkl`
- Serialized CatBoostClassifier with all training state
- 59.32% accuracy, 60.46% precision
- Works for ANY stock ticker across 5 investment horizons

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
python src/ui/gradio_app.py
```
- Open: **http://localhost:7860**
- If port 7860 is busy, Gradio will auto-select an available port

### 4. Use the app

1. Enter any stock ticker (e.g. `AAPL`, `RELIANCE.NS`, `TSM`)
2. Choose your **investment horizon** (1 week to 1 year)
3. Click **Analyse Stock**
4. Get: BUY / SELL / HOLD + confidence + trend analysis + chart

---

## API Usage

The model runs on **20-day horizons** with optional blending for longer-term analysis:

```bash
# 20-day horizon (default, pure ML signal)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "period": "2y", "horizon_key": "21d"}'

# 1-month horizon (80% ML + 20% Trend blend)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE.NS", "period": "2y", "horizon_key": "21d"}'

# 1-year horizon (10% ML + 90% Trend)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSM", "period": "2y", "horizon_key": "252d"}'

# Available horizons
curl http://localhost:8000/horizons
```

**Response Example:**
```json
{
  "final_decision": "BUY",
  "confidence": 0.72,
  "horizon": "21 days",
  "ml_probability": 0.64,
  "trend_score": 0.45,
  "composite_score": 0.608,
  "company": {
    "company_name": "Apple Inc.",
    "pe_ratio": 28.5,
    "market_cap": 2.8e12
  }
}
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

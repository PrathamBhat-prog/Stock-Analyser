"""
ML training and inference configuration.
"""

# Expanded universe for ~25k+ training rows (10y history per ticker)
DEFAULT_TRAIN_TICKERS = [
    # US large caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "V", "MA", "JNJ", "UNH", "WMT", "HD", "PG",
    "XOM", "KO", "PEP", "DIS", "NFLX", "AMD", "INTC", "CSCO",
    # India NSE
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
]

FORECAST_HORIZON_DAYS = 5
TRAIN_PERIOD = "10y"
TARGET_MIN_ROWS = 25_000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ROC-AUC is more reliable than F1 for imbalanced direction labels
PRIMARY_METRIC = "roc_auc"

MODEL_CANDIDATES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "lstm",
]

# LSTM hyperparameters
# Sequence length of 60 days (~quarterly window) gives the LSTM enough
# context to capture momentum, mean-reversion, and volatility cycles.
SEQUENCE_LENGTH = 60
LSTM_HIDDEN_SIZE = 128   # wider hidden state for richer 10y feature set
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 60         # more epochs; early stopping will regularise
LSTM_BATCH_SIZE = 256    # larger batch — stable gradients on big dataset
LSTM_LEARNING_RATE = 0.001
LSTM_PATIENCE = 8        # slightly more patience before early stop

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = "artifacts/models"
BEST_MODEL_PATH = "artifacts/models/best_model.joblib"
BEST_LSTM_PATH = "artifacts/models/best_model.pt"
LSTM_SCALER_PATH = "artifacts/models/lstm_scaler.joblib"
MODEL_METADATA_PATH = "artifacts/models/model_metadata.json"
FEATURE_IMPORTANCE_PATH = "artifacts/models/feature_importance.csv"
BENCHMARK_REPORT_PATH = "artifacts/models/benchmark_comparison.json"

MLFLOW_EXPERIMENT_TRAINING = "stock-ml-training"
MLFLOW_EXPERIMENT_INFERENCE = "stock-analysis-pipeline"

MIN_INFERENCE_PERIOD = "2y"
SHORT_PERIODS = {"1d", "5d", "1mo", "3mo"}

# Market benchmarks (from peer-reviewed literature & industry reports):
#   random_guess              : theoretical floor (efficient market null)
#   buy_and_hold_majority     : naive majority-class baseline
#   arima_directional_proxy   : classical time-series model directional proxy
#   xgboost_gbm_industry      : XGBoost on raw OHLCV (industry survey median)
#   lstm_daily_direction_lit  : LSTM on price/volume (literature median, Fischer & Krauss 2018)
#   transformer_finance_lit   : Temporal Fusion Transformer (Lim et al. 2021)
MARKET_BENCHMARKS = {
    "random_guess":              {"accuracy": 0.50, "f1": 0.50, "roc_auc": 0.50},
    "buy_and_hold_majority":     {"accuracy": 0.52, "f1": 0.55, "roc_auc": 0.52},
    "arima_directional_proxy":   {"accuracy": 0.51, "f1": 0.52, "roc_auc": 0.53},
    "xgboost_gbm_industry":      {"accuracy": 0.55, "f1": 0.56, "roc_auc": 0.57},
    "lstm_daily_direction_lit":  {"accuracy": 0.54, "f1": 0.58, "roc_auc": 0.56},
    "transformer_finance_lit":   {"accuracy": 0.56, "f1": 0.60, "roc_auc": 0.58},
}

"""
Canonical feature column list shared between training and inference.

Derived automatically from the constants in src/data/features.py so that
any change to LAG_PERIODS or ROLLING_WINDOWS is reflected everywhere
without manual edits.
"""

from src.data.features import LAG_PERIODS, ROLLING_WINDOWS

# ── Lag features ───────────────────────────────────────────────────────────────
_LAG_FEATURES: list[str] = []
for _lag in LAG_PERIODS:
    _LAG_FEATURES.extend([
        f"Return_lag_{_lag}",
        f"Close_lag_{_lag}",
        f"Volume_lag_{_lag}",
    ])

# ── Rolling statistics features ────────────────────────────────────────────────
_ROLLING_FEATURES: list[str] = []
for _w in ROLLING_WINDOWS:
    _ROLLING_FEATURES.extend([
        f"Return_mean_{_w}",
        f"Return_std_{_w}",
        f"Close_mean_{_w}",
        f"Volume_mean_{_w}",
    ])

# ── Full feature list (order matters for LSTM) ────────────────────────────────
FEATURE_COLUMNS: list[str] = [
    # Returns
    "Daily_Return",
    "Log_Return",
    # Lag features
    *_LAG_FEATURES,
    # Rolling stats (5/10/20/60d)
    *_ROLLING_FEATURES,
    # Momentum
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Momentum_60",
    # Volume dynamics
    "Volume_ratio_20",
    "Volume_ratio_5",
    # RSI
    "RSI_14",
    "RSI_28",
    # MACD
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    # Bollinger Bands
    "BB_Mid",
    "BB_Upper",
    "BB_Lower",
    "BB_Width",
    "BB_Pct",
    # ATR
    "ATR_14",
    "ATR_14_pct",
    # OBV
    "OBV",
    "OBV_ROC_10",
    # Price vs MAs (regime)
    "Close_vs_MA20",
    "Close_vs_MA60",
    # Range
    "HL_Range",
    "HL_Range_5ma",
]

TARGET_COLUMN = "target_up"

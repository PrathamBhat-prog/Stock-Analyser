"""
Time-series feature columns shared between training and inference.
Generated from lag/rolling structure in src/data/features.py.
"""

from src.data.features import LAG_PERIODS, ROLLING_WINDOWS

_LAG_FEATURES = []
for lag in LAG_PERIODS:
    _LAG_FEATURES.extend([
        f"Return_lag_{lag}",
        f"Close_lag_{lag}",
        f"Volume_lag_{lag}",
    ])

_ROLLING_FEATURES = []
for window in ROLLING_WINDOWS:
    _ROLLING_FEATURES.extend([
        f"Return_mean_{window}",
        f"Return_std_{window}",
        f"Close_mean_{window}",
        f"Volume_mean_{window}",
    ])

FEATURE_COLUMNS = [
    "Daily_Return",
    "Log_Return",
    *_LAG_FEATURES,
    *_ROLLING_FEATURES,
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Volume_ratio_20",
]

TARGET_COLUMN = "target_up"

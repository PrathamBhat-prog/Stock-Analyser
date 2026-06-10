import numpy as np
import pandas as pd

# Lag periods used for time-series feature engineering
LAG_PERIODS = [1, 2, 3, 5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20]


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-series ML features from OHLCV history.

    Includes:
    - Lagged returns and prices (autoregressive structure)
    - Rolling mean/std of returns (local regime)
    - Volume dynamics
    - Momentum over multiple horizons
    """
    df = df.copy()

    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    for lag in LAG_PERIODS:
        df[f"Return_lag_{lag}"] = df["Daily_Return"].shift(lag)
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"Return_mean_{window}"] = df["Daily_Return"].rolling(window=window).mean()
        df[f"Return_std_{window}"] = df["Daily_Return"].rolling(window=window).std()
        df[f"Close_mean_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"Volume_mean_{window}"] = df["Volume"].rolling(window=window).mean()

    df["Momentum_5"] = df["Close"].pct_change(periods=5)
    df["Momentum_10"] = df["Close"].pct_change(periods=10)
    df["Momentum_20"] = df["Close"].pct_change(periods=20)
    df["Volume_ratio_20"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

    return df


# Backward-compatible alias used by existing imports
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_series_features(df)

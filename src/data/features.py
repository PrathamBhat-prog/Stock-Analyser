"""
Lean feature engineering -- 12 high-signal, research-backed features.

Reduced from 60 to 12 features by eliminating:
  - Raw prices (scale-dependent, cannot generalise across tickers)
  - Redundant lags (lag_1 and lag_5 capture autocorrelation adequately)
  - Duplicate indicators (BB_Mid/Upper/Lower -> BB_Pct; ATR -> ATR_14_pct)
  - Cumulative non-stationary features (OBV -> OBV_ROC_10)
  - Multiple MACD lines (MACD_Hist is the single most informative)
  - Second RSI period (RSI_14 is the industry standard)

Final 12 features (one per signal category):
  Daily_Return   -- current price impulse
  Return_lag_1   -- 1-day autocorrelation
  Return_lag_5   -- weekly memory effect
  Return_std_20  -- 20-day realised volatility (risk)
  Momentum_20    -- 20-day trend strength
  RSI_14         -- overbought / oversold oscillator
  MACD_Hist      -- MACD histogram (crossover strength)
  BB_Pct         -- Bollinger Band position (normalised)
  ATR_14_pct     -- normalised true range (intraday volatility)
  OBV_ROC_10     -- 10-day volume momentum
  Close_vs_MA20  -- price vs 20d MA (z-score, trend regime)
  Volume_ratio_5 -- 5-day volume spike detection

All features are: backward-looking, normalised, stationary, cross-ticker.
"""

import numpy as np
import pandas as pd


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 12 selected features in-place and return the DataFrame.
    Requires columns: Open, High, Low, Close, Volume, Date.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # 1. Daily return
    df["Daily_Return"] = close.pct_change()

    # 2-3. Lagged returns (autocorrelation)
    df["Return_lag_1"] = df["Daily_Return"].shift(1)
    df["Return_lag_5"] = df["Daily_Return"].shift(5)

    # 4. 20-day realised volatility
    df["Return_std_20"] = df["Daily_Return"].rolling(20).std()

    # 5. 20-day price momentum
    df["Momentum_20"] = close.pct_change(20)

    # 6. RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 7. MACD Histogram (most informative single MACD signal)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - sig

    # 8. Bollinger Band % position (normalised 0-1)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_Pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # 9. ATR-14 as % of close (normalised volatility)
    prev_c = close.shift(1)
    tr     = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    atr14          = tr.ewm(com=13, min_periods=14).mean()
    df["ATR_14_pct"] = atr14 / close.replace(0, np.nan)

    # 10. OBV 10-day rate of change (stationary volume momentum)
    sign = np.sign(close.diff()).fillna(0)
    obv  = (sign * vol).cumsum()
    df["OBV_ROC_10"] = obv.pct_change(10)

    # 11. Price position relative to 20d MA (z-score)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std().replace(0, np.nan)
    df["Close_vs_MA20"] = (close - ma20) / std20

    # 12. 5-day volume ratio (spike detection)
    df["Volume_ratio_5"] = vol / vol.rolling(5).mean().replace(0, np.nan)

    # Label: 1 if price higher in 5 trading days
    df["target_up"] = (close.shift(-5) > close).astype(int)

    return df


# Canonical list -- import this everywhere
FEATURE_COLUMNS = [
    "Daily_Return",
    "Return_lag_1",
    "Return_lag_5",
    "Return_std_20",
    "Momentum_20",
    "RSI_14",
    "MACD_Hist",
    "BB_Pct",
    "ATR_14_pct",
    "OBV_ROC_10",
    "Close_vs_MA20",
    "Volume_ratio_5",
]

TARGET_COLUMN = "target_up"

"""
Production-grade feature engineering for stock direction prediction.

Feature categories (all derived from OHLCV — no lookahead leakage):
  1. Returns & log-returns         — autoregressive price structure
  2. Lagged returns / prices / vol  — memory effects (lags 1-20 d)
  3. Rolling statistics              — local regime (5/10/20/60 d)
  4. Momentum                        — trend-following signals
  5. RSI (Relative Strength Index)   — mean-reversion signal
  6. MACD + signal line              — trend-change detection
  7. Bollinger Bands                 — volatility breakout
  8. ATR (Average True Range)        — realised volatility proxy
  9. OBV (On-Balance Volume)         — volume-price confluence
  10. Volatility ratios               — regime change detection
"""

import numpy as np
import pandas as pd

# ── Lag / rolling window constants ────────────────────────────────────────────
LAG_PERIODS      = [1, 2, 3, 5, 10, 20]
ROLLING_WINDOWS  = [5, 10, 20, 60]       # 60-day (~quarterly) window added


# ── Helper: EMA ───────────────────────────────────────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── RSI ───────────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ──────────────────────────────────────────────────────────────────────
def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Bollinger Bands ───────────────────────────────────────────────────────────
def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid    = close.rolling(window).mean()
    std    = close.rolling(window).std()
    upper  = mid + num_std * std
    lower  = mid - num_std * std
    bw     = (upper - lower) / mid.replace(0, np.nan)   # bandwidth normalised
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, bw, pct_b


# ── ATR ───────────────────────────────────────────────────────────────────────
def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ── OBV ───────────────────────────────────────────────────────────────────────
def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


# ── Main feature builder ──────────────────────────────────────────────────────

def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build production-grade ML features from OHLCV history.

    All features are strictly backward-looking — no future data leaks in.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume (from yfinance).

    Returns
    -------
    pd.DataFrame with all original columns plus engineered features.
    """
    df = df.copy()

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── 1. Returns ────────────────────────────────────────────────────────────
    df["Daily_Return"] = close.pct_change()
    df["Log_Return"]   = np.log(close / close.shift(1))

    # ── 2. Lagged features ────────────────────────────────────────────────────
    for lag in LAG_PERIODS:
        df[f"Return_lag_{lag}"]  = df["Daily_Return"].shift(lag)
        df[f"Close_lag_{lag}"]   = close.shift(lag)
        df[f"Volume_lag_{lag}"]  = volume.shift(lag)

    # ── 3. Rolling statistics ─────────────────────────────────────────────────
    for window in ROLLING_WINDOWS:
        df[f"Return_mean_{window}"] = df["Daily_Return"].rolling(window).mean()
        df[f"Return_std_{window}"]  = df["Daily_Return"].rolling(window).std()
        df[f"Close_mean_{window}"]  = close.rolling(window).mean()
        df[f"Volume_mean_{window}"] = volume.rolling(window).mean()

    # ── 4. Momentum ───────────────────────────────────────────────────────────
    df["Momentum_5"]  = close.pct_change(periods=5)
    df["Momentum_10"] = close.pct_change(periods=10)
    df["Momentum_20"] = close.pct_change(periods=20)
    df["Momentum_60"] = close.pct_change(periods=60)

    # ── 5. Volume dynamics ────────────────────────────────────────────────────
    df["Volume_ratio_20"] = volume / volume.rolling(window=20).mean()
    df["Volume_ratio_5"]  = volume / volume.rolling(window=5).mean()

    # ── 6. RSI ────────────────────────────────────────────────────────────────
    df["RSI_14"] = _rsi(close, period=14)
    df["RSI_28"] = _rsi(close, period=28)

    # ── 7. MACD ───────────────────────────────────────────────────────────────
    macd_line, signal_line, macd_hist = _macd(close)
    df["MACD"]         = macd_line
    df["MACD_Signal"]  = signal_line
    df["MACD_Hist"]    = macd_hist

    # ── 8. Bollinger Bands ────────────────────────────────────────────────────
    bb_mid, bb_upper, bb_lower, bb_bw, bb_pct = _bollinger(close)
    df["BB_Mid"]   = bb_mid
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower
    df["BB_Width"] = bb_bw     # normalised bandwidth (vol measure)
    df["BB_Pct"]   = bb_pct   # position within bands (0–1)

    # ── 9. ATR ────────────────────────────────────────────────────────────────
    df["ATR_14"] = _atr(high, low, close, period=14)
    # Normalise by close to make it price-scale independent
    df["ATR_14_pct"] = df["ATR_14"] / close.replace(0, np.nan)

    # ── 10. OBV ───────────────────────────────────────────────────────────────
    df["OBV"] = _obv(close, volume)
    # OBV momentum: rate-of-change over 10 days
    df["OBV_ROC_10"] = df["OBV"].pct_change(periods=10)

    # ── 11. Price vs moving averages (regime) ─────────────────────────────────
    df["Close_vs_MA20"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    df["Close_vs_MA60"] = (close - close.rolling(60).mean()) / close.rolling(60).std().replace(0, np.nan)

    # ── 12. High-Low range ────────────────────────────────────────────────────
    df["HL_Range"]     = (high - low) / close.replace(0, np.nan)
    df["HL_Range_5ma"] = df["HL_Range"].rolling(5).mean()

    return df


# Backward-compatible alias used by existing imports
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_series_features(df)

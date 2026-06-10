"""
Trend analysis agent — ARIMA-inspired, pure-pandas implementation.

Detects market regime using the same OHLCV features already computed.
No statsmodels or external TS library required — uses rolling statistics
to approximate trend + momentum + volatility regime classification.

This makes the system work for ANY ticker, not just trained ones,
because it operates on feature patterns (RSI, MACD, MA slopes, etc.)
rather than ticker-specific learned parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ma_slope(series: pd.Series, window: int) -> float:
    """Slope of the last `window` values (normalised by mean price)."""
    tail = series.dropna().tail(window)
    if len(tail) < 2:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    slope = np.polyfit(x, tail.values, 1)[0]
    mean_val = tail.mean()
    return float(slope / max(abs(mean_val), 1e-9))


def analyze_trend(df: pd.DataFrame) -> dict:
    """
    ARIMA-inspired trend + momentum + volatility regime analysis.

    Uses only backward-looking features already in the dataframe.
    Works for ANY ticker — no ticker-specific training required.

    Returns
    -------
    dict with keys:
        trend_label     : 'Strong Uptrend' | 'Uptrend' | 'Sideways' |
                          'Downtrend' | 'Strong Downtrend'
        trend_score     : float in [-1, 1] (positive = bullish, negative = bearish)
        momentum_label  : 'Overbought' | 'Bullish' | 'Neutral' |
                          'Bearish' | 'Oversold'
        volatility_label: 'High Volatility' | 'Normal' | 'Low Volatility'
        summary         : plain-English summary for laymen
        signals         : list of readable signal strings
        current_price   : float
        price_change_pct: float (vs 20 trading days ago)
    """
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 30:
        return {
            "trend_label": "Insufficient Data",
            "trend_score": 0.0,
            "momentum_label": "Neutral",
            "volatility_label": "Normal",
            "summary": "Not enough price history to detect a trend.",
            "signals": [],
            "current_price": float(df["Close"].iloc[-1]) if len(df) else 0.0,
            "price_change_pct": 0.0,
        }

    close = df["Close"]
    current_price = float(close.iloc[-1])

    # ── Price change vs 20 trading days ago ───────────────────────────────────
    price_20d_ago = float(close.iloc[-min(21, len(close))])
    price_change_pct = (current_price - price_20d_ago) / max(abs(price_20d_ago), 1e-9) * 100

    signals: list[str] = []
    score = 0.0   # accumulate in [-1, 1]

    # ── 1. Moving average alignment (trend) ───────────────────────────────────
    ma20 = close.rolling(20).mean().iloc[-1] if "Close_mean_20" not in df.columns else df["Close_mean_20"].iloc[-1]
    ma60_col = df["Close_mean_60"].iloc[-1] if "Close_mean_60" in df.columns else close.rolling(60).mean().iloc[-1]

    above_ma20 = current_price > ma20
    above_ma60 = current_price > ma60_col
    ma20_slope = _ma_slope(close.rolling(20).mean().dropna(), window=10)
    ma60_slope = _ma_slope(close.rolling(60).mean().dropna(), window=10)

    if above_ma20 and above_ma60:
        score += 0.3
        signals.append("Price is above both short-term (20d) and long-term (60d) moving averages — bullish alignment")
    elif not above_ma20 and not above_ma60:
        score -= 0.3
        signals.append("Price is below both short-term (20d) and long-term (60d) moving averages — bearish alignment")
    else:
        signals.append("Price is between short- and long-term moving averages — mixed signal")

    if ma20_slope > 0.0002:
        score += 0.15
        signals.append("20-day average is rising — short-term uptrend")
    elif ma20_slope < -0.0002:
        score -= 0.15
        signals.append("20-day average is falling — short-term downtrend")

    # ── 2. RSI (momentum) ─────────────────────────────────────────────────────
    rsi14 = None
    if "RSI_14" in df.columns:
        rsi14 = float(df["RSI_14"].iloc[-1])
        if rsi14 > 70:
            score -= 0.1   # overbought — mean reversion risk
            signals.append(f"RSI = {rsi14:.0f} — overbought (stock may be due for a pullback)")
        elif rsi14 < 30:
            score += 0.1   # oversold — bounce potential
            signals.append(f"RSI = {rsi14:.0f} — oversold (stock may bounce back up)")
        elif rsi14 > 55:
            score += 0.08
            signals.append(f"RSI = {rsi14:.0f} — momentum is bullish")
        elif rsi14 < 45:
            score -= 0.08
            signals.append(f"RSI = {rsi14:.0f} — momentum is bearish")
        else:
            signals.append(f"RSI = {rsi14:.0f} — neutral momentum")

    # ── 3. MACD ────────────────────────────────────────────────────────────────
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd_val   = float(df["MACD"].iloc[-1])
        macd_sig   = float(df["MACD_Signal"].iloc[-1])
        macd_hist  = float(df["MACD_Hist"].iloc[-1]) if "MACD_Hist" in df.columns else macd_val - macd_sig
        prev_hist  = float(df["MACD_Hist"].iloc[-2]) if "MACD_Hist" in df.columns and len(df) > 1 else macd_hist

        if macd_val > macd_sig and macd_hist > 0:
            score += 0.15
            signals.append("MACD is above its signal line — trend is gaining upward momentum")
        elif macd_val < macd_sig and macd_hist < 0:
            score -= 0.15
            signals.append("MACD is below its signal line — trend is losing momentum / bearish")
        # Crossover detection
        if prev_hist < 0 and macd_hist > 0:
            score += 0.1
            signals.append("MACD just crossed above signal line — potential bullish reversal")
        elif prev_hist > 0 and macd_hist < 0:
            score -= 0.1
            signals.append("MACD just crossed below signal line — potential bearish reversal")

    # ── 4. Bollinger Band position ─────────────────────────────────────────────
    if "BB_Pct" in df.columns and "BB_Width" in df.columns:
        bb_pct   = float(df["BB_Pct"].iloc[-1])
        bb_width = float(df["BB_Width"].iloc[-1])

        if bb_pct > 0.9:
            score -= 0.08
            signals.append("Price is near the upper Bollinger Band — potential short-term pullback")
        elif bb_pct < 0.1:
            score += 0.08
            signals.append("Price is near the lower Bollinger Band — potential short-term bounce")

    # ── 5. Momentum (price change) ────────────────────────────────────────────
    if "Momentum_20" in df.columns:
        mom20 = float(df["Momentum_20"].iloc[-1])
        if mom20 > 0.05:
            score += 0.1
            signals.append(f"20-day price momentum is +{mom20:.1%} — strong upward move")
        elif mom20 < -0.05:
            score -= 0.1
            signals.append(f"20-day price momentum is {mom20:.1%} — strong downward move")

    # ── 6. Volume confirmation ────────────────────────────────────────────────
    if "Volume_ratio_20" in df.columns:
        vol_ratio = float(df["Volume_ratio_20"].iloc[-1])
        if vol_ratio > 1.5 and score > 0:
            score += 0.05
            signals.append(f"Volume is {vol_ratio:.1f}x the 20-day average — high participation confirms the uptrend")
        elif vol_ratio > 1.5 and score < 0:
            score -= 0.05
            signals.append(f"Volume is {vol_ratio:.1f}x the 20-day average — high participation confirms the downtrend")

    # ── Clamp score ────────────────────────────────────────────────────────────
    score = max(-1.0, min(1.0, score))

    # ── Volatility regime ─────────────────────────────────────────────────────
    volatility_label = "Normal"
    if "ATR_14_pct" in df.columns:
        atr_pct = float(df["ATR_14_pct"].iloc[-1])
        if atr_pct > 0.025:
            volatility_label = "High Volatility"
        elif atr_pct < 0.008:
            volatility_label = "Low Volatility"

    # ── Trend label ────────────────────────────────────────────────────────────
    if score >= 0.4:
        trend_label = "Strong Uptrend"
    elif score >= 0.15:
        trend_label = "Uptrend"
    elif score <= -0.4:
        trend_label = "Strong Downtrend"
    elif score <= -0.15:
        trend_label = "Downtrend"
    else:
        trend_label = "Sideways / Consolidating"

    # ── Momentum label ─────────────────────────────────────────────────────────
    if rsi14 is not None:
        if rsi14 > 70:
            momentum_label = "Overbought"
        elif rsi14 < 30:
            momentum_label = "Oversold"
        elif rsi14 > 55:
            momentum_label = "Bullish"
        elif rsi14 < 45:
            momentum_label = "Bearish"
        else:
            momentum_label = "Neutral"
    else:
        momentum_label = "Neutral"

    # ── Plain-English summary ──────────────────────────────────────────────────
    direction_word = "up" if price_change_pct > 0 else "down"
    summary = (
        f"Over the past month, this stock moved {direction_word} by "
        f"{abs(price_change_pct):.1f}%. "
        f"The market trend is currently showing a {trend_label.lower()}. "
    )
    if volatility_label == "High Volatility":
        summary += "The stock is moving with higher-than-usual swings — higher risk. "
    elif volatility_label == "Low Volatility":
        summary += "The stock is relatively calm with small daily moves — lower risk. "

    return {
        "trend_label":      trend_label,
        "trend_score":      round(score, 3),
        "momentum_label":   momentum_label,
        "volatility_label": volatility_label,
        "summary":          summary,
        "signals":          signals[:6],   # top-6 signals for display
        "current_price":    round(current_price, 2),
        "price_change_pct": round(price_change_pct, 2),
    }

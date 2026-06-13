# -*- coding: utf-8 -*-
"""
Sniper v5 Production Predictor -- CatBoostClassifier
Features: momentum_20d, dist_52w_high, rolling_sentiment_20d,
          vol_ratio_5d, VIX, sent_lag_1/3/5, sent_vix_interaction, vix_velocity
Threshold: 0.52 (optimised for precision)
"""

from __future__ import annotations
import logging
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.news_fetcher import fetch_headlines
from src.data.sentiment_scorer import score_headlines_vader
from src.data.sentiment_cache import build_sentiment_features, save_sentiment

logger = logging.getLogger(__name__)

SNIPER_MODEL_PATH  = os.path.join("artifacts", "models", "trading_model_sniper_v5.pkl")
CONF_THRESHOLD     = 0.52   # Sniper v5 sweet spot (60.46% precision)

FEATURE_COLS = [
    "momentum_20d", "dist_52w_high", "rolling_sentiment_20d",
    "vol_ratio_5d", "VIX", "sent_lag_1", "sent_lag_3",
    "sent_lag_5", "sent_vix_interaction", "vix_velocity",
]


def _fetch_vix() -> tuple[float, float]:
    """Return (current VIX level, VIX 5-day velocity)."""
    try:
        vix_df = yf.Ticker("^VIX").history(period="1mo", auto_adjust=True)
        vix    = float(vix_df["Close"].iloc[-1])
        vel    = float(vix_df["Close"].pct_change(5).iloc[-1])
        return vix, vel
    except Exception as exc:
        logger.warning(f"VIX fetch failed: {exc}. Using neutral defaults.")
        return 20.0, 0.0


def _build_ohlcv_features(df: pd.DataFrame) -> dict:
    """Compute the 4 OHLCV-derived features from price history."""
    close = df["Close"]
    vol   = df["Volume"]

    momentum_20d = float(close.pct_change(20).iloc[-1])

    high_52w       = float(close.rolling(252, min_periods=1).max().iloc[-1])
    dist_52w_high  = float((close.iloc[-1] - high_52w) / high_52w) if high_52w else 0.0

    vol_ratio_5d = float(
        (vol.iloc[-1] / vol.rolling(5).mean().iloc[-1])
        if vol.rolling(5).mean().iloc[-1] > 0 else 1.0
    )

    return {
        "momentum_20d":  momentum_20d,
        "dist_52w_high": dist_52w_high,
        "vol_ratio_5d":  vol_ratio_5d,
    }


class SniperPredictor:
    """
    Loads trading_model_sniper_v5.pkl (CatBoost) and runs inference
    with live GDELT sentiment + VIX macro features.
    """

    def __init__(self):
        self._model = None

    @property
    def is_available(self) -> bool:
        return os.path.exists(SNIPER_MODEL_PATH)

    def _load(self):
        if self._model is None:
            if not self.is_available:
                raise FileNotFoundError(
                    f"Sniper v5 model not found at {SNIPER_MODEL_PATH}. "
                    "Download trading_model_sniper_v5.pkl from Colab and place it there."
                )
            with open(SNIPER_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            logger.info("Sniper v5 CatBoost model loaded.")

    def predict(
        self,
        ticker:       str,
        df:           pd.DataFrame,
        company_name: str = "",
    ) -> dict:
        """
        Full inference with live sentiment + VIX.
        Returns dict with probability_up, signal, confidence, threshold.
        """
        self._load()

        # 1. Live VIX
        vix, vix_vel = _fetch_vix()

        # 2. Live sentiment (GDELT -> yfinance fallback -> cache)
        headlines = fetch_headlines(ticker, company_name)
        sent_score = score_headlines_vader(headlines)
        today = date.today()
        save_sentiment(ticker, today, sent_score, len(headlines))

        # 3. Build sentiment lag features from cache
        sent_feats = build_sentiment_features(ticker, today, vix)
        # Override rolling_sentiment_20d with today's live score
        sent_feats["rolling_sentiment_20d"] = sent_score

        # 4. OHLCV features
        ohlcv_feats = _build_ohlcv_features(df)

        # 5. Assemble feature vector (must match FEATURE_COLS order exactly)
        row = {
            "momentum_20d":          ohlcv_feats["momentum_20d"],
            "dist_52w_high":         ohlcv_feats["dist_52w_high"],
            "rolling_sentiment_20d": sent_feats["rolling_sentiment_20d"],
            "vol_ratio_5d":          ohlcv_feats["vol_ratio_5d"],
            "VIX":                   vix,
            "sent_lag_1":            sent_feats.get("sent_lag_1", 0.0),
            "sent_lag_3":            sent_feats.get("sent_lag_3", 0.0),
            "sent_lag_5":            sent_feats.get("sent_lag_5", 0.0),
            "sent_vix_interaction":  sent_score * vix,
            "vix_velocity":          vix_vel,
        }
        X = pd.DataFrame([row])[FEATURE_COLS]

        # 6. Predict
        proba_up   = float(self._model.predict_proba(X)[0, 1])
        above_conf = proba_up >= CONF_THRESHOLD

        if proba_up >= CONF_THRESHOLD:
            signal = "BULLISH"
        elif proba_up <= (1 - CONF_THRESHOLD):
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        logger.info(
            f"Sniper v5 [{ticker}]: prob_up={proba_up:.3f} "
            f"signal={signal} sent={sent_score:+.3f} VIX={vix:.1f}"
        )

        return {
            "signal":               signal,
            "probability_up":       round(proba_up, 4),
            "confidence":           round(max(proba_up, 1 - proba_up), 4),
            "above_threshold":      above_conf,
            "threshold":            CONF_THRESHOLD,
            "model_name":           "CatBoost Sniper v5",
            "forecast_horizon_days": 20,
            "sentiment_score":      round(sent_score, 4),
            "vix":                  round(vix, 2),
            "vix_velocity":         round(vix_vel, 4),
            "headline_count":       len(headlines),
            "available":            True,
            "reason": (
                f"CatBoost Sniper v5 gives {proba_up:.1%} probability of price "
                f"being higher in 20 trading days. "
                f"News sentiment: {sent_score:+.3f}. "
                f"VIX: {vix:.1f} ({'fearful' if vix > 25 else 'calm'} market)."
            ),
        }

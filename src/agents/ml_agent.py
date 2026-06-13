# -*- coding: utf-8 -*-
"""
ML Prediction Agent -- Sniper v5 (CatBoost) primary, sklearn fallback.
"""
import logging
import pandas as pd

from src.models.sniper_predictor import SniperPredictor
from src.models.predictor import StockDirectionPredictor

logger = logging.getLogger(__name__)
_NEUTRAL = {
    "signal":               "NEUTRAL",
    "confidence":           0.5,
    "probability_up":       0.5,
    "model_name":           None,
    "forecast_horizon_days": 20,
    "available":            False,
    "reason":               "No trained model found. Run `python train.py` to train.",
}


class MLPredictionAgent:
    """
    Tries Sniper v5 (CatBoost + sentiment) first.
    Falls back to the original sklearn model if Sniper v5 is unavailable.
    """

    def __init__(self):
        self._sniper   = SniperPredictor()
        self._fallback = StockDirectionPredictor()

    def analyze(
        self,
        df:           pd.DataFrame,
        ticker:       str = "",
        company_name: str = "",
    ) -> dict:

        # --- Primary: Sniper v5 ---
        if self._sniper.is_available:
            try:
                result = self._sniper.predict(ticker, df, company_name)
                logger.info(f"Sniper v5 signal: {result['signal']} ({result['probability_up']:.3f})")
                return result
            except Exception as exc:
                logger.warning(f"Sniper v5 failed, falling back: {exc}")

        # --- Fallback: original sklearn model ---
        if self._fallback.is_available:
            try:
                result = self._fallback.predict_latest(df)
                result["available"] = True
                return result
            except Exception as exc:
                logger.warning(f"Fallback sklearn model also failed: {exc}")

        return _NEUTRAL

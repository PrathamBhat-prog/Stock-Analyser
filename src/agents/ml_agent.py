import pandas as pd

from src.models.predictor import StockDirectionPredictor


class MLPredictionAgent:
    """
    ML-based direction agent using a trained sklearn model.
    Falls back gracefully if no model has been trained yet.
    """

    def __init__(self):
        self.predictor = StockDirectionPredictor()

    def analyze(self, df: pd.DataFrame) -> dict:
        if not self.predictor.is_available:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "probability_up": 0.5,
                "model_name": None,
                "forecast_horizon_days": None,
                "reason": (
                    "No trained ML model found. Run `python train.py` to train models."
                ),
                "available": False,
            }

        try:
            result = self.predictor.predict_latest(df)
            result["available"] = True
            return result
        except ValueError as exc:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "probability_up": 0.5,
                "model_name": None,
                "forecast_horizon_days": None,
                "reason": str(exc),
                "available": False,
            }

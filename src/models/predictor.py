import json
import os

import joblib
import pandas as pd

from src.config.ml_config import (
    BEST_LSTM_PATH,
    BEST_MODEL_PATH,
    LSTM_SCALER_PATH,
    MODEL_METADATA_PATH,
)
from src.models.feature_columns import FEATURE_COLUMNS
from src.models.lstm_model import LSTMTrainer


class StockDirectionPredictor:
    """Load trained sklearn or LSTM model and produce predictions."""

    def __init__(self, metadata_path: str = MODEL_METADATA_PATH):
        self.metadata_path = metadata_path
        self._model = None
        self._lstm: LSTMTrainer | None = None
        self._metadata = None

    @property
    def is_available(self) -> bool:
        return os.path.exists(self.metadata_path)

    def load(self) -> None:
        if not self.is_available:
            raise FileNotFoundError(
                "Trained model not found. Run `python train.py` first."
            )

        with open(self.metadata_path, encoding="utf-8") as f:
            self._metadata = json.load(f)

        model_type = self._metadata.get("model_type", "sklearn")
        if model_type == "lstm":
            if not os.path.exists(BEST_LSTM_PATH):
                raise FileNotFoundError(f"LSTM weights not found at {BEST_LSTM_PATH}")
            self._lstm = LSTMTrainer.load(BEST_LSTM_PATH, LSTM_SCALER_PATH)
        else:
            model_path = self._metadata.get("model_path", BEST_MODEL_PATH)
            self._model = joblib.load(model_path)

    def _ensure_loaded(self) -> None:
        if self._metadata is None:
            self.load()

    @property
    def metadata(self) -> dict:
        self._ensure_loaded()
        return self._metadata

    def predict_latest(self, df: pd.DataFrame) -> dict:
        self._ensure_loaded()

        model_type = self._metadata.get("model_type", "sklearn")
        if model_type == "lstm":
            proba_up = self._lstm.predict_proba_latest(df)
            pred_class = int(proba_up >= 0.5)
        else:
            feature_df = df[FEATURE_COLUMNS].dropna()
            if feature_df.empty:
                raise ValueError("Not enough feature history for ML prediction.")
            latest = feature_df.iloc[[-1]]
            proba_up = float(self._model.predict_proba(latest)[0, 1])
            pred_class = int(self._model.predict(latest)[0])

        if proba_up >= 0.55:
            signal = "BULLISH"
        elif proba_up <= 0.45:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        horizon = self._metadata.get("forecast_horizon_days", 5)
        model_name = self._metadata.get("best_model_name", "unknown")

        return {
            "signal": signal,
            "confidence": round(max(proba_up, 1 - proba_up), 2),
            "probability_up": round(proba_up, 4),
            "predicted_class": pred_class,
            "model_name": model_name,
            "model_type": model_type,
            "forecast_horizon_days": horizon,
            "reason": (
                f"{model_type.upper()} model ({model_name}) estimates {proba_up:.1%} "
                f"probability that price will be higher in {horizon} trading days."
            ),
        }

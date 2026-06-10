"""
Full ML inference pipeline for any stock ticker.

Generalises to ANY company — the ML model was trained on FEATURE PATTERNS
(RSI, MACD, momentum, Bollinger, etc.), not on specific ticker identities.
Any company whose OHLCV data can be fetched from yfinance will work.
"""

import json
import os

import mlflow
import yfinance as yf

from src.config.ml_config import (
    MIN_INFERENCE_PERIOD,
    MLFLOW_EXPERIMENT_INFERENCE,
    SHORT_PERIODS,
)
from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_time_series_features

from src.agents.ml_agent import MLPredictionAgent
from src.agents.decision_agent import MLDecisionAgent
from src.agents.trend_agent import analyze_trend


def _get_company_info(ticker: str) -> dict:
    """Fetch company name and sector from yfinance (best-effort)."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector":       info.get("sector", "Unknown"),
            "industry":     info.get("industry", "Unknown"),
            "currency":     info.get("currency", "USD"),
            "exchange":     info.get("exchange", ""),
        }
    except Exception:
        return {
            "company_name": ticker.upper(),
            "sector":       "Unknown",
            "industry":     "Unknown",
            "currency":     "USD",
            "exchange":     "",
        }


class StockAnalysisPipeline:
    """
    ML-only time-series inference pipeline.

    Works for ANY publicly listed company on any yfinance-supported exchange:
    - US stocks (AAPL, MSFT, GOOGL …)
    - Indian NSE stocks (RELIANCE.NS, TCS.NS …)
    - European stocks (ASML.AS, SAP.DE …)
    - ETFs, indices, crypto (BTC-USD …)

    The ML model generalises because it operates on technical indicator patterns
    (RSI, MACD, Bollinger, momentum), not on company-specific learned weights.
    """

    def __init__(self):
        project_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mlruns_path    = os.path.join(project_root, "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_INFERENCE)

        self.ml_agent       = MLPredictionAgent()
        self.decision_agent = MLDecisionAgent()

    @staticmethod
    def _effective_period(period: str) -> str:
        """Ensure enough history for lag/rolling/MACD/RSI features (need ~100d min)."""
        if period in SHORT_PERIODS:
            return MIN_INFERENCE_PERIOD
        return period

    def run(self, ticker: str, period: str = "2y") -> dict:
        with mlflow.start_run():
            fetch_period = self._effective_period(period)
            mlflow.log_param("ticker",       ticker)
            mlflow.log_param("period",       period)
            mlflow.log_param("fetch_period", fetch_period)

            # ── 1. Fetch & engineer features ──────────────────────────────────
            df           = fetch_stock_data(ticker=ticker, period=fetch_period)
            df           = validate_stock_data(df)
            df           = add_time_series_features(df)

            # ── 2. Company metadata ────────────────────────────────────────────
            company_info = _get_company_info(ticker)

            # ── 3. ML prediction (generalises to any ticker) ───────────────────
            ml_result    = self.ml_agent.analyze(df)

            # ── 4. Trend analysis (ARIMA-inspired, pure pandas) ────────────────
            trend_result = analyze_trend(df)

            # ── 5. Final decision ──────────────────────────────────────────────
            decision     = self.decision_agent.decide(ml_result)

            # ── 6. MLflow logging ──────────────────────────────────────────────
            mlflow.log_param("ml_signal",      ml_result.get("signal",         "N/A"))
            mlflow.log_param("ml_model",       ml_result.get("model_name",     "none"))
            mlflow.log_param("final_decision", decision["final_decision"])
            mlflow.log_param("trend_label",    trend_result["trend_label"])
            mlflow.log_metric("confidence",    decision["confidence"])
            if ml_result.get("available"):
                mlflow.log_metric("ml_probability_up", ml_result["probability_up"])
            mlflow.log_metric("trend_score",   trend_result["trend_score"])

            # ── 7. Save decision artifact ──────────────────────────────────────
            artifact_path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                "decision.json",
            )
            full_result = {**decision, "company": company_info, "trend": trend_result}
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(full_result, f, indent=2)
            mlflow.log_artifact(artifact_path)

        return full_result

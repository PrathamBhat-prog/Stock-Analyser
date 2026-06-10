"""
Full ML inference pipeline for any stock ticker, any investment horizon.

Generalises to ANY company: the ML model operates on technical indicator
patterns (RSI, MACD, momentum, Bollinger, etc.), not on ticker identities.
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
from src.agents.decision_agent import MLDecisionAgent, HORIZONS, DEFAULT_HORIZON
from src.agents.trend_agent import analyze_trend


def _get_company_info(ticker: str) -> dict:
    """Fetch company name and sector from yfinance (best-effort)."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector":       info.get("sector",   "Unknown"),
            "industry":     info.get("industry", "Unknown"),
            "currency":     info.get("currency", "USD"),
            "exchange":     info.get("exchange", ""),
            "market_cap":   info.get("marketCap"),
            "pe_ratio":     info.get("trailingPE"),
            "52w_high":     info.get("fiftyTwoWeekHigh"),
            "52w_low":      info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {
            "company_name": ticker.upper(),
            "sector":       "Unknown",
            "industry":     "Unknown",
            "currency":     "USD",
            "exchange":     "",
            "market_cap":   None,
            "pe_ratio":     None,
            "52w_high":     None,
            "52w_low":      None,
        }


class StockAnalysisPipeline:
    """
    End-to-end inference pipeline supporting any ticker and any investment horizon.

    Supported tickers (any yfinance-supported exchange):
      US:     AAPL, MSFT, TSLA, GOOGL, NVDA ...
      India:  RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS ...
      EU:     ASML.AS, SAP.DE, HSBA.L, BP.L ...
      Asia:   005930.KS (Samsung), 9988.HK (Alibaba) ...
      Crypto: BTC-USD, ETH-USD ...
      ETFs:   SPY, QQQ, NIFTY50.NS ...

    Supported horizons (no retraining needed):
      5d   - 1 week     - ML model primary (80% weight)
      21d  - 1 month    - ML + Trend equal (50/50)
      63d  - 3 months   - Trend dominant  (70%)
      126d - 6 months   - Trend dominant  (85%)
      252d - 1 year     - Trend dominant  (90%)
    """

    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mlruns_path  = os.path.join(project_root, "mlruns")
        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_INFERENCE)

        self.ml_agent       = MLPredictionAgent()
        self.decision_agent = MLDecisionAgent()

    @staticmethod
    def _effective_period(period: str) -> str:
        if period in SHORT_PERIODS:
            return MIN_INFERENCE_PERIOD
        return period

    def run(
        self,
        ticker:      str,
        period:      str = "2y",
        horizon_key: str = DEFAULT_HORIZON,
    ) -> dict:
        with mlflow.start_run():
            fetch_period = self._effective_period(period)
            mlflow.log_param("ticker",       ticker)
            mlflow.log_param("period",       period)
            mlflow.log_param("fetch_period", fetch_period)
            mlflow.log_param("horizon",      horizon_key)

            # 1. Fetch OHLCV + engineer features
            df           = fetch_stock_data(ticker=ticker, period=fetch_period)
            df           = validate_stock_data(df)
            df           = add_time_series_features(df)

            # 2. Company metadata
            company_info = _get_company_info(ticker)

            # 3. ML prediction (always runs for its short-term signal)
            ml_result    = self.ml_agent.analyze(df)

            # 4. Trend analysis (ARIMA-inspired, pure pandas)
            trend_result = analyze_trend(df)

            # 5. Multi-horizon decision
            decision     = self.decision_agent.decide(
                ml_result    = ml_result,
                trend_result = trend_result,
                horizon_key  = horizon_key,
            )

            # 6. MLflow logging
            mlflow.log_param("final_decision", decision["final_decision"])
            mlflow.log_param("trend_label",    trend_result["trend_label"])
            mlflow.log_metric("confidence",    decision["confidence"])
            mlflow.log_metric("composite",     decision["composite_score"])
            if ml_result.get("available"):
                mlflow.log_metric("ml_probability_up", ml_result["probability_up"])
            mlflow.log_metric("trend_score",   trend_result["trend_score"])

            # 7. Save artifact
            artifact_path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                "decision.json",
            )
            full_result = {
                **decision,
                "company": company_info,
                "trend":   trend_result,
            }
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(full_result, f, indent=2)
            mlflow.log_artifact(artifact_path)

        return full_result

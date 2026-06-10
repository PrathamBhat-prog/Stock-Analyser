import json
import os
import mlflow

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


class StockAnalysisPipeline:
    """
    ML-only time-series inference pipeline with MLflow logging.
    """

    def __init__(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mlruns_path = os.path.join(project_root, "mlruns")

        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_INFERENCE)

        self.ml_agent = MLPredictionAgent()
        self.decision_agent = MLDecisionAgent()

    @staticmethod
    def _effective_period(period: str) -> str:
        """Ensure enough history for lag/rolling features."""
        if period in SHORT_PERIODS:
            return MIN_INFERENCE_PERIOD
        return period

    def run(self, ticker: str, period: str = "1y") -> dict:
        with mlflow.start_run():
            fetch_period = self._effective_period(period)
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("period", period)
            mlflow.log_param("fetch_period", fetch_period)

            df = fetch_stock_data(ticker=ticker, period=fetch_period)
            df = validate_stock_data(df)
            df = add_time_series_features(df)

            ml_result = self.ml_agent.analyze(df)
            decision = self.decision_agent.decide(ml_result)

            mlflow.log_param("ml_signal", ml_result.get("signal", "N/A"))
            mlflow.log_param("ml_model", ml_result.get("model_name", "none"))
            mlflow.log_param("final_decision", decision["final_decision"])

            mlflow.log_metric("confidence", decision["confidence"])
            if ml_result.get("available"):
                mlflow.log_metric("ml_probability_up", ml_result["probability_up"])

            artifact_path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                "decision.json",
            )
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(decision, f, indent=2)

            mlflow.log_artifact(artifact_path)

        return decision

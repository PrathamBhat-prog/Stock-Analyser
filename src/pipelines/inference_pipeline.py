import json
import os
import mlflow

from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_technical_indicators

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.risk_agent import RiskAnalysisAgent
from src.agents.decision_agent import DecisionAggregationAgent


class StockAnalysisPipeline:
    """
    End-to-end pipeline with MLflow logging.
    """

    def __init__(self):
        # 🔑 FORCE MLflow to use project-root/mlruns
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mlruns_path = os.path.join(project_root, "mlruns")

        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        mlflow.set_experiment("stock-analysis-pipeline")

        self.technical_agent = TechnicalAnalysisAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.decision_agent = DecisionAggregationAgent()

    def run(self, ticker: str, period: str = "1y") -> dict:
        """
        Run full pipeline and log results using MLflow.
        """

        with mlflow.start_run():
            # ---- Parameters ----
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("period", period)

            # ---- Data pipeline ----
            df = fetch_stock_data(ticker=ticker, period=period)
            df = validate_stock_data(df)
            df = add_technical_indicators(df)

            # ---- Agent outputs ----
            technical_result = self.technical_agent.analyze(df)
            risk_result = self.risk_agent.analyze(df)

            decision = self.decision_agent.aggregate(
                technical_result=technical_result,
                risk_result=risk_result
            )

            # ---- Log agent decisions ----
            mlflow.log_param("technical_signal", technical_result["signal"])
            mlflow.log_param("risk_level", risk_result["risk_level"])
            mlflow.log_param("final_decision", decision["final_decision"])

            # ---- Metrics ----
            mlflow.log_metric("confidence", decision["confidence"])
            mlflow.log_metric("risk_score", risk_result["risk_score"])

            # ---- Artifact ----
            with open("decision.json", "w") as f:
                json.dump(decision, f, indent=2)

            mlflow.log_artifact("decision.json")

        return decision

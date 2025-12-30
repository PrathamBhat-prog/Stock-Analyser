from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_technical_indicators

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.risk_agent import RiskAnalysisAgent
from src.agents.decision_agent import DecisionAggregationAgent


class StockAnalysisPipeline:
    """
    End-to-end pipeline that orchestrates:
    data ingestion -> validation -> feature engineering
    -> multi-agent analysis -> final decision
    """

    def __init__(self):
        self.technical_agent = TechnicalAnalysisAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.decision_agent = DecisionAggregationAgent()

    def run(self, ticker: str, period: str = "1y") -> dict:
        """
        Run the full stock analysis pipeline.

        Parameters:
            ticker (str): Stock symbol (e.g., 'AAPL')
            period (str): Historical period (e.g., '1y')

        Returns:
            dict: Final decision output
        """

        # 1. Data ingestion
        df = fetch_stock_data(ticker=ticker, period=period)

        # 2. Data validation
        df = validate_stock_data(df)

        # 3. Feature engineering
        df = add_technical_indicators(df)

        # 4. Agent analyses
        technical_result = self.technical_agent.analyze(df)
        risk_result = self.risk_agent.analyze(df)

        # 5. Decision aggregation
        final_decision = self.decision_agent.aggregate(
            technical_result=technical_result,
            risk_result=risk_result
        )

        return final_decision

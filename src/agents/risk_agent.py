import pandas as pd


class RiskAnalysisAgent:
    """
    Risk analysis agent that evaluates market uncertainty
    using volatility-based metrics.
    """

    def _to_scalar(self, value) -> float:
        """
        Safely convert pandas Series or scalar to float.
        Prevents pandas FutureWarning and comparison issues.
        """
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        return float(value)

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze market risk based on recent volatility.

        Parameters:
            df (pd.DataFrame): Stock data with Volatility_20 column

        Returns:
            dict: risk_level, risk_score, and reasoning
        """

        latest = df.iloc[-1]
        volatility = self._to_scalar(latest["Volatility_20"])

        # Normalize volatility into a 0–1 risk score
        # Typical daily volatility ranges:
        # < 1%   -> low
        # 1–2%   -> medium
        # > 2%   -> high
        risk_score = min(volatility / 0.03, 1.0)

        if risk_score < 0.33:
            risk_level = "LOW"
            reason = (
                "Market volatility is low, indicating stable price movements "
                "and lower short-term risk."
            )

        elif risk_score < 0.66:
            risk_level = "MEDIUM"
            reason = (
                "Market volatility is moderate, suggesting increased uncertainty "
                "and potential price swings."
            )

        else:
            risk_level = "HIGH"
            reason = (
                "Market volatility is high, indicating significant uncertainty "
                "and elevated risk."
            )

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "reason": reason
        }

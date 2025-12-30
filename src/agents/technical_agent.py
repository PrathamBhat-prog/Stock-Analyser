import pandas as pd


class TechnicalAnalysisAgent:
    """
    Rule-based technical analysis agent.
    Produces interpretable trading signals using technical indicators.
    """

    def _to_scalar(self, value) -> float:
        """
        Safely convert a pandas Series or scalar to a float.
        This avoids pandas FutureWarning and comparison issues.
        """
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        return float(value)

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Analyze the latest technical indicators and return a trading signal.

        Parameters:
            df (pd.DataFrame): Stock data with technical indicators

        Returns:
            dict: signal, confidence, and reasoning
        """

        # Take the most recent row
        latest = df.iloc[-1]

        # Extract scalar values safely
        close = self._to_scalar(latest["Close"])
        sma_20 = self._to_scalar(latest["SMA_20"])
        sma_50 = self._to_scalar(latest["SMA_50"])
        volatility = self._to_scalar(latest["Volatility_20"])

        # Default output
        signal = "NEUTRAL"
        confidence = 0.5
        reason = "Market shows no strong trend."

        # Bullish condition
        if close > sma_20 and sma_20 > sma_50:
            signal = "BULLISH"
            confidence = min(0.7 + volatility, 0.95)
            reason = (
                "Price is above both 20-day and 50-day moving averages, "
                "indicating a strong upward trend."
            )

        # Bearish condition
        elif close < sma_20 and sma_20 < sma_50:
            signal = "BEARISH"
            confidence = min(0.7 + volatility, 0.95)
            reason = (
                "Price is below both 20-day and 50-day moving averages, "
                "indicating a strong downward trend."
            )

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "reason": reason
        }

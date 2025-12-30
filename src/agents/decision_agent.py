class DecisionAggregationAgent:
    """
    Aggregates outputs from multiple agents
    to produce a final trading decision.
    """

    def aggregate(self, technical_result: dict, risk_result: dict) -> dict:
        """
        Combine technical and risk agent outputs into a final decision.

        Parameters:
            technical_result (dict): Output from TechnicalAnalysisAgent
            risk_result (dict): Output from RiskAnalysisAgent

        Returns:
            dict: Final decision with reasoning
        """

        signal = technical_result["signal"]
        tech_confidence = technical_result["confidence"]

        risk_level = risk_result["risk_level"]
        risk_score = risk_result["risk_score"]

        final_decision = "HOLD"
        confidence = 0.5
        reasoning = "Insufficient alignment between agents."

        # --- Decision Logic ---

        if signal == "BULLISH" and risk_level == "LOW":
            final_decision = "BUY"
            confidence = min(tech_confidence * 1.1, 0.95)
            reasoning = (
                "Technical indicators show a bullish trend and market risk is low, "
                "supporting a buy decision."
            )

        elif signal == "BEARISH" and risk_level == "LOW":
            final_decision = "SELL"
            confidence = min(tech_confidence * 1.1, 0.95)
            reasoning = (
                "Technical indicators show a bearish trend and market risk is low, "
                "supporting a sell decision."
            )

        elif risk_level == "HIGH":
            final_decision = "HOLD"
            confidence = max(0.4, 1 - risk_score)
            reasoning = (
                "Market risk is high, indicating elevated uncertainty. "
                "Holding position is safer despite technical signals."
            )

        elif signal == "NEUTRAL":
            final_decision = "HOLD"
            confidence = 0.5
            reasoning = (
                "Technical indicators do not show a clear trend. "
                "Holding position is recommended."
            )

        return {
            "final_decision": final_decision,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "agent_summary": {
                "technical": technical_result,
                "risk": risk_result
            }
        }

class MLDecisionAgent:
    """
    Maps ML time-series model output to a final trading decision.
    No rule-based technical or risk logic — ML probability only.
    """

    BUY_THRESHOLD = 0.58
    SELL_THRESHOLD = 0.42

    def decide(self, ml_result: dict) -> dict:
        if not ml_result.get("available", False):
            return {
                "final_decision": "HOLD",
                "confidence": 0.0,
                "reasoning": ml_result.get(
                    "reason",
                    "ML model unavailable. Run `python train.py` first.",
                ),
                "agent_summary": {"ml": ml_result},
            }

        proba_up = ml_result["probability_up"]
        horizon = ml_result.get("forecast_horizon_days", 5)
        model_name = ml_result.get("model_name", "unknown")

        if proba_up >= self.BUY_THRESHOLD:
            final_decision = "BUY"
            confidence = proba_up
            reasoning = (
                f"Time-series model ({model_name}) assigns {proba_up:.1%} probability "
                f"that price rises over the next {horizon} trading days."
            )
        elif proba_up <= self.SELL_THRESHOLD:
            final_decision = "SELL"
            confidence = 1 - proba_up
            reasoning = (
                f"Time-series model ({model_name}) assigns {1 - proba_up:.1%} probability "
                f"that price falls over the next {horizon} trading days."
            )
        else:
            final_decision = "HOLD"
            confidence = max(proba_up, 1 - proba_up)
            reasoning = (
                f"Time-series model ({model_name}) is uncertain "
                f"({proba_up:.1%} up / {1 - proba_up:.1%} down). "
                f"No clear edge for the next {horizon} trading days."
            )

        return {
            "final_decision": final_decision,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "agent_summary": {"ml": ml_result},
        }

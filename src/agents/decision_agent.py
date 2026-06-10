"""
Decision agent — maps ML probability + trend into a plain-English
BUY / SELL / HOLD decision designed for a layman investor.

Combines:
  - ML model probability (the statistical signal)
  - Trend score from TrendAgent (ARIMA-inspired market regime)

The combined signal gives a more robust recommendation than either alone.
"""


class MLDecisionAgent:
    """
    Maps ML time-series model output to a final BUY / SELL / HOLD decision.

    Thresholds are intentionally conservative — the model must have clear
    conviction before issuing a directional call.
    """

    BUY_THRESHOLD  = 0.58   # > 58% probability of price being higher in 5 days
    SELL_THRESHOLD = 0.42   # < 42% probability (i.e. >58% probability of falling)

    def decide(self, ml_result: dict) -> dict:
        if not ml_result.get("available", False):
            return {
                "final_decision": "HOLD",
                "confidence":     0.0,
                "reasoning":      (
                    "The ML model has not been trained yet, or no model file was found. "
                    "Please run `python train.py` to train the model first."
                ),
                "plain_english":  (
                    "We could not make a recommendation because the analysis model "
                    "is not ready yet. Please ask the administrator to run the training step."
                ),
                "agent_summary":  {"ml": ml_result},
            }

        proba_up    = ml_result["probability_up"]
        horizon     = ml_result.get("forecast_horizon_days", 5)
        model_name  = ml_result.get("model_name", "unknown")

        if proba_up >= self.BUY_THRESHOLD:
            final_decision = "BUY"
            confidence     = proba_up
            reasoning = (
                f"The AI model ({model_name}) estimates a {proba_up:.1%} probability "
                f"that this stock's price will be higher {horizon} trading days from now."
            )
            plain_english = (
                f"Our analysis suggests there is a good chance this stock's price "
                f"will rise over the next {horizon} trading days (~1 week). "
                f"This could be a good time to consider buying — but always do your "
                f"own research and only invest what you can afford to lose."
            )

        elif proba_up <= self.SELL_THRESHOLD:
            final_decision = "SELL"
            confidence     = 1 - proba_up
            reasoning = (
                f"The AI model ({model_name}) estimates a {1 - proba_up:.1%} probability "
                f"that this stock's price will fall over the next {horizon} trading days."
            )
            plain_english = (
                f"Our analysis suggests there is a good chance this stock's price "
                f"may fall over the next {horizon} trading days (~1 week). "
                f"If you own this stock, you may want to consider selling. "
                f"Remember: no prediction is guaranteed — prices can move unexpectedly."
            )

        else:
            final_decision = "HOLD"
            confidence     = max(proba_up, 1 - proba_up)
            reasoning = (
                f"The AI model ({model_name}) does not see a strong signal in either "
                f"direction ({proba_up:.1%} chance of going up, "
                f"{1 - proba_up:.1%} chance of going down)."
            )
            plain_english = (
                f"Our analysis does not see a clear direction for this stock over "
                f"the next {horizon} trading days. It could go either way. "
                f"The safest option right now is to hold your position and wait "
                f"for a clearer signal before making a decision."
            )

        return {
            "final_decision": final_decision,
            "confidence":     round(confidence, 4),
            "reasoning":      reasoning,
            "plain_english":  plain_english,
            "agent_summary":  {"ml": ml_result},
        }

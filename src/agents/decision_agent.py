"""
Multi-horizon decision agent.

Investment horizons supported:
  Short-term  (5 trading days  ~ 1 week)   : ML model primary signal
  Medium-term (21 trading days ~ 1 month)  : ML + trend composite
  Long-term   (63 trading days ~ 3 months) : ML + trend + momentum composite
  Very long   (126 days ~ 6 months)        : Trend + momentum dominant
  Annual      (252 days ~ 1 year)          : Trend regime + fundamental proxy

For horizons beyond 5 days the ML model (trained on 5-day labels) still
provides the short-term directional impulse, but its weight is reduced and
the trend-analysis score (from TrendAgent) takes increasing weight.

This approach requires NO retraining — it combines the existing signal
with technical analysis in a disciplined, weight-adjusted framework.
"""

from __future__ import annotations

HORIZONS = {
    "5d":   {"label": "1 Week  (5 trading days)",    "days": 5,   "ml_weight": 0.80, "trend_weight": 0.20},
    "21d":  {"label": "1 Month (21 trading days)",   "days": 21,  "ml_weight": 0.50, "trend_weight": 0.50},
    "63d":  {"label": "3 Months (63 trading days)",  "days": 63,  "ml_weight": 0.30, "trend_weight": 0.70},
    "126d": {"label": "6 Months (126 trading days)", "days": 126, "ml_weight": 0.15, "trend_weight": 0.85},
    "252d": {"label": "1 Year  (252 trading days)",  "days": 252, "ml_weight": 0.10, "trend_weight": 0.90},
}

DEFAULT_HORIZON = "5d"

BUY_THRESHOLD  = 0.58
SELL_THRESHOLD = 0.42


class MLDecisionAgent:
    """
    Maps ML probability + trend score into a BUY / SELL / HOLD decision
    for any investment horizon, with plain-English reasoning for laymen.

    Horizon weighting logic:
      - Short-term:   ML model dominates (80%) — trained for this exact task
      - Medium-term:  Equal weight (50/50)
      - Long-term:    Trend dominates (70-90%) — ML signal too short for these horizons
    """

    def decide(
        self,
        ml_result: dict,
        trend_result: dict | None = None,
        horizon_key: str = DEFAULT_HORIZON,
    ) -> dict:
        horizon_cfg  = HORIZONS.get(horizon_key, HORIZONS[DEFAULT_HORIZON])
        horizon_days = horizon_cfg["days"]
        horizon_lbl  = horizon_cfg["label"]
        ml_w         = horizon_cfg["ml_weight"]
        trend_w      = horizon_cfg["trend_weight"]

        # ── ML signal ─────────────────────────────────────────────────────────
        if not ml_result.get("available", False):
            ml_score   = 0.5           # neutral fallback
            model_name = "unavailable"
            ml_note    = "ML model not ready — train first with `python train.py`."
        else:
            ml_score   = ml_result["probability_up"]   # 0-1, >0.5 = bullish
            model_name = ml_result.get("model_name", "unknown")
            ml_note    = None

        # ── Trend score  (TrendAgent returns score in [-1, 1]) ─────────────────
        trend_score    = 0.0
        trend_label    = "Unknown"
        trend_summary  = ""
        if trend_result:
            trend_score   = float(trend_result.get("trend_score", 0.0))
            trend_label   = trend_result.get("trend_label", "Unknown")
            trend_summary = trend_result.get("summary", "")

        # Normalise trend_score from [-1,1] to [0,1] for blending
        trend_prob = (trend_score + 1.0) / 2.0

        # ── Composite score ────────────────────────────────────────────────────
        composite = ml_w * ml_score + trend_w * trend_prob

        # ── Decision ───────────────────────────────────────────────────────────
        if composite >= BUY_THRESHOLD:
            decision   = "BUY"
            confidence = composite
        elif composite <= SELL_THRESHOLD:
            decision   = "SELL"
            confidence = 1.0 - composite
        else:
            decision   = "HOLD"
            confidence = max(composite, 1.0 - composite)

        # ── Reasoning ─────────────────────────────────────────────────────────
        if ml_note:
            technical_reason = ml_note
        else:
            technical_reason = (
                f"ML model ({model_name}): {ml_score:.1%} probability of price rising in 5 days. "
                f"Trend analysis: {trend_label} (score {trend_score:+.2f}). "
                f"Composite signal for {horizon_lbl}: {composite:.1%} "
                f"(ML weight {ml_w:.0%}, Trend weight {trend_w:.0%})."
            )

        plain_english = _plain_english(decision, confidence, horizon_days, horizon_lbl, trend_label, ml_w)

        return {
            "final_decision":   decision,
            "confidence":       round(confidence, 4),
            "horizon":          horizon_lbl,
            "horizon_days":     horizon_days,
            "composite_score":  round(composite, 4),
            "ml_probability":   round(ml_score, 4),
            "trend_score":      round(trend_score, 4),
            "ml_weight":        ml_w,
            "trend_weight":     trend_w,
            "reasoning":        technical_reason,
            "plain_english":    plain_english,
            "agent_summary":    {"ml": ml_result},
        }


def _plain_english(
    decision: str,
    confidence: float,
    horizon_days: int,
    horizon_lbl: str,
    trend_label: str,
    ml_weight: float,
) -> str:
    pct = f"{confidence:.0%}"
    horizon_plain = {
        5:   "about 1 week",
        21:  "about 1 month",
        63:  "about 3 months",
        126: "about 6 months",
        252: "about 1 year",
    }.get(horizon_days, f"{horizon_days} trading days")

    method_note = (
        "based mainly on the ML model's short-term signal"
        if ml_weight >= 0.6 else
        "based on a combination of ML signal and market trend analysis"
        if ml_weight >= 0.3 else
        "based mainly on the long-term market trend analysis"
    )

    if decision == "BUY":
        return (
            f"Our analysis ({method_note}) suggests there is a good chance "
            f"this stock's price will be higher {horizon_plain} from now. "
            f"The market trend is currently showing a {trend_label.lower()}. "
            f"AI confidence: {pct}. "
            f"This could be a good time to consider buying — "
            f"but always do your own research and only invest what you can afford to lose."
        )
    elif decision == "SELL":
        return (
            f"Our analysis ({method_note}) suggests there is a reasonable chance "
            f"this stock's price may be lower {horizon_plain} from now. "
            f"The market trend is currently showing a {trend_label.lower()}. "
            f"AI confidence: {pct}. "
            f"If you own this stock, you may want to consider reducing your position. "
            f"Remember: no prediction is guaranteed — always consult a financial advisor."
        )
    else:
        return (
            f"Our analysis ({method_note}) does not see a strong directional signal "
            f"for this stock over the next {horizon_plain}. "
            f"The market trend is showing a {trend_label.lower()}. "
            f"The safest approach right now is to hold and wait for a clearer signal. "
            f"Patience is often the best strategy in uncertain markets."
        )

# -*- coding: utf-8 -*-
"""
Sentiment scorer using VADER (fast, no GPU).
Produces compound scores [-1, +1] for financial headlines.

Upgrade path:
  VADER (now) -> FinBERT (HuggingFace, +3-5% AUC) -> GPT-4 API
"""

from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _vader = None
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")


def score_headlines_vader(headlines: list[str]) -> float:
    """
    Score a list of headlines with VADER.
    Returns mean compound score in [-1, +1].
    0.0 = neutral / no headlines.
    """
    if not VADER_AVAILABLE or not headlines:
        return 0.0
    scores = [_vader.polarity_scores(h)["compound"] for h in headlines if h]
    return float(np.mean(scores)) if scores else 0.0


def compute_sentiment(headlines: list[str]) -> dict:
    """
    Returns a dict with all sentiment features needed by the model.

    Keys match trading_model_v5.pkl feature names:
      rolling_sentiment_20d  -- primary sentiment score (use for today's value)
      sent_lag_1             -- caller should shift this from cache
      sent_lag_3             -- caller should shift this from cache
      sent_lag_5             -- caller should shift this from cache
      sent_vix_interaction   -- sentiment * VIX (caller fills in VIX)
    """
    score = score_headlines_vader(headlines)
    logger.info(f"Sentiment score: {score:+.4f} ({len(headlines)} headlines)")
    return {
        "rolling_sentiment_20d": score,
        "sent_positive": max(0.0, score),
        "sent_negative": min(0.0, score),
        "headline_count": len(headlines),
    }

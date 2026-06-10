"""
Finance-grade evaluation metrics for stock direction prediction.

Why these metrics matter:
- MAE / RMSE / MAPE : measure how far predicted probabilities are from true labels
  (used as regression-style confidence calibration checks).
- Directional Accuracy: percentage of correct UP/DOWN calls — directly maps to
  a coin-flip baseline (50%).  Even 53%+ is economically significant at scale.
- Sharpe Ratio (annualised): risk-adjusted return of a naïve strategy that goes
  long when the model predicts UP.  A Sharpe > 1.0 is considered production-ready
  by most quant funds.
- ROC-AUC: discrimination power regardless of threshold (our primary selection metric).
- F1 / Precision / Recall : class-imbalance aware classification metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core regression-style metrics on probability outputs
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Mean Absolute Error between predicted probability and binary label."""
    return float(np.mean(np.abs(probas - y_true)))


def rmse(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Root Mean Squared Error between predicted probability and binary label."""
    return float(np.sqrt(np.mean((probas - y_true) ** 2)))


def mape(y_true: np.ndarray, probas: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    Clips denominator to eps to avoid division-by-zero on zero-labels.
    """
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs(probas - y_true) / denom))


# ---------------------------------------------------------------------------
# Finance-specific metrics
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, preds: np.ndarray) -> float:
    """
    Fraction of correct directional calls (UP vs DOWN).
    Equivalent to accuracy on a binary direction label.
    A random walk gives ~50%; >52% is considered signal.
    """
    return float(accuracy_score(y_true, preds))


def sharpe_ratio(
    y_true: np.ndarray,
    preds: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Annualised Sharpe ratio of a naïve long-flat strategy driven by model predictions.

    Strategy:
        - Go long (+1) when model predicts UP (pred == 1)
        - Stay flat (0) when model predicts DOWN (pred == 0)
        - Daily P&L proxy: direction_correct * 1  OR  direction_wrong * -1

    Parameters
    ----------
    y_true        : actual binary direction labels (1 = price went up)
    preds         : model binary predictions
    risk_free_rate: annualised risk-free rate (default 0.0)
    trading_days  : trading days per year for annualisation (default 252)

    Returns
    -------
    float : annualised Sharpe ratio (nan if std is zero)
    """
    # Daily strategy returns: +1 if correct long call, -1 if wrong
    long_mask = preds == 1
    if long_mask.sum() == 0:
        return float("nan")

    returns = np.where(
        long_mask,
        np.where(y_true == 1, 1.0, -1.0),  # long positions
        0.0,                                 # flat positions
    )

    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    std = excess.std()
    if std < 1e-9:
        return float("nan")

    return float((excess.mean() / std) * np.sqrt(trading_days))


# ---------------------------------------------------------------------------
# Master helper used by all trainers
# ---------------------------------------------------------------------------

def compute_all(
    y_true: np.ndarray,
    preds: np.ndarray,
    probas: np.ndarray,
) -> dict:
    """
    Compute the full suite of classification + finance metrics.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc,
        mae, rmse, mape, directional_accuracy, sharpe_ratio
    """
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    probas = np.asarray(probas)

    return {
        # ── Classification ──────────────────────────────────────────────
        "accuracy":   float(accuracy_score(y_true, preds)),
        "precision":  float(precision_score(y_true, preds, zero_division=0)),
        "recall":     float(recall_score(y_true, preds, zero_division=0)),
        "f1":         float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_true, probas)),
        # ── Probability calibration ──────────────────────────────────────
        "mae":        mae(y_true, probas),
        "rmse":       rmse(y_true, probas),
        "mape":       mape(y_true, probas),
        # ── Finance-specific ─────────────────────────────────────────────
        "directional_accuracy": directional_accuracy(y_true, preds),
        "sharpe_ratio":         sharpe_ratio(y_true, preds),
    }

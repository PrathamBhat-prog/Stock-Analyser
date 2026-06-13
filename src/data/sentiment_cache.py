# -*- coding: utf-8 -*-
"""
SQLite cache for daily sentiment scores.
Schema: sentiment_cache(ticker TEXT, date TEXT, score REAL, article_count INT)

Why cache?
  - GDELT rate limits requests
  - Past dates' sentiment never changes -- no need to re-fetch
  - Provides sent_lag_1/3/5 from historical cache rows
"""

from __future__ import annotations
import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("artifacts") / "sentiment.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            ticker        TEXT NOT NULL,
            date          TEXT NOT NULL,
            score         REAL NOT NULL DEFAULT 0.0,
            article_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    return conn


def save_sentiment(ticker: str, day: date, score: float, count: int = 0) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sentiment_cache VALUES (?,?,?,?)",
            (ticker.upper(), day.isoformat(), score, count),
        )
        conn.commit()
    logger.debug(f"Saved sentiment: {ticker} {day} score={score:+.4f}")


def get_sentiment(ticker: str, day: date) -> float | None:
    """Returns cached score or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT score FROM sentiment_cache WHERE ticker=? AND date=?",
            (ticker.upper(), day.isoformat()),
        ).fetchone()
    return float(row[0]) if row else None


def get_sentiment_window(ticker: str, end_date: date, days: int = 20) -> list[float]:
    """
    Returns list of daily scores for the last `days` calendar days.
    Missing days are filled with 0.0 (neutral).
    Used to compute rolling_sentiment_20d and lag features.
    """
    start = end_date - timedelta(days=days)
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT date, score FROM sentiment_cache "
            "WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
            (ticker.upper(), start.isoformat(), end_date.isoformat()),
        ).fetchall()
    score_map = {r[0]: r[1] for r in rows}
    return [score_map.get((start + timedelta(days=i)).isoformat(), 0.0)
            for i in range(days + 1)]


def build_sentiment_features(ticker: str, today: date, vix: float) -> dict:
    """
    Build all sentiment features required by trading_model_v5.pkl.
    Call this during inference in the pipeline.
    """
    window = get_sentiment_window(ticker, today, days=20)
    roll20 = float(sum(window) / max(len(window), 1))

    def lag_score(n: int) -> float:
        day = today - timedelta(days=n)
        return get_sentiment(ticker, day) or 0.0

    return {
        "rolling_sentiment_20d": roll20,
        "sent_lag_1":            lag_score(1),
        "sent_lag_3":            lag_score(3),
        "sent_lag_5":            lag_score(5),
        "sent_vix_interaction":  roll20 * vix,
    }

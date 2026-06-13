# -*- coding: utf-8 -*-
"""
News fetcher for production sentiment pipeline.
Sources (in priority order):
  1. GDELT Project API  -- free, no key, global news, 15-min updates
  2. yfinance .news     -- free fallback, recent headlines only
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMEOUT = 10
GDELT_RATE_SLEEP = 0.5   # seconds between requests (polite)


def fetch_gdelt_headlines(query: str, days_back: int = 3) -> list[str]:
    """
    Fetch up to 50 article titles from GDELT for `query`.
    Returns empty list on any failure.
    """
    params = {
        "query":      query,
        "mode":       "artlist",
        "maxrecords": 50,
        "timespan":   f"{days_back}d",
        "format":     "json",
    }
    try:
        r = requests.get(GDELT_URL, params=params, timeout=GDELT_TIMEOUT)
        if r.status_code != 200:
            logger.warning(f"GDELT HTTP {r.status_code} for query={query!r}")
            return []
        articles = r.json().get("articles", [])
        titles   = [a.get("title", "") for a in articles if a.get("title")]
        logger.info(f"GDELT: {len(titles)} headlines for {query!r}")
        time.sleep(GDELT_RATE_SLEEP)
        return titles
    except Exception as exc:
        logger.warning(f"GDELT fetch failed for {query!r}: {exc}")
        return []


def fetch_yfinance_headlines(ticker: str) -> list[str]:
    """
    Fetch recent headlines from yfinance (fallback).
    Returns empty list on failure.
    """
    try:
        import yfinance as yf
        news   = yf.Ticker(ticker).news or []
        titles = [n.get("title", "") for n in news[:30] if n.get("title")]
        logger.info(f"yfinance: {len(titles)} headlines for {ticker}")
        return titles
    except Exception as exc:
        logger.warning(f"yfinance news failed for {ticker}: {exc}")
        return []


def fetch_headlines(ticker: str, company_name: str = "") -> list[str]:
    """
    Main entry point. Tries GDELT first, falls back to yfinance.
    `company_name` improves GDELT query quality (e.g. 'Apple' for AAPL).
    """
    query = company_name or ticker.replace(".NS", "").replace(".AS", "")
    headlines = fetch_gdelt_headlines(f"{query} stock earnings")

    if not headlines:
        logger.info(f"GDELT empty, falling back to yfinance for {ticker}")
        headlines = fetch_yfinance_headlines(ticker)

    return headlines

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
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMEOUT = 10
GDELT_RATE_SLEEP = 0.5   # seconds between requests (polite)
GDELT_MAX_RETRIES = 3
GDELT_CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours
CACHE_DIR = Path(__file__).parents[2] / ".cache" / "news"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    # Simple on-disk cache to reduce GDELT calls and avoid rate limits
    cache_key = hashlib.sha256(f"{query}|{days_back}".encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            age = time.time() - payload.get("ts", 0)
            if age < GDELT_CACHE_TTL_SECONDS:
                logger.info(f"GDELT cache hit for {query!r} (age={age:.0f}s)")
                return payload.get("titles", [])
        except Exception:
            # ignore cache read errors and continue to fetch
            pass

    # Retry logic with exponential backoff for 429/5xx responses
    backoff = 1.0
    for attempt in range(1, GDELT_MAX_RETRIES + 1):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=GDELT_TIMEOUT)
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                titles = [a.get("title", "") for a in articles if a.get("title")]
                logger.info(f"GDELT: {len(titles)} headlines for {query!r}")
                # cache result
                try:
                    with cache_file.open("w", encoding="utf-8") as fh:
                        json.dump({"ts": time.time(), "titles": titles}, fh)
                except Exception:
                    pass
                time.sleep(GDELT_RATE_SLEEP)
                return titles
            elif r.status_code == 429:
                logger.warning(f"GDELT HTTP 429 (rate limit) for query={query!r}, attempt={attempt}")
            else:
                logger.warning(f"GDELT HTTP {r.status_code} for query={query!r}")
        except Exception as exc:
            logger.warning(f"GDELT fetch failed for {query!r}: {exc}")

        # backoff and retry
        time.sleep(backoff)
        backoff *= 2

    logger.warning(f"GDELT: exhausted retries for {query!r}")
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

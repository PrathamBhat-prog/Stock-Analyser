import yfinance as yf
import pandas as pd


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.

    Parameters:
        ticker (str): Stock symbol (e.g., 'AAPL')
        period (str): Data period (e.g., '1y', '6mo', '5d')
        interval (str): Data interval (e.g., '1d', '1h')

    Returns:
        pd.DataFrame: Historical stock data
    """

    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False
    )

    if data is None or data.empty:
        raise ValueError(
            f"No data found for ticker {ticker}. "
            f"Check ticker symbol or internet connection."
        )

    data.reset_index(inplace=True)
    return data

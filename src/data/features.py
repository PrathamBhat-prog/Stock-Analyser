import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to stock data.

    Indicators:
    - Simple Moving Averages (20, 50)
    - Daily returns
    - Volatility (20-day rolling std)
    """

    df = df.copy()

    # Simple Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Volatility (risk measure)
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    return df

import pandas as pd


def add_direction_label(
    df: pd.DataFrame,
    horizon: int = 5,
    target_col: str = "target_up",
) -> pd.DataFrame:
    """
    Binary label: 1 if close price is higher after `horizon` trading days, else 0.
    Rows without a future price are dropped.
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df[target_col] = (future_close > df["Close"]).astype(int)
    df = df.dropna(subset=[target_col])
    return df

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def validate_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate stock market data before further processing.

    Checks:
    - Required columns exist
    - No missing values
    - Correct data types

    Returns:
        Cleaned DataFrame
    """

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Check for missing values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        df = df.dropna(subset=REQUIRED_COLUMNS)

    # 3. Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # 4. Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    return df

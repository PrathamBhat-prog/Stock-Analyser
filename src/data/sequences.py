import numpy as np
import pandas as pd

from src.models.feature_columns import FEATURE_COLUMNS, TARGET_COLUMN


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build LSTM sequences without crossing ticker boundaries.
    Each sample: (seq_len, n_features) -> binary label.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _, group in df.groupby("ticker", sort=False):
        group = group.sort_values("Date")
        features = group[feature_cols].values.astype(np.float32)
        targets = group[target_col].values.astype(np.int64)

        if len(group) <= seq_len:
            continue

        for i in range(seq_len, len(group)):
            sequences.append(features[i - seq_len : i])
            labels.append(targets[i])

    if not sequences:
        raise ValueError("No sequences could be built. Increase history or reduce seq_len.")

    return np.stack(sequences), np.array(labels)

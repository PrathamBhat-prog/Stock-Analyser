# =============================================================================
# Stock Market ML Model Testing Script
# Run this file in Google Colab cell-by-cell (each section is one cell)
#
# Tests:
#   1. HistGradientBoostingClassifier  (winner from local training)
#   2. LSTM (PyTorch, 2-layer bidirectional)
#
# No local repo required — everything is self-contained.
# =============================================================================

# %% [CELL 1] Install dependencies
# -------------------------------------------------------
# Paste this block into the FIRST Colab cell and run it.
# -------------------------------------------------------
# !pip install -q yfinance scikit-learn torch matplotlib pandas numpy


# %% [CELL 2] Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("All imports OK")
print(f"PyTorch version: {torch.__version__}")


# %% [CELL 3] Configuration — change tickers / period here
TICKERS  = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
             "RELIANCE.NS", "TCS.NS", "INFY.NS"]   # add/remove freely
PERIOD   = "5y"          # "2y", "5y", "10y"
HORIZON  = 5             # predict direction N days ahead
SEQ_LEN  = 30            # LSTM lookback window (days)
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
# test = remaining 15%


# %% [CELL 4] Feature engineering (same as production pipeline)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """60 production-grade technical features — strictly backward-looking."""
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # Returns
    df["Daily_Return"] = close.pct_change()
    df["Log_Return"]   = np.log(close / close.shift(1))

    # Lags
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"Return_lag_{lag}"]  = df["Daily_Return"].shift(lag)
        df[f"Close_lag_{lag}"]   = close.shift(lag)
        df[f"Volume_lag_{lag}"]  = vol.shift(lag)

    # Rolling stats
    for w in [5, 10, 20, 60]:
        df[f"Return_mean_{w}"] = df["Daily_Return"].rolling(w).mean()
        df[f"Return_std_{w}"]  = df["Daily_Return"].rolling(w).std()
        df[f"Close_mean_{w}"]  = close.rolling(w).mean()
        df[f"Volume_mean_{w}"] = vol.rolling(w).mean()

    # Momentum
    for p in [5, 10, 20, 60]:
        df[f"Momentum_{p}"] = close.pct_change(p)

    # Volume ratios
    df["Volume_ratio_20"] = vol / vol.rolling(20).mean()
    df["Volume_ratio_5"]  = vol / vol.rolling(5).mean()

    # RSI
    for period in [14, 28]:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
        loss  = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd, sig, macd - sig

    # Bollinger Bands
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_Mid"]   = bb_mid
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower
    df["BB_Width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    df["BB_Pct"]   = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # ATR
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    df["ATR_14"]     = tr.ewm(com=13, min_periods=14).mean()
    df["ATR_14_pct"] = df["ATR_14"] / close.replace(0, np.nan)

    # OBV
    sign      = np.sign(close.diff()).fillna(0)
    df["OBV"]        = (sign * vol).cumsum()
    df["OBV_ROC_10"] = df["OBV"].pct_change(10)

    # Regime
    df["Close_vs_MA20"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    df["Close_vs_MA60"] = (close - close.rolling(60).mean()) / close.rolling(60).std().replace(0, np.nan)

    # HL Range
    df["HL_Range"]     = (high - low) / close.replace(0, np.nan)
    df["HL_Range_5ma"] = df["HL_Range"].rolling(5).mean()

    # Target: 1 if price higher in HORIZON days
    df["target"] = (close.shift(-HORIZON) > close).astype(int)

    return df

FEATURE_COLS = [
    "Daily_Return", "Log_Return",
    *[f"Return_lag_{l}" for l in [1,2,3,5,10,20]],
    *[f"Close_lag_{l}"  for l in [1,2,3,5,10,20]],
    *[f"Volume_lag_{l}" for l in [1,2,3,5,10,20]],
    *[f"Return_mean_{w}" for w in [5,10,20,60]],
    *[f"Return_std_{w}"  for w in [5,10,20,60]],
    *[f"Close_mean_{w}"  for w in [5,10,20,60]],
    *[f"Volume_mean_{w}" for w in [5,10,20,60]],
    *[f"Momentum_{p}" for p in [5,10,20,60]],
    "Volume_ratio_20", "Volume_ratio_5",
    "RSI_14", "RSI_28",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Mid", "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct",
    "ATR_14", "ATR_14_pct",
    "OBV", "OBV_ROC_10",
    "Close_vs_MA20", "Close_vs_MA60",
    "HL_Range", "HL_Range_5ma",
]
print(f"Feature count: {len(FEATURE_COLS)}")


# %% [CELL 5] Fetch data and build dataset
print(f"Fetching {len(TICKERS)} tickers, period={PERIOD} ...")
frames = []
for ticker in TICKERS:
    try:
        raw = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True)
        raw = raw.reset_index()
        raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
        raw = raw.rename(columns={"Date": "Date"})
        if len(raw) < 150:
            print(f"  Skipping {ticker}: only {len(raw)} rows")
            continue
        df  = add_features(raw)
        df  = df.dropna(subset=FEATURE_COLS + ["target"])
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df)} rows")
    except Exception as e:
        print(f"  {ticker} ERROR: {e}")

all_data = pd.concat(frames, ignore_index=True)
# Chronological sort
all_data = all_data.sort_values("Date").reset_index(drop=True)

n        = len(all_data)
n_train  = int(n * TRAIN_PCT)
n_val    = int(n * (TRAIN_PCT + VAL_PCT))

train_df = all_data.iloc[:n_train]
val_df   = all_data.iloc[n_train:n_val]
test_df  = all_data.iloc[n_val:]

X_train, y_train = train_df[FEATURE_COLS].values, train_df["target"].values
X_val,   y_val   = val_df[FEATURE_COLS].values,   val_df["target"].values
X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df["target"].values

print(f"\nDataset: {n} total rows | train={len(X_train)} val={len(X_val)} test={len(X_test)}")
print(f"Class balance (train): {y_train.mean():.2%} positive")


# %% [CELL 6] ============================================================
#             MODEL 1: HistGradientBoostingClassifier
# ========================================================================
print("\n" + "="*60)
print("  MODEL 1: HistGradientBoostingClassifier")
print("="*60)

hgb = HistGradientBoostingClassifier(
    max_iter        = 400,
    max_leaf_nodes  = 31,
    learning_rate   = 0.05,
    min_samples_leaf= 20,
    l2_regularization= 0.1,
    random_state    = 42,
    early_stopping  = True,
    validation_fraction= 0.1,
    n_iter_no_change= 10,
    verbose         = 1,       # shows iteration progress
)

hgb.fit(X_train, y_train)

def evaluate(model, X, y, label):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(f"\n  [{label}]")
    print(f"    Accuracy  : {accuracy_score(y, pred):.4f}")
    print(f"    Precision : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"    F1        : {f1_score(y, pred, zero_division=0):.4f}")
    print(f"    ROC-AUC   : {roc_auc_score(y, proba):.4f}")
    return roc_auc_score(y, proba)

evaluate(hgb, X_val,  y_val,  "Validation")
evaluate(hgb, X_test, y_test, "Test      ")

# Feature importance plot
importances = hgb.feature_importances_
top_idx = np.argsort(importances)[-20:]
plt.figure(figsize=(10, 6))
plt.barh([FEATURE_COLS[i] for i in top_idx], importances[top_idx], color="#3B82F6")
plt.title("HistGradientBoosting - Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("hgb_feature_importance.png", dpi=150)
plt.show()
print("Saved: hgb_feature_importance.png")


# %% [CELL 7] ============================================================
#             MODEL 2: LSTM (PyTorch)
# ========================================================================
print("\n" + "="*60)
print("  MODEL 2: LSTM (PyTorch)")
print("="*60)

# Scale features for LSTM
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_val   = scaler.transform(X_val)
Xs_test  = scaler.transform(X_test)


def make_sequences(X, y, seq_len):
    """Convert flat feature matrix to overlapping windows."""
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xseq_train, yseq_train = make_sequences(Xs_train, y_train, SEQ_LEN)
Xseq_val,   yseq_val   = make_sequences(Xs_val,   y_val,   SEQ_LEN)
Xseq_test,  yseq_test  = make_sequences(Xs_test,  y_test,  SEQ_LEN)

print(f"Sequence shapes  train={Xseq_train.shape}  val={Xseq_val.shape}  test={Xseq_test.shape}")

BATCH_SIZE   = 256
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
EPOCHS       = 30       # reduce to 10 for quick test
PATIENCE     = 6
LEARNING_RATE= 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xseq_train), torch.from_numpy(yseq_train)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xseq_val), torch.from_numpy(yseq_val)),
    batch_size=BATCH_SIZE, shuffle=False
)

# LSTM Model definition
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.norm  = nn.LayerNorm(hidden_size)
        self.head  = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = self.norm(out[:, -1, :])   # last timestep
        return self.head(out).squeeze(1)


model     = StockLSTM(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-5
)

# ── Training loop ──────────────────────────────────────────────────────
history = {"train_loss": [], "val_auc": []}
best_auc   = 0.0
best_state = None
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds = model(Xb.to(DEVICE)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    val_acc = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    scheduler.step(val_auc)

    history["train_loss"].append(avg_loss)
    history["val_auc"].append(val_auc)

    print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}  val_acc={val_acc:.4f}")

    if val_auc > best_auc:
        best_auc   = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# Restore best weights
if best_state:
    model.load_state_dict(best_state)

# ── Epoch plot ─────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(history["train_loss"], color="#3B82F6", linewidth=2, label="Train Loss")
ax2.plot(history["val_auc"],    color="#F97316", linewidth=2, label="Val AUC")

best_ep = int(np.argmax(history["val_auc"]))
ax2.axvline(best_ep, color="#10B981", linestyle="--", linewidth=1.5,
            label=f"Best epoch={best_ep+1} (AUC={history['val_auc'][best_ep]:.4f})")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss", color="#3B82F6")
ax2.set_ylabel("Val ROC-AUC", color="#F97316")
ax1.set_title("LSTM Training Progress — Loss & Validation AUC")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("lstm_epoch_plot.png", dpi=150)
plt.show()
print("Saved: lstm_epoch_plot.png")

# ── Test evaluation ────────────────────────────────────────────────────
model.eval()
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xseq_test), torch.from_numpy(yseq_test)),
    batch_size=BATCH_SIZE, shuffle=False
)
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = model(Xb.to(DEVICE)).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

preds_bin = (np.array(all_preds) >= 0.5).astype(int)
print(f"\n  [LSTM Test Results]")
print(f"    Accuracy  : {accuracy_score(all_labels, preds_bin):.4f}")
print(f"    Precision : {precision_score(all_labels, preds_bin, zero_division=0):.4f}")
print(f"    Recall    : {recall_score(all_labels, preds_bin, zero_division=0):.4f}")
print(f"    F1        : {f1_score(all_labels, preds_bin, zero_division=0):.4f}")
print(f"    ROC-AUC   : {roc_auc_score(all_labels, all_preds):.4f}")
print(f"    Best val AUC: {best_auc:.4f} (epoch {best_ep+1})")


# %% [CELL 8] Final comparison table
print("\n" + "="*60)
print("  FINAL MODEL COMPARISON")
print("="*60)

hgb_proba_test = hgb.predict_proba(X_test)[:, 1]
hgb_pred_test  = (hgb_proba_test >= 0.5).astype(int)

lstm_auc  = roc_auc_score(all_labels, all_preds)
lstm_f1   = f1_score(all_labels, preds_bin, zero_division=0)
lstm_acc  = accuracy_score(all_labels, preds_bin)

hgb_auc   = roc_auc_score(y_test, hgb_proba_test)
hgb_f1    = f1_score(y_test, hgb_pred_test, zero_division=0)
hgb_acc   = accuracy_score(y_test, hgb_pred_test)

print(f"\n  {'Model':<30} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
print(f"  {'-'*60}")
print(f"  {'HistGradientBoosting':<30} {hgb_acc:>10.4f} {hgb_f1:>10.4f} {hgb_auc:>10.4f}")
print(f"  {'LSTM (PyTorch)':<30} {lstm_acc:>10.4f} {lstm_f1:>10.4f} {lstm_auc:>10.4f}")
print(f"  {'Random Guess (baseline)':<30} {'~0.500':>10} {'~0.500':>10} {'0.500':>10}")
print()
print("NOTE: ~50% accuracy is expected for 5-day direction prediction.")
print("F1 and ROC-AUC are more meaningful than accuracy for this task.")
print("Plots saved: hgb_feature_importance.png  lstm_epoch_plot.png")

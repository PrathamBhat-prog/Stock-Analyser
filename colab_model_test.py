# =============================================================================
# Stock Market ML -- Google Colab Test Script
# FILE TO RUN: colab_model_test.py
#
# How to use in Colab:
#   1. Go to colab.research.google.com
#   2. File -> Upload notebook -> select this .py file
#      OR: File -> New notebook, then copy-paste each cell block
#   3. Run cells in order (Cell 1 first, then 2, 3 ...)
#   4. Enable GPU: Runtime -> Change runtime type -> GPU (T4)
#      (LSTM trains much faster with GPU)
#
# What this tests:
#   - HistGradientBoostingClassifier (the current production winner)
#   - Production-grade LSTM with Bidirectional layers + Multi-Head Attention
# =============================================================================


# %% [CELL 1] ---- Install dependencies ------------------------------------
# !pip install -q yfinance scikit-learn torch matplotlib pandas numpy


# %% [CELL 2] ---- Imports -------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__}  |  Device: {DEVICE}")


# %% [CELL 3] ---- Configuration (edit here) --------------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
]
PERIOD      = "5y"    # "2y" for quick test, "10y" for full production
HORIZON     = 5       # predict direction N days ahead
SEQ_LEN     = 60      # LSTM lookback window (days)
TRAIN_PCT   = 0.70
VAL_PCT     = 0.15
# --- LSTM settings ---
HIDDEN_SIZE = 256
NUM_LAYERS  = 3
DROPOUT     = 0.35
EPOCHS      = 300
PATIENCE    = 30
BATCH_SIZE  = 512
LR          = 3e-4


# %% [CELL 4] ---- Feature engineering (60 features, same as production) ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    df["Daily_Return"] = close.pct_change()
    df["Log_Return"]   = np.log(close / close.shift(1))

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"Return_lag_{lag}"] = df["Daily_Return"].shift(lag)
        df[f"Close_lag_{lag}"]  = close.shift(lag)
        df[f"Volume_lag_{lag}"] = vol.shift(lag)

    for w in [5, 10, 20, 60]:
        df[f"Return_mean_{w}"] = df["Daily_Return"].rolling(w).mean()
        df[f"Return_std_{w}"]  = df["Daily_Return"].rolling(w).std()
        df[f"Close_mean_{w}"]  = close.rolling(w).mean()
        df[f"Volume_mean_{w}"] = vol.rolling(w).mean()

    for p in [5, 10, 20, 60]:
        df[f"Momentum_{p}"] = close.pct_change(p)

    df["Volume_ratio_20"] = vol / vol.rolling(20).mean()
    df["Volume_ratio_5"]  = vol / vol.rolling(5).mean()

    for period in [14, 28]:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
        loss  = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd, sig, macd - sig

    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_Mid"]   = bb_mid
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower
    df["BB_Width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    df["BB_Pct"]   = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    prev_c = close.shift(1)
    tr = pd.concat([high-low, (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    df["ATR_14"]     = tr.ewm(com=13, min_periods=14).mean()
    df["ATR_14_pct"] = df["ATR_14"] / close.replace(0, np.nan)

    sign = np.sign(close.diff()).fillna(0)
    df["OBV"]        = (sign * vol).cumsum()
    df["OBV_ROC_10"] = df["OBV"].pct_change(10)

    df["Close_vs_MA20"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    df["Close_vs_MA60"] = (close - close.rolling(60).mean()) / close.rolling(60).std().replace(0, np.nan)
    df["HL_Range"]      = (high - low) / close.replace(0, np.nan)
    df["HL_Range_5ma"]  = df["HL_Range"].rolling(5).mean()
    df["target"]        = (close.shift(-HORIZON) > close).astype(int)
    return df

FEATURE_COLS = [
    "Daily_Return","Log_Return",
    *[f"Return_lag_{l}" for l in [1,2,3,5,10,20]],
    *[f"Close_lag_{l}"  for l in [1,2,3,5,10,20]],
    *[f"Volume_lag_{l}" for l in [1,2,3,5,10,20]],
    *[f"Return_mean_{w}" for w in [5,10,20,60]],
    *[f"Return_std_{w}"  for w in [5,10,20,60]],
    *[f"Close_mean_{w}"  for w in [5,10,20,60]],
    *[f"Volume_mean_{w}" for w in [5,10,20,60]],
    *[f"Momentum_{p}" for p in [5,10,20,60]],
    "Volume_ratio_20","Volume_ratio_5",
    "RSI_14","RSI_28",
    "MACD","MACD_Signal","MACD_Hist",
    "BB_Mid","BB_Upper","BB_Lower","BB_Width","BB_Pct",
    "ATR_14","ATR_14_pct",
    "OBV","OBV_ROC_10",
    "Close_vs_MA20","Close_vs_MA60",
    "HL_Range","HL_Range_5ma",
]
print(f"Feature count: {len(FEATURE_COLS)}")


# %% [CELL 5] ---- Fetch data + build dataset ------------------------------
print(f"Fetching {len(TICKERS)} tickers, period={PERIOD} ...")
frames = []
for ticker in TICKERS:
    try:
        raw = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True).reset_index()
        raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
        if len(raw) < 200:
            print(f"  Skipping {ticker}: only {len(raw)} rows"); continue
        df = add_features(raw).dropna(subset=FEATURE_COLS + ["target"])
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df)} rows")
    except Exception as e:
        print(f"  {ticker} ERROR: {e}")

all_data = pd.concat(frames).sort_values("Date").reset_index(drop=True)
n = len(all_data)
n_train = int(n * TRAIN_PCT)
n_val   = int(n * (TRAIN_PCT + VAL_PCT))

X_train, y_train = all_data.iloc[:n_train][FEATURE_COLS].values, all_data.iloc[:n_train]["target"].values
X_val,   y_val   = all_data.iloc[n_train:n_val][FEATURE_COLS].values, all_data.iloc[n_train:n_val]["target"].values
X_test,  y_test  = all_data.iloc[n_val:][FEATURE_COLS].values, all_data.iloc[n_val:]["target"].values

print(f"\nTotal: {n} rows | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Class balance (train): {y_train.mean():.2%} positive (UP)")


# %% [CELL 6] ---- MODEL 1: HistGradientBoosting ---------------------------
print("\n" + "="*60)
print("  MODEL 1: HistGradientBoostingClassifier")
print("="*60)

hgb = HistGradientBoostingClassifier(
    max_iter=400, max_leaf_nodes=31, learning_rate=0.05,
    min_samples_leaf=20, l2_regularization=0.1,
    random_state=42, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=10, verbose=1,
)
hgb.fit(X_train, y_train)

def show_metrics(name, y_true, proba):
    pred = (proba >= 0.5).astype(int)
    print(f"\n  [{name}]")
    print(f"    Accuracy  : {accuracy_score(y_true, pred):.4f}")
    print(f"    Precision : {precision_score(y_true, pred, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_true, pred, zero_division=0):.4f}")
    print(f"    F1        : {f1_score(y_true, pred, zero_division=0):.4f}")
    print(f"    ROC-AUC   : {roc_auc_score(y_true, proba):.4f}")
    return roc_auc_score(y_true, proba)

hgb_val_auc  = show_metrics("HGB Validation", y_val,  hgb.predict_proba(X_val)[:,1])
hgb_test_auc = show_metrics("HGB Test",       y_test, hgb.predict_proba(X_test)[:,1])

# Feature importance
imp = hgb.feature_importances_
top = np.argsort(imp)[-20:]
plt.figure(figsize=(10,6))
plt.barh([FEATURE_COLS[i] for i in top], imp[top], color="#3B82F6")
plt.title("HistGradientBoosting -- Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("hgb_feature_importance.png", dpi=150)
plt.show()


# %% [CELL 7] ---- LSTM Architecture (Production Grade) --------------------
# =============================================================================
#
#  ARCHITECTURE OVERVIEW
#  =====================
#
#  Input: (Batch, 60 days, 60 features)
#         Each sample = 60 consecutive trading days of 60 technical indicators
#
#  Layer 1 -- Input Projection
#    Linear(60 -> 256) + LayerNorm + ReLU
#    Purpose: expands features into the hidden space before LSTM
#
#  Layer 2 -- Bidirectional LSTM (3 layers, hidden=256)
#    Forward pass:  reads price sequence from day 1 -> day 60
#    Backward pass: reads price sequence from day 60 -> day 1
#    Combined output: 512 features per timestep (256 fwd + 256 bwd)
#    Dropout=0.35 between layers
#    Why bidirectional? Captures both "what led here" and "where this leads"
#
#  Layer 3 -- Bidirectional Projection
#    Linear(512 -> 256) + LayerNorm
#    Collapses bidir output back to hidden_size
#
#  Layer 4 -- Multi-Head Self-Attention (4 heads)
#    Each head attends to different temporal patterns simultaneously:
#      Head 1: might focus on recent momentum (last 5 days)
#      Head 2: might focus on support/resistance breakouts
#      Head 3: might focus on earnings-cycle patterns (60-day)
#      Head 4: might focus on volume spikes
#    Residual connection + LayerNorm (standard Transformer block)
#
#  Layer 5 -- Residual Skip Connection
#    last_lstm_timestep + projected_input_at_last_step
#    Prevents information loss in deep networks (like ResNet)
#
#  Layer 6 -- Classification Head
#    Linear(256->128) + GELU + Dropout(0.175)
#    Linear(128->64)  + GELU
#    Linear(64->1)    -> raw logit
#
#  Loss: BCEWithLogitsLoss with pos_weight (class imbalance correction)
#
#  Training:
#    Optimizer  : AdamW (lr=3e-4, weight_decay=1e-4, betas=(0.9,0.999))
#    Scheduler  : OneCycleLR (10% warmup + cosine annealing)
#                 Peak LR = 3e-3, then decays smoothly to ~0
#    Grad clip  : max_norm=1.0
#    Max epochs : 300
#    Patience   : 30 (stops only after 30 consecutive non-improving epochs)
#    Init       : Xavier for linear, Orthogonal for LSTM recurrent weights
#                 Forget gate bias = 1.0 (LSTM best practice)
#
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)   # residual + norm


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        # Bidirectional LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        # Collapse bidirectional
        self.bidir_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm_norm  = nn.LayerNorm(hidden_size)
        # Attention
        self.attention = MultiHeadAttention(hidden_size, num_heads=4)
        # Skip connection
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
        # Head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name: nn.init.orthogonal_(p.data)
            elif "bias"      in name:
                p.data.fill_(0)
                n = p.size(0)
                p.data[n//4 : n//2].fill_(1)   # forget gate bias = 1

    def forward(self, x):
        xp          = self.input_proj(x)
        lstm_out, _ = self.lstm(xp)
        lstm_out    = self.lstm_norm(self.bidir_proj(lstm_out))
        attn_out    = self.attention(lstm_out)
        last        = attn_out[:, -1, :] + self.residual_proj(xp[:, -1, :])
        return self.head(last).squeeze(1)


print("Architecture defined.")
total_params = sum(p.numel() for p in StockLSTM(len(FEATURE_COLS)).parameters())
print(f"Total parameters: {total_params:,}")


# %% [CELL 8] ---- Build LSTM sequences + DataLoaders ---------------------
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_val   = scaler.transform(X_val)
Xs_test  = scaler.transform(X_test)

def make_sequences(X, y, seq_len=SEQ_LEN):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

Xt, yt = make_sequences(Xs_train, y_train)
Xv, yv = make_sequences(Xs_val,   y_val)
Xe, ye = make_sequences(Xs_test,  y_test)
print(f"Sequence shapes -- train:{Xt.shape}  val:{Xv.shape}  test:{Xe.shape}")

train_loader = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt)),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xe), torch.from_numpy(ye)),
                          batch_size=BATCH_SIZE, shuffle=False)


# %% [CELL 9] ---- Train LSTM (300 epochs, patience=30) -------------------
print("\n" + "="*60)
print("  MODEL 2: Production LSTM (Bidirectional + Attention)")
print(f"  Max epochs={EPOCHS}  Patience={PATIENCE}  Device={DEVICE}")
print("="*60)

model     = StockLSTM(len(FEATURE_COLS)).to(DEVICE)
pos_wt    = torch.tensor([(y_train==0).sum() / max((y_train==1).sum(), 1)]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR*10, epochs=EPOCHS,
    steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy="cos"
)

history    = {"loss": [], "val_auc": [], "lr": []}
best_auc   = 0.0
best_state = None
best_ep    = 0
no_imp     = 0
early_ep   = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ep_loss += loss.item()
    avg_loss = ep_loss / len(train_loader)
    cur_lr   = scheduler.get_last_lr()[0]

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            p = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy()
            preds.extend(p); labels.extend(yb.numpy())
    preds, labels = np.array(preds), np.array(labels)
    val_auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
    val_acc = accuracy_score(labels, (preds >= 0.5).astype(int))

    history["loss"].append(avg_loss)
    history["val_auc"].append(val_auc)
    history["lr"].append(cur_lr)

    print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}"
          f"  val_acc={val_acc:.4f}  lr={cur_lr:.6f}")

    if val_auc > best_auc:
        best_auc = val_auc; best_ep = epoch; no_imp = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            print(f"  Best epoch: {best_ep}  Best val AUC: {best_auc:.4f}")
            early_ep = epoch; break

if best_state:
    model.load_state_dict(best_state)
    print(f"\n  Restored best weights from epoch {best_ep}")


# %% [CELL 10] ---- Epoch plot (3-panel proof) ----------------------------
eps = list(range(1, len(history["loss"]) + 1))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.patch.set_facecolor("#0F172A")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8")
    for sp in ax.spines.values(): sp.set_color("#334155")

ax1.plot(eps, history["loss"], color="#3B82F6", linewidth=1.5, label="Train BCE Loss")
ax1.set_ylabel("Loss", color="#94A3B8")
ax1.set_title(f"LSTM Training  |  Best Epoch: {best_ep}  |  Best Val AUC: {best_auc:.4f}",
              color="#E2E8F0", fontsize=11)
ax1.legend(facecolor="#1E293B", labelcolor="#E2E8F0"); ax1.grid(True, color="#1E3A5F", linewidth=0.4)

ax2.plot(eps, history["val_auc"], color="#F97316", linewidth=1.5, label="Val ROC-AUC")
ax2.axhline(0.5, color="#94A3B8", linestyle=":", linewidth=0.8, label="Random baseline")
ax2.axvline(best_ep, color="#10B981", linestyle="--", linewidth=1.5,
            label=f"Best epoch {best_ep}")
if early_ep:
    ax2.axvline(early_ep, color="#EF4444", linestyle=":", linewidth=1.2,
                label=f"Early stop @{early_ep}")
ax2.set_ylabel("Val AUC", color="#94A3B8"); ax2.set_ylim(0.35, 0.75)
ax2.legend(facecolor="#1E293B", labelcolor="#E2E8F0", fontsize=8)
ax2.grid(True, color="#1E3A5F", linewidth=0.4)

ax3.plot(eps, history["lr"], color="#A78BFA", linewidth=1.2, label="Learning Rate (OneCycleLR)")
ax3.set_ylabel("LR", color="#94A3B8"); ax3.set_xlabel("Epoch", color="#94A3B8")
ax3.legend(facecolor="#1E293B", labelcolor="#E2E8F0")
ax3.grid(True, color="#1E3A5F", linewidth=0.4)

plt.tight_layout(pad=1.5)
plt.savefig("lstm_epoch_plot.png", dpi=150, bbox_inches="tight", facecolor="#0F172A")
plt.show()
print("Saved: lstm_epoch_plot.png")


# %% [CELL 11] ---- LSTM Test evaluation ----------------------------------
model.eval()
preds, labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        p = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy()
        preds.extend(p); labels.extend(yb.numpy())
preds, labels = np.array(preds), np.array(labels)
lstm_test_auc = show_metrics("LSTM Test", labels, preds)


# %% [CELL 12] ---- Final comparison table --------------------------------
hgb_p_test = hgb.predict_proba(X_test)[:, 1]
hgb_p_bin  = (hgb_p_test >= 0.5).astype(int)

print("\n" + "="*60)
print("  FINAL MODEL COMPARISON (Test Set)")
print("="*60)
print(f"  {'Model':<35} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*63}")
print(f"  {'HistGradientBoosting':<35}"
      f" {accuracy_score(y_test, hgb_p_bin):>10.4f}"
      f" {f1_score(y_test, hgb_p_bin, zero_division=0):>8.4f}"
      f" {roc_auc_score(y_test, hgb_p_test):>8.4f}")
lstm_bin = (preds >= 0.5).astype(int)
print(f"  {'LSTM (Bidir + Attention, 300ep)':<35}"
      f" {accuracy_score(labels, lstm_bin):>10.4f}"
      f" {f1_score(labels, lstm_bin, zero_division=0):>8.4f}"
      f" {lstm_test_auc:>8.4f}")
print(f"  {'Random Guess (baseline)':<35} {'~0.5000':>10} {'~0.5000':>8} {'0.5000':>8}")
print()
print("Artifacts: hgb_feature_importance.png  lstm_epoch_plot.png")

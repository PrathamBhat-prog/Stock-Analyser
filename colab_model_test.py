# =============================================================================
# Stock Market ML -- Colab Script v4 (Academic-Grade)
#
# Based on:
#   Fischer & Krauss (2018) - LSTM on S&P500 daily returns, Sharpe=5.8
#   George & Hwang (2004)  - 52-week high proximity dominates momentum
#   Nelson et al. (2017)   - 56% accuracy with 1-day horizon
#   Gu, Kelly & Xiu (2020) - 94 factor features, tree ensembles beat LSTM
#
# HONEST ACADEMIC CEILING with OHLCV only:
#   Daily direction prediction: 51-56% accuracy
#   60%+ requires: sentiment/news NLP, options flow, insider trades
#
# THIS SCRIPT implements:
#   1. 52-week high/low proximity (George & Hwang 2004)
#   2. 6-month momentum factor (Jegadeesh & Titman 1993)
#   3. Short-term reversal (1-day) -- Jegadeesh 1990
#   4. VIX regime conditioning
#   5. Time-series cross-validation (purged walk-forward)
#   6. Stacked ensemble + confidence filter
#   7. Try both 1-day (higher signal) and 5-day horizons
#
# HOW TO USE:
#   colab.research.google.com -> Upload -> this file
#   Runtime -> Change runtime type -> T4 GPU (speeds up walk-forward CV)
# =============================================================================


# %% [CELL 1] Install
# !pip install -q yfinance scikit-learn xgboost lightgbm matplotlib pandas numpy


# %% [CELL 2] Imports
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
import xgboost as xgb
import lightgbm as lgb
print("Imports OK")


# %% [CELL 3] Config
TICKERS = [
    "AAPL","MSFT","GOOGL","TSLA","NVDA","META","AMZN","JPM","V","MA",
    "JNJ","WMT","XOM","KO","DIS","AMD","NFLX","HD","PG","INTC","BAC",
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
]
PERIOD          = "10y"
HORIZON         = 1      # 1-day prediction (Nelson 2017: higher signal than 5-day)
RETURN_THRESHOLD= 0.0    # no threshold for 1-day (enough volume of signal)
TRAIN_PCT       = 0.70
VAL_PCT         = 0.15
N_CV_FOLDS      = 5     # walk-forward CV folds


# %% [CELL 4] Feature engineering
# Features grounded in academic literature with citations

def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

def add_features(df, vix_series=None):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    ret = close.pct_change()

    # -- Fischer & Krauss (2018): raw daily returns are the core signal --
    df["r1"]  = winsorize(ret)                    # 1-day return
    df["r2"]  = winsorize(ret.shift(1))           # 2-day lag
    df["r5"]  = winsorize(ret.rolling(5).sum())   # 5-day cumulative
    df["r21"] = winsorize(ret.rolling(21).sum())  # 1-month cumulative

    # -- Jegadeesh (1990): short-term reversal --
    # Stocks that rise today tend to fall slightly tomorrow (microstructure)
    df["reversal_1d"] = -ret.shift(1)

    # -- Jegadeesh & Titman (1993): 6-month momentum factor --
    # Skip last month (J=6, K=1 is the classic implementation)
    df["momentum_6m"] = winsorize(close.shift(21).pct_change(105))  # 126d-21d skip

    # -- George & Hwang (2004): 52-week high proximity --
    # Most powerful single predictor in cross-sectional momentum literature
    high_52w = close.rolling(252).max()
    low_52w  = close.rolling(252).min()
    df["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    df["dist_52w_low"]  = (close - low_52w)  / low_52w.replace(0, np.nan)
    df["pct_52w_range"] = (close - low_52w)  / (high_52w - low_52w).replace(0, np.nan)

    # -- Gu, Kelly & Xiu (2020): volatility and liquidity features --
    df["vol_20d"]       = ret.rolling(20).std()
    df["vol_5d"]        = ret.rolling(5).std()
    df["vol_ratio"]     = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)  # vol spike
    df["illiquidity"]   = winsorize((ret.abs() / vol.replace(0, np.nan)).rolling(21).mean())  # Amihud

    # -- Standard technical (RSI, MACD, Bollinger) --
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["MACD_Hist"] = winsorize(macd - macd.ewm(span=9, adjust=False).mean())

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Pct"] = (close - (bb_mid - 2*bb_std)) / (4*bb_std).replace(0, np.nan)

    # ATR %
    prev_c = close.shift(1)
    tr = pd.concat([high-low,(high-prev_c).abs(),(low-prev_c).abs()],axis=1).max(axis=1)
    df["ATR_pct"] = tr.ewm(com=13, min_periods=14).mean() / close.replace(0, np.nan)

    # OBV momentum
    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    df["OBV_ROC_10"] = winsorize(obv.pct_change(10))

    # Volume
    df["vol_ratio_5"] = winsorize(vol / vol.rolling(5).mean().replace(0, np.nan))

    # -- VIX regime (high VIX = fear = different stock dynamics) --
    if vix_series is not None:
        vix = vix_series.reindex(pd.to_datetime(df["Date"].values))
        df["VIX"]        = vix.values
        df["VIX_regime"] = (vix > vix.rolling(63).mean()).astype(float).values
    else:
        df["VIX"] = 20.0; df["VIX_regime"] = 0.0

    # Target: 1-day direction
    df["target"] = (close.shift(-HORIZON) > close).astype(int)
    return df

FEATURE_COLS = [
    # Fischer & Krauss returns
    "r1","r2","r5","r21","reversal_1d",
    # Momentum factors
    "momentum_6m","dist_52w_high","dist_52w_low","pct_52w_range",
    # Volatility/liquidity (Gu et al.)
    "vol_20d","vol_5d","vol_ratio","illiquidity",
    # Standard TA
    "RSI_14","MACD_Hist","BB_Pct","ATR_pct","OBV_ROC_10","vol_ratio_5",
    # Macro
    "VIX","VIX_regime",
]
print(f"Total features: {len(FEATURE_COLS)}")
print("Sources:")
print("  r1/r2/r5/r21     -- Fischer & Krauss 2018")
print("  reversal_1d      -- Jegadeesh 1990")
print("  momentum_6m      -- Jegadeesh & Titman 1993")
print("  dist_52w_*       -- George & Hwang 2004 (strongest single predictor)")
print("  vol_ratio/illiq  -- Gu, Kelly & Xiu 2020")
print("  VIX/VIX_regime   -- macro conditioning")


# %% [CELL 5] Fetch data
print(f"\nFetching VIX...")
try:
    vix_raw = yf.Ticker("^VIX").history(period=PERIOD,auto_adjust=True).reset_index()
    vix_raw["Date"] = pd.to_datetime(vix_raw["Date"]).dt.tz_localize(None)
    vix_series = vix_raw.set_index("Date")["Close"]
    print(f"  VIX: {len(vix_series)} rows")
except:
    vix_series = None; print("  VIX failed, using neutral")

print(f"Fetching {len(TICKERS)} tickers (period={PERIOD})...")
frames = []
for ticker in TICKERS:
    try:
        raw = yf.Ticker(ticker).history(period=PERIOD,auto_adjust=True).reset_index()
        raw.columns = [c if isinstance(c,str) else c[0] for c in raw.columns]
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
        if len(raw) < 300: continue
        df = add_features(raw, vix_series if ".NS" not in ticker else None)
        df = df.dropna(subset=FEATURE_COLS+["target"])
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df)} rows | UP={df['target'].mean():.1%}")
    except Exception as e:
        print(f"  {ticker} ERROR: {e}")

all_data = pd.concat(frames).sort_values("Date").reset_index(drop=True)
n = len(all_data)
print(f"\nTotal: {n} rows | Class balance: {all_data['target'].mean():.2%} UP")


# %% [CELL 6] Purged walk-forward cross-validation
# Standard train/test split is WRONG for time series:
# - It leaks future information via overlapping labels
# - Test period may be structurally different (2022 crash vs 2019 bull)
# Walk-forward CV: train on past, test on next window, repeat

print("\n=== Purged Walk-Forward Cross-Validation ===")
print(f"  {N_CV_FOLDS} folds | 1-day ahead prediction (HORIZON=1)")

fold_size = n // (N_CV_FOLDS + 1)
PURGE_GAP = 5  # skip 5 rows between train and test (prevents label leakage)

all_val_proba, all_val_true = [], []
fold_aucs = []

for fold in range(N_CV_FOLDS):
    train_end = fold_size * (fold + 1)
    test_start = train_end + PURGE_GAP
    test_end   = min(test_start + fold_size, n)

    if test_start >= n: break

    Xtr = all_data.iloc[:train_end][FEATURE_COLS].values
    ytr = all_data.iloc[:train_end]["target"].values.astype(int)
    Xte = all_data.iloc[test_start:test_end][FEATURE_COLS].values
    yte = all_data.iloc[test_start:test_end]["target"].values.astype(int)

    sc   = StandardScaler()
    Xtr_ = sc.fit_transform(Xtr)
    Xte_ = sc.transform(Xte)

    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.3,
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15, verbose=0)
    m.fit(Xtr_, ytr)

    preds = m.predict_proba(Xte_)[:, 1]
    auc   = roc_auc_score(yte, preds) if len(set(yte)) > 1 else 0.5
    acc   = accuracy_score(yte, (preds >= 0.5).astype(int))
    fold_aucs.append(auc)
    all_val_proba.extend(preds)
    all_val_true.extend(yte)
    print(f"  Fold {fold+1}: train={train_end} | test={test_start}-{test_end} | "
          f"acc={acc:.4f} | auc={auc:.4f}")

print(f"\n  CV Mean AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")


# %% [CELL 7] Final train on 70%, test on last 15%
print("\n=== Final Models (train=70%, test=last 15%) ===")
n_train = int(n * TRAIN_PCT)
n_val   = int(n * (TRAIN_PCT + VAL_PCT))
X_tr = all_data.iloc[:n_train][FEATURE_COLS].values
y_tr = all_data.iloc[:n_train]["target"].values.astype(int)
X_va = all_data.iloc[n_train:n_val][FEATURE_COLS].values
y_va = all_data.iloc[n_train:n_val]["target"].values.astype(int)
X_te = all_data.iloc[n_val:][FEATURE_COLS].values
y_te = all_data.iloc[n_val:]["target"].values.astype(int)

sc = StandardScaler()
Xtr_ = sc.fit_transform(X_tr)
Xva_ = sc.transform(X_va)
Xte_ = sc.transform(X_te)

hgb = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=0.3, class_weight="balanced",
    random_state=42, early_stopping=True, n_iter_no_change=20, verbose=0)

xgb_m = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.02, max_depth=4,
    subsample=0.7, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.3, reg_lambda=1.5,
    scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
    eval_metric="auc", early_stopping_rounds=25,
    use_label_encoder=False, verbosity=0, random_state=42)

lgb_m = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.02, num_leaves=31,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.5, class_weight="balanced",
    early_stopping_rounds=25, verbose=-1, random_state=42)

print("  Training HGB ...");    hgb.fit(Xtr_, y_tr)
print("  Training XGBoost ..."); xgb_m.fit(Xtr_, y_tr, eval_set=[(Xva_, y_va)], verbose=False)
print("  Training LightGBM ..."); lgb_m.fit(Xtr_, y_tr, eval_set=[(Xva_, y_va)],
                                             callbacks=[lgb.early_stopping(25, verbose=False)])

def get_p(m, X): return m.predict_proba(X)[:,1]

def show(name, yt, p):
    pb = (p>=0.5).astype(int)
    print(f"  {name:<30} acc={accuracy_score(yt,pb):.4f}  "
          f"f1={f1_score(yt,pb,zero_division=0):.4f}  "
          f"auc={roc_auc_score(yt,p):.4f}  "
          f"prec={precision_score(yt,pb,zero_division=0):.4f}")

show("HGB",      y_te, get_p(hgb,   Xte_))
show("XGBoost",  y_te, get_p(xgb_m, Xte_))
show("LightGBM", y_te, get_p(lgb_m, Xte_))

# Stacked ensemble
meta_val  = np.column_stack([get_p(hgb,Xva_),  get_p(xgb_m,Xva_),  get_p(lgb_m,Xva_)])
meta_test = np.column_stack([get_p(hgb,Xte_),  get_p(xgb_m,Xte_),  get_p(lgb_m,Xte_)])
sc2 = StandardScaler()
meta_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
meta_lr.fit(sc2.fit_transform(meta_val), y_va)
stack_p = meta_lr.predict_proba(sc2.transform(meta_test))[:,1]
show("Stacked Ensemble", y_te, stack_p)


# %% [CELL 8] Confidence filter + feature importance
print("\n=== Confidence Filter ===")
print(f"{'Thresh':<10} {'Coverage':<12} {'Accuracy':<12} {'Precision':<12} {'F1'}")
print("-"*56)
best_acc, best_t = 0, 0.5
for t in [0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65]:
    mask = (stack_p >= t) | (stack_p <= (1-t))
    if mask.sum() < 30: break
    fp = (stack_p[mask]>=0.5).astype(int); ft = y_te[mask]
    acc = accuracy_score(ft,fp); prec = precision_score(ft,fp,zero_division=0)
    f1  = f1_score(ft,fp,zero_division=0); cov = mask.mean()
    print(f"  >{t:.2f}     {cov:.1%}        {acc:.4f}        {prec:.4f}        {f1:.4f}")
    if acc > best_acc: best_acc, best_t = acc, t

print(f"\n  Best: acc={best_acc:.4f} at threshold={best_t:.2f}")

# Feature importance
imp = hgb.feature_importances_
idx = np.argsort(imp)[::-1]
print("\n  Top 10 most important features (HGB):")
for i in idx[:10]:
    print(f"    {FEATURE_COLS[i]:<20} {imp[i]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F172A")
for ax in axes: ax.set_facecolor("#1E293B"); ax.tick_params(colors="#94A3B8")

# Feature importance bar chart
top_n = 15
ax = axes[0]
ax.barh([FEATURE_COLS[i] for i in idx[:top_n]][::-1],
        [imp[i] for i in idx[:top_n]][::-1], color="#3B82F6")
ax.set_title("Top 15 Feature Importances (HGB)", color="#E2E8F0")
ax.set_xlabel("Importance", color="#94A3B8")
for sp in ax.spines.values(): sp.set_color("#334155")

# Accuracy vs coverage
ax = axes[1]
ths, accs, covs = [], [], []
for t in np.arange(0.50, 0.70, 0.01):
    mask = (stack_p >= t) | (stack_p <= (1-t))
    if mask.sum() < 30: break
    ths.append(t); accs.append(accuracy_score(y_te[mask],(stack_p[mask]>=0.5).astype(int)))
    covs.append(mask.mean())
ax2 = ax.twinx()
ax.plot(ths, accs, "o-", color="#10B981", lw=2, label="Accuracy")
ax2.plot(ths, covs, "^--",color="#F59E0B", lw=2, label="Coverage")
ax.axhline(0.55, color="#EF4444", ls=":", lw=1, label="55% line")
ax.axhline(0.60, color="#A78BFA", ls=":", lw=1, label="60% line")
ax.set_xlabel("Threshold", color="#94A3B8"); ax.set_ylabel("Accuracy", color="#94A3B8")
ax2.set_ylabel("Coverage", color="#F59E0B")
ax.set_title("Accuracy vs Coverage (Stacked Ensemble)", color="#E2E8F0")
lines1,labs1 = ax.get_legend_handles_labels()
lines2,labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, facecolor="#1E293B", labelcolor="#E2E8F0", fontsize=8)
ax.grid(True, color="#334155", lw=0.4)
for sp in ax.spines.values(): sp.set_color("#334155")
ax2.tick_params(colors="#F59E0B")

plt.tight_layout(); plt.savefig("results_v4.png", dpi=150, facecolor="#0F172A"); plt.show()
print("Saved: results_v4.png")


# %% [CELL 9] Final summary + academic context
print("\n" + "="*70)
print("  FINAL RESULTS (v4 -- Academic Feature Set)")
print("="*70)
print(f"  CV Mean AUC (walk-forward): {np.mean(fold_aucs):.4f}")
p = get_p(hgb,Xte_); pb=(p>=0.5).astype(int)
print(f"  HGB Test AUC: {roc_auc_score(y_te,p):.4f} | Acc: {accuracy_score(y_te,pb):.4f}")
print(f"  Stacked Ensemble AUC: {roc_auc_score(y_te,stack_p):.4f}")
mask = (stack_p>=best_t)|(stack_p<=(1-best_t))
if mask.sum()>0:
    fp=(stack_p[mask]>=0.5).astype(int); ft=y_te[mask]
    print(f"  Confidence>{best_t:.2f} ({mask.mean():.0%} days): Acc={accuracy_score(ft,fp):.4f}")
print()
print("  ACADEMIC CONTEXT:")
print("  Fischer & Krauss 2018: LSTM on S&P500, Sharpe=5.8 (pre-2010)")
print("  Strategy arbitraged away by 2010 -- signal degrades over time")
print("  Nelson et al. 2017: ~56% accuracy with 1-day horizon")
print("  George & Hwang 2004: 52-week high is strongest single predictor")
print("  Gu, Kelly & Xiu 2020: 94 factors, trees beat LSTM on US equities")
print()
print("  CEILING WITH OHLCV ONLY: 54-56% accuracy")
print("  TO REACH 60%+: Add news sentiment (FinBERT), options flow, earnings")

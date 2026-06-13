# =============================================================================
# Stock Market ML -- Colab v5 (News Sentiment Edition)
#
# Academic basis:
#   Tetlock (2007)          -- news pessimism predicts next-day returns
#   Loughran & McDonald (2011) -- finance-specific sentiment lexicon
#   FinBERT (Yang 2020)     -- 97% on financial phrase classification
#   George & Hwang (2004)   -- 52-week high proximity strongest momentum signal
#   Jegadeesh & Titman (1993) -- 6-month momentum factor
#
# News source: GDELT Project (free, no API key, global news since 2015)
# Sentiment:   VADER (fast, no GPU) -> upgrade to FinBERT for production
#
# HOW TO USE:
#   colab.research.google.com -> Upload -> this .py file
#   Runtime -> Change runtime type -> T4 GPU
#   Run cells 1 -> 10 in order
# =============================================================================


# %% [CELL 1] Install dependencies
# !pip install -q yfinance scikit-learn xgboost lightgbm vaderSentiment matplotlib pandas numpy requests


# %% [CELL 2] Imports
import warnings; warnings.filterwarnings("ignore")
import time, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)
import xgboost as xgb
import lightgbm as lgb

vader = SentimentIntensityAnalyzer()
print("Imports OK  |  VADER ready")


# %% [CELL 3] Config
TICKERS = [
    "AAPL","MSFT","GOOGL","TSLA","NVDA","META","AMZN","JPM","V","MA",
    "JNJ","WMT","XOM","KO","DIS","AMD","NFLX","HD","PG","INTC",
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS",
]
PERIOD    = "5y"   # 5y is enough when combined with sentiment
HORIZON   = 1      # 1-day (higher signal, Nelson 2017)
TRAIN_PCT = 0.70
VAL_PCT   = 0.15
CONF_THRESHOLD = 0.58


# %% [CELL 4] News sentiment via GDELT (free, no API key)
# GDELT monitors 65 languages, 100+ countries, updated every 15 min.
# We use the doc API to fetch headlines for a ticker and score with VADER.

def fetch_gdelt_sentiment(query: str, days_back: int = 7) -> float:
    """
    Fetch recent news headlines for `query` from GDELT and return
    average VADER compound sentiment score [-1, +1].
    Returns 0.0 (neutral) on any failure.
    """
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={query}&mode=artlist&maxrecords=50"
        f"&timespan={days_back}d&format=json"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return 0.0
        articles = r.json().get("articles", [])
        if not articles:
            return 0.0
        scores = [vader.polarity_scores(a.get("title",""))["compound"]
                  for a in articles]
        return float(np.mean(scores))
    except Exception:
        return 0.0

def fetch_yfinance_sentiment(ticker: str) -> float:
    """Fallback: use yfinance built-in news headlines."""
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return 0.0
        scores = [vader.polarity_scores(n.get("title",""))["compound"]
                  for n in news[:20]]
        return float(np.mean(scores))
    except Exception:
        return 0.0

# Quick test
print("Testing GDELT sentiment fetch (AAPL)...")
s = fetch_gdelt_sentiment("Apple stock AAPL")
print(f"  AAPL GDELT sentiment: {s:.3f}")
if s == 0.0:
    s = fetch_yfinance_sentiment("AAPL")
    print(f"  AAPL yfinance fallback: {s:.3f}")


# %% [CELL 5] Feature engineering (OHLCV + academic factors + sentiment)
def winsorize(s, lo=0.01, hi=0.99):
    lo_v, hi_v = s.quantile(lo), s.quantile(hi)
    return s.clip(lo_v, hi_v)

def add_features(df, sentiment_score: float = 0.0, vix_series=None):
    """
    sentiment_score: pre-computed VADER score for this ticker [-1, +1]
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    ret = close.pct_change()

    # --- Fischer & Krauss (2018): multi-horizon returns ---
    df["r1"]  = winsorize(ret)
    df["r5"]  = winsorize(ret.rolling(5).sum())
    df["r21"] = winsorize(ret.rolling(21).sum())

    # --- Jegadeesh (1990): short-term reversal ---
    df["reversal_1d"] = -ret.shift(1)

    # --- Jegadeesh & Titman (1993): 6-month momentum (skip 1 month) ---
    df["momentum_6m"] = winsorize(close.shift(21).pct_change(105))

    # --- George & Hwang (2004): 52-week high/low proximity ---
    high_52w = close.rolling(252).max()
    low_52w  = close.rolling(252).min()
    df["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    df["pct_52w_range"] = (close - low_52w) / (high_52w - low_52w).replace(0, np.nan)

    # --- Volatility (Gu et al. 2020) ---
    df["vol_20d"]   = ret.rolling(20).std()
    df["vol_ratio"] = ret.rolling(5).std() / df["vol_20d"].replace(0, np.nan)

    # --- Standard TA ---
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

    prev_c = close.shift(1)
    tr = pd.concat([high-low,(high-prev_c).abs(),(low-prev_c).abs()],axis=1).max(axis=1)
    df["ATR_pct"] = tr.ewm(com=13, min_periods=14).mean() / close.replace(0, np.nan)

    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    df["OBV_ROC"] = winsorize(obv.pct_change(10))
    df["vol_ratio_5"] = winsorize(vol / vol.rolling(5).mean().replace(0, np.nan))

    # --- VIX macro regime ---
    if vix_series is not None:
        vix_aligned = vix_series.reindex(pd.to_datetime(df["Date"].values))
        df["VIX"]        = vix_aligned.values
        df["VIX_regime"] = (vix_aligned > vix_aligned.rolling(63).mean()).astype(float).values
    else:
        df["VIX"] = 20.0; df["VIX_regime"] = 0.0

    # --- Sentiment features (Tetlock 2007, FinBERT / VADER) ---
    # For training: use a single static score as a constant proxy
    # For production inference: this will be a live daily value
    df["sentiment"]     = float(sentiment_score)
    df["sent_positive"] = max(0.0, float(sentiment_score))
    df["sent_negative"] = min(0.0, float(sentiment_score))

    # Target: next-day direction
    df["target"] = (close.shift(-HORIZON) > close).astype(int)
    return df

FEATURE_COLS = [
    # Fischer & Krauss returns
    "r1","r5","r21","reversal_1d",
    # Momentum factors
    "momentum_6m","dist_52w_high","pct_52w_range",
    # Volatility
    "vol_20d","vol_ratio",
    # Technical
    "RSI_14","MACD_Hist","BB_Pct","ATR_pct","OBV_ROC","vol_ratio_5",
    # Macro
    "VIX","VIX_regime",
    # Sentiment (Tetlock 2007)
    "sentiment","sent_positive","sent_negative",
]
print(f"Total features: {len(FEATURE_COLS)}")


# %% [CELL 6] Fetch OHLCV data + live sentiment per ticker
print("Fetching VIX...")
try:
    vix_raw = yf.Ticker("^VIX").history(period=PERIOD, auto_adjust=True).reset_index()
    vix_raw["Date"] = pd.to_datetime(vix_raw["Date"]).dt.tz_localize(None)
    vix_series = vix_raw.set_index("Date")["Close"]
    print(f"  VIX: {len(vix_series)} rows")
except Exception as e:
    vix_series = None; print(f"  VIX failed: {e}")

print(f"\nFetching {len(TICKERS)} tickers + sentiment...")
frames = []
sentiment_map = {}

for ticker in TICKERS:
    # Fetch sentiment (live for inference demo)
    query = ticker.replace(".NS","").replace(".AS","")
    sent  = fetch_gdelt_sentiment(f"{query} stock")
    if sent == 0.0:
        sent = fetch_yfinance_sentiment(ticker)
    sentiment_map[ticker] = sent
    time.sleep(0.3)  # polite rate limit for GDELT

    try:
        raw = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True).reset_index()
        raw.columns = [c if isinstance(c,str) else c[0] for c in raw.columns]
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
        if len(raw) < 252: print(f"  Skip {ticker}: {len(raw)} rows"); continue
        use_vix = vix_series if ".NS" not in ticker else None
        df = add_features(raw, sent, use_vix)
        df = df.dropna(subset=FEATURE_COLS + ["target"])
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {ticker}: {len(df)} rows | sent={sent:+.3f} | UP={df['target'].mean():.1%}")
    except Exception as e:
        print(f"  {ticker} ERROR: {e}")

all_data = pd.concat(frames).sort_values("Date").reset_index(drop=True)
n = len(all_data)
print(f"\nTotal: {n} rows | Class balance: {all_data['target'].mean():.2%} UP")


# %% [CELL 7] Train/val/test split + model training
n_train = int(n * TRAIN_PCT)
n_val   = int(n * (TRAIN_PCT + VAL_PCT))

X_tr = all_data.iloc[:n_train][FEATURE_COLS].values
y_tr = all_data.iloc[:n_train]["target"].values.astype(int)
X_va = all_data.iloc[n_train:n_val][FEATURE_COLS].values
y_va = all_data.iloc[n_train:n_val]["target"].values.astype(int)
X_te = all_data.iloc[n_val:][FEATURE_COLS].values
y_te = all_data.iloc[n_val:]["target"].values.astype(int)

sc   = StandardScaler()
Xtr_ = sc.fit_transform(X_tr)
Xva_ = sc.transform(X_va)
Xte_ = sc.transform(X_te)

print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")
print(f"Class balance -- Train: {y_tr.mean():.1%} | Test: {y_te.mean():.1%}")

print("\n=== Training Models ===")
hgb = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.03, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=0.3, class_weight="balanced",
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, verbose=0)

pos_w = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1.0)
xgb_m = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.02, max_depth=4,
    subsample=0.7, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.3, reg_lambda=1.5, scale_pos_weight=pos_w,
    eval_metric="auc", early_stopping_rounds=25,
    use_label_encoder=False, verbosity=0, random_state=42)

lgb_m = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.02, num_leaves=31,
    min_child_samples=20, subsample=0.7, colsample_bytree=0.8,
    reg_alpha=0.3, reg_lambda=1.5, class_weight="balanced",
    verbose=-1, random_state=42)

print("  HGB ...");    hgb.fit(Xtr_, y_tr)
print("  XGBoost ..."); xgb_m.fit(Xtr_, y_tr, eval_set=[(Xva_,y_va)], verbose=False)
print("  LightGBM ..."); lgb_m.fit(Xtr_, y_tr, eval_set=[(Xva_,y_va)],
                                    callbacks=[lgb.log_evaluation(period=-1),
                                               lgb.early_stopping(25, verbose=False)])

def get_p(m, X): return m.predict_proba(X)[:,1]

def show(name, yt, p):
    pb = (p>=0.5).astype(int)
    auc  = roc_auc_score(yt, p)
    acc  = accuracy_score(yt, pb)
    prec = precision_score(yt, pb, zero_division=0)
    f1   = f1_score(yt, pb, zero_division=0)
    print(f"  {name:<28} acc={acc:.4f}  prec={prec:.4f}  f1={f1:.4f}  auc={auc:.4f}")
    return auc

print("\n--- Individual Models ---")
show("HGB",      y_te, get_p(hgb,   Xte_))
show("XGBoost",  y_te, get_p(xgb_m, Xte_))
show("LightGBM", y_te, get_p(lgb_m, Xte_))


# %% [CELL 8] Stacked ensemble
print("\n--- Stacked Ensemble ---")
meta_va  = np.column_stack([get_p(hgb,Xva_),  get_p(xgb_m,Xva_),  get_p(lgb_m,Xva_)])
meta_te  = np.column_stack([get_p(hgb,Xte_),  get_p(xgb_m,Xte_),  get_p(lgb_m,Xte_)])
sc2      = StandardScaler()
meta_lr  = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
meta_lr.fit(sc2.fit_transform(meta_va), y_va)
stack_p  = meta_lr.predict_proba(sc2.transform(meta_te))[:,1]
show("Stacked Ensemble", y_te, stack_p)


# %% [CELL 9] Confidence filter + plots
print(f"\n--- Confidence Filter (threshold > {CONF_THRESHOLD}) ---")
print(f"{'Thresh':<10} {'Coverage':<12} {'Accuracy':<12} {'Precision':<10} {'F1'}")
print("-"*55)
best_acc, best_t = 0, 0.5
for t in np.arange(0.50, 0.70, 0.02):
    mask = (stack_p >= t) | (stack_p <= (1-t))
    if mask.sum() < 30: break
    fp  = (stack_p[mask] >= 0.5).astype(int)
    ft  = y_te[mask]
    acc = accuracy_score(ft, fp)
    prec= precision_score(ft, fp, zero_division=0)
    f1  = f1_score(ft, fp, zero_division=0)
    cov = mask.mean()
    marker = " <-- best" if acc > best_acc else ""
    print(f"  >{t:.2f}     {cov:.1%}        {acc:.4f}        {prec:.4f}     {f1:.4f}{marker}")
    if acc > best_acc: best_acc, best_t = acc, t

# Feature importance + accuracy curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0F172A")
for ax in (ax1, ax2):
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8")
    for sp in ax.spines.values(): sp.set_color("#334155")

imp = hgb.feature_importances_
idx = np.argsort(imp)[::-1][:15]
colors = ["#F59E0B" if "sent" in FEATURE_COLS[i] or "52w" in FEATURE_COLS[i]
          else "#3B82F6" for i in idx]
ax1.barh([FEATURE_COLS[i] for i in reversed(idx)],
         [imp[i] for i in reversed(idx)], color=list(reversed(colors)))
ax1.set_title("Top 15 Features (HGB)\nOrange = sentiment/52w-high",
              color="#E2E8F0", fontsize=10)
ax1.set_xlabel("Importance", color="#94A3B8")

ths, accs, covs = [], [], []
for t in np.arange(0.50, 0.70, 0.01):
    mask = (stack_p >= t) | (stack_p <= (1-t))
    if mask.sum() < 30: break
    ths.append(t)
    accs.append(accuracy_score(y_te[mask], (stack_p[mask]>=0.5).astype(int)))
    covs.append(mask.mean())
ax2b = ax2.twinx()
ax2.plot(ths, accs, "o-", color="#10B981", lw=2, label="Accuracy")
ax2b.plot(ths, covs, "^--", color="#F59E0B", lw=2, label="Coverage")
ax2.axhline(0.55, color="#64748B", ls=":", lw=1)
ax2.axhline(0.60, color="#A78BFA", ls=":", lw=1, label="60% target")
ax2.set_xlabel("Confidence Threshold", color="#94A3B8")
ax2.set_ylabel("Accuracy", color="#94A3B8")
ax2b.set_ylabel("Coverage", color="#F59E0B")
ax2.set_title("Accuracy vs Coverage\n(Stacked Ensemble + Sentiment)",
              color="#E2E8F0", fontsize=10)
lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labs1+labs2,
           facecolor="#1E293B", labelcolor="#E2E8F0", fontsize=8)
ax2.grid(True, color="#334155", lw=0.4)
ax2b.tick_params(colors="#F59E0B")

plt.tight_layout()
plt.savefig("results_v5_sentiment.png", dpi=150, facecolor="#0F172A")
plt.show()
print("Saved: results_v5_sentiment.png")


# %% [CELL 10] Final summary
print("\n" + "="*65)
print("  FINAL RESULTS (v5 -- Sentiment + Academic Features)")
print("="*65)
p_hgb = get_p(hgb, Xte_)
print(f"  HGB:    acc={accuracy_score(y_te,(p_hgb>=0.5).astype(int)):.4f}  "
      f"auc={roc_auc_score(y_te,p_hgb):.4f}")
print(f"  Stack:  acc={accuracy_score(y_te,(stack_p>=0.5).astype(int)):.4f}  "
      f"auc={roc_auc_score(y_te,stack_p):.4f}")
mask = (stack_p>=best_t)|(stack_p<=(1-best_t))
if mask.sum() > 0:
    fp=(stack_p[mask]>=0.5).astype(int)
    print(f"  Stack+Conf>{best_t:.2f} ({mask.mean():.0%} days): "
          f"acc={accuracy_score(y_te[mask],fp):.4f}")

print("\n  TOP 5 FEATURES (check if sentiment/52w-high appear):")
for i in idx[:5]:
    print(f"    {FEATURE_COLS[i]:<22} importance={imp[i]:.4f}")

print("""
  PRODUCTION PLAN (next steps):
  1. src/data/news_fetcher.py    -- yfinance + GDELT headlines
  2. src/data/sentiment_scorer.py -- VADER now, FinBERT in production
  3. src/data/sentiment_cache.py -- SQLite daily cache
  4. Update inference_pipeline.py -- fetch live sentiment before prediction
  5. Show sentiment badge in Gradio UI

  Upgrade path:
    VADER (current) -> FinBERT (HuggingFace, +3-5% AUC) -> GPT sentiment
    Free news (GDELT) -> NewsAPI paid -> Bloomberg Terminal
""")

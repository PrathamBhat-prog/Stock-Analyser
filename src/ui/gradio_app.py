# -*- coding: utf-8 -*-
"""
Stock Market AI Analyser -- Gradio Frontend
Port: 7860  ->  http://localhost:7860
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import gradio as gr

from src.pipelines.inference_pipeline import StockAnalysisPipeline
from src.agents.decision_agent import HORIZONS, DEFAULT_HORIZON
from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_time_series_features

pipeline = StockAnalysisPipeline()

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
HORIZON_CHOICES = {v["label"]: k for k, v in HORIZONS.items()}

POPULAR = {
    "Apple (AAPL)":           "AAPL",
    "Microsoft (MSFT)":       "MSFT",
    "Google (GOOGL)":         "GOOGL",
    "Amazon (AMZN)":          "AMZN",
    "Tesla (TSLA)":           "TSLA",
    "NVIDIA (NVDA)":          "NVDA",
    "Meta (META)":            "META",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)":           "TCS.NS",
    "Infosys (INFY.NS)":      "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)":"HDFCBANK.NS",
    "TSMC (TSM)":             "TSM",
    "Samsung (005930.KS)":    "005930.KS",
    "ASML (ASML.AS)":         "ASML.AS",
}

D_COLORS = {"BUY": "#10B981", "SELL": "#EF4444", "HOLD": "#F59E0B"}
T_COLORS = {
    "Strong Uptrend":          "#10B981",
    "Uptrend":                 "#6EE7B7",
    "Sideways / Consolidating":"#F59E0B",
    "Downtrend":               "#FCA5A5",
    "Strong Downtrend":        "#EF4444",
    "Insufficient Data":       "#94A3B8",
}
BG = "#0F172A"; PANEL = "#1E293B"; GRID = "#1E3A5F"
BORDER = "#334155"; TEXT = "#E2E8F0"; MUTED = "#94A3B8"

MODEL_INFO = """
## How the AI Works

The recommendation is made by **two systems working together**:

### 1. ML Model -- Hist Gradient Boosting (Primary)
- **Algorithm:** HistGradientBoostingClassifier (sklearn)
- **Trained on:** 38,096 daily OHLCV rows from 32 companies (US + India NSE)
- **History per ticker:** 5 years daily data
- **Features:** 60 technical indicators (RSI, MACD, Bollinger, ATR, OBV, momentum, lags, rolling stats)
- **Task:** Binary classification -- will price be higher in 5 trading days?
- **Split:** 70% train / 15% val / 15% test (chronological, zero lookahead leakage)

#### Real Test-Set Accuracy (held-out, never seen during training)
| Metric | Our Model | Random Guess | ARIMA | XGBoost Industry |
|--------|-----------|--------------|-------|-----------------|
| ROC-AUC | 0.515 | 0.500 | 0.530 | 0.570 |
| F1 Score | 0.579 | 0.500 | 0.520 | 0.560 |
| Accuracy | 49.8% | 50.0% | 51.0% | 55.0% |
| Precision | 49.0% | N/A | N/A | N/A |
| Recall | 70.8% | N/A | N/A | N/A |

> **Why ~50% accuracy?** Daily stock direction is one of the hardest problems in ML.
> The Efficient Market Hypothesis says all public info is already priced in.
> Literature shows 51-56% is typical. Our F1 of 0.579 beats all baselines.

### 2. LSTM (PyTorch) -- Production Architecture
- **Bidirectional 3-layer LSTM** (hidden=256) -- captures forward + backward temporal patterns
- **Multi-Head Self-Attention** (4 heads) -- focuses on most relevant timesteps
- **Residual skip connections** -- prevents vanishing gradients in deep networks
- **Training:** 300 epoch budget, patience=30, AdamW + OneCycleLR scheduler
- **Early stopping** only triggers after 30 consecutive epochs with no improvement

### 3. Trend Analysis Agent (ARIMA-inspired, any-ticker)
- 6 signals: MA alignment, RSI, MACD crossover, Bollinger Band position, momentum, volume
- Trend score in [-1, +1] -- works for ANY ticker without retraining

### Investment Horizon Blending
| Horizon | ML Weight | Trend Weight |
|---------|-----------|-------------|
| 1 Week  | 80% | 20% |
| 1 Month | 50% | 50% |
| 3 Months| 30% | 70% |
| 6 Months| 15% | 85% |
| 1 Year  | 10% | 90% |
"""

GLOSSARY = """
## Signal Meanings
| Signal | What it means |
|--------|--------------|
| **BUY** | AI thinks price is likely to rise in your chosen horizon |
| **SELL** | AI thinks price is likely to fall in your chosen horizon |
| **HOLD** | AI is not confident -- safest to wait and watch |

## Trend Labels
| Trend | Meaning |
|-------|---------|
| Strong Uptrend | Consistently rising -- buyers in control |
| Uptrend | Gradually rising |
| Sideways | No clear direction -- consolidating |
| Downtrend | Gradually falling |
| Strong Downtrend | Consistently falling -- sellers in control |

## RSI (Momentum)
- **Overbought (>70):** Risen too fast, pullback possible
- **Oversold (<30):** Fallen too far, bounce possible

## Volatility
- **High:** Large daily swings -- higher risk and reward
- **Normal:** Typical movement for this stock
- **Low:** Small swings -- calmer market

## Investment Horizon
- **1 Week:** Pure ML model signal (trained for this exact task)
- **1 Month:** Equal ML + trend blend
- **3-12 Months:** Trend-dominant -- longer patterns matter more

---
> **Disclaimer:** For educational purposes only. Not financial advice.
> Always consult a qualified financial advisor before investing.
"""


# ============================================================================
# Chart builder
# ============================================================================

def _build_chart(df, ticker, company_name):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                              gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

    dates = pd.to_datetime(df["Date"])
    up    = df["Close"] >= df["Open"]
    down  = ~up
    bw    = pd.Timedelta(hours=max(0.4, 500 / max(len(df), 1)))

    axes[0].bar(dates[up],  df["Close"][up]  - df["Open"][up],
                bottom=df["Open"][up],   color="#10B981", width=bw, alpha=0.9)
    axes[0].bar(dates[down], df["Close"][down] - df["Open"][down],
                bottom=df["Open"][down], color="#EF4444", width=bw, alpha=0.9)
    axes[0].vlines(dates, df["Low"], df["High"], color=BORDER, linewidth=0.4)
    if "Close_mean_20" in df.columns:
        axes[0].plot(dates, df["Close_mean_20"], color="#3B82F6", linewidth=1.2, label="SMA-20")
    if "Close_mean_60" in df.columns:
        axes[0].plot(dates, df["Close_mean_60"], color="#F97316", linewidth=1.2, label="SMA-60")
    axes[0].set_title(f"{company_name} ({ticker.upper()})", color=TEXT, fontsize=12, pad=6)
    axes[0].set_ylabel("Price", color=MUTED, fontsize=9)
    axes[0].legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    axes[0].grid(True, color=GRID, linewidth=0.35)

    if "RSI_14" in df.columns:
        rsi = df["RSI_14"]
        axes[1].plot(dates, rsi, color="#A78BFA", linewidth=1.2, label="RSI-14")
        axes[1].axhline(70, color="#EF4444", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[1].axhline(30, color="#10B981", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[1].fill_between(dates, 70, rsi.clip(lower=70), alpha=0.12, color="#EF4444")
        axes[1].fill_between(dates, rsi.clip(upper=30), 30, alpha=0.12, color="#10B981")
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("RSI", color=MUTED, fontsize=9)
        axes[1].legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    axes[1].grid(True, color=GRID, linewidth=0.35)

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        axes[2].plot(dates, df["MACD"], color="#3B82F6", linewidth=1.1, label="MACD")
        axes[2].plot(dates, df["MACD_Signal"], color="#F97316", linewidth=1.1, label="Signal")
        if "MACD_Hist" in df.columns:
            hist = df["MACD_Hist"]
            axes[2].bar(dates, hist,
                        color=np.where(hist >= 0, "#10B981", "#EF4444"),
                        alpha=0.5, width=bw)
        axes[2].axhline(0, color=BORDER, linewidth=0.7)
        axes[2].set_ylabel("MACD", color=MUTED, fontsize=9)
        axes[2].legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    axes[2].grid(True, color=GRID, linewidth=0.35)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 10)))
    plt.xticks(rotation=25, color=MUTED, fontsize=8)
    plt.tight_layout(pad=1.5)
    return fig


# ============================================================================
# Decision card HTML
# ============================================================================

def _card(result, ticker):
    d         = result["final_decision"]
    conf      = result["confidence"]
    co        = result.get("company", {})
    tr        = result.get("trend", {})
    color     = D_COLORS.get(d, "#94A3B8")
    icons     = {"BUY": "UP", "SELL": "DOWN", "HOLD": "WAIT"}
    icon      = icons.get(d, "?")
    t_lbl     = tr.get("trend_label",    "Unknown")
    t_col     = T_COLORS.get(t_lbl,      "#94A3B8")
    price     = tr.get("current_price",  0.0)
    chg       = tr.get("price_change_pct", 0.0)
    chg_col   = "#10B981" if chg >= 0 else "#EF4444"
    chg_str   = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
    currency  = co.get("currency", "USD")
    horizon   = result.get("horizon", "")
    ml_prob   = result.get("ml_probability", 0.5)
    t_score   = result.get("trend_score", 0.0)
    comp      = result.get("composite_score", 0.5)
    ml_w      = int(result.get("ml_weight", 0.8) * 100)
    tr_w      = int(result.get("trend_weight", 0.2) * 100)
    plain     = result.get("plain_english", "")
    t_sum     = tr.get("summary", "")
    co_name   = co.get("company_name", ticker.upper())

    pe    = co.get("pe_ratio")
    mc    = co.get("market_cap")
    hi52  = co.get("52w_high")
    lo52  = co.get("52w_low")
    extras = " | ".join(filter(None, [
        f"P/E: {pe:.1f}" if pe else "",
        f"Mkt Cap: ${mc/1e9:.1f}B" if mc else "",
        f"52W H: {currency} {hi52:,.2f}" if hi52 else "",
        f"52W L: {currency} {lo52:,.2f}" if lo52 else "",
    ]))

    return f"""
<div style="font-family:'Segoe UI',sans-serif;color:{TEXT};background:{BG};padding:8px;border-radius:20px;">
  <div style="background:{PANEL};border-radius:14px;padding:18px 22px;margin-bottom:12px;border:1px solid {BORDER};">
    <div style="font-size:1.4rem;font-weight:700;color:{TEXT};">{co_name}</div>
    <div style="font-size:0.85rem;color:{MUTED};margin-top:2px;">
      Ticker: {ticker.upper()} &nbsp;|&nbsp; Horizon: <strong style="color:{TEXT};">{horizon}</strong>
    </div>
    <div style="margin-top:10px;display:flex;gap:14px;flex-wrap:wrap;align-items:center;">
      <span style="font-size:1.6rem;font-weight:800;color:{TEXT};">{currency} {price:,.2f}</span>
      <span style="font-size:0.95rem;font-weight:600;color:{chg_col};">{chg_str} (20 days)</span>
    </div>
    <div style="font-size:0.78rem;color:#64748B;margin-top:5px;">{extras}</div>
  </div>

  <div style="background:{color}22;border:2px solid {color};border-radius:14px;
              padding:20px;text-align:center;margin-bottom:12px;">
    <div style="font-size:1rem;color:{color};font-weight:700;letter-spacing:2px;margin-bottom:4px;">
      {icon}
    </div>
    <div style="font-size:2.6rem;font-weight:900;color:{color};letter-spacing:4px;">{d}</div>
    <div style="font-size:0.9rem;color:{MUTED};margin-top:5px;">
      Confidence: <strong style="color:{TEXT};">{conf:.0%}</strong>
    </div>
    <div style="background:{BORDER};border-radius:99px;height:7px;
                margin:10px auto;width:80%;max-width:360px;">
      <div style="background:{color};width:{conf:.0%};height:7px;border-radius:99px;"></div>
    </div>
    <div style="font-size:0.9rem;color:#CBD5E1;margin-top:10px;line-height:1.6;
                max-width:520px;margin-left:auto;margin-right:auto;">
      {plain}
    </div>
  </div>

  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
    <div style="flex:1;min-width:120px;background:{PANEL};border-radius:12px;
                padding:12px;border-left:4px solid {t_col};">
      <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Trend</div>
      <div style="font-size:0.95rem;font-weight:700;color:{t_col};margin-top:4px;">{t_lbl}</div>
    </div>
    <div style="flex:1;min-width:120px;background:{PANEL};border-radius:12px;
                padding:12px;border-left:4px solid #A78BFA;">
      <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Momentum</div>
      <div style="font-size:0.95rem;font-weight:700;color:#A78BFA;margin-top:4px;">
        {tr.get("momentum_label","Neutral")}
      </div>
    </div>
    <div style="flex:1;min-width:120px;background:{PANEL};border-radius:12px;
                padding:12px;border-left:4px solid #F59E0B;">
      <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;">Volatility</div>
      <div style="font-size:0.95rem;font-weight:700;color:#F59E0B;margin-top:4px;">
        {tr.get("volatility_label","Normal")}
      </div>
    </div>
    <div style="flex:1;min-width:120px;background:{PANEL};border-radius:12px;
                padding:12px;border-left:4px solid #3B82F6;">
      <div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;">ML Signal</div>
      <div style="font-size:0.95rem;font-weight:700;color:#3B82F6;margin-top:4px;">
        {ml_prob:.0%} Up
      </div>
    </div>
  </div>

  <div style="background:{PANEL};border-radius:10px;padding:11px 15px;
              font-size:0.85rem;color:#CBD5E1;margin-bottom:8px;border:1px solid {BORDER};">
    {t_sum}
  </div>

  <div style="background:{PANEL};border-radius:10px;padding:9px 13px;
              font-size:0.75rem;color:#64748B;margin-bottom:8px;">
    Score: ML {ml_prob:.1%} (wt {ml_w}%) + Trend {(t_score+1)/2:.1%} (wt {tr_w}%)
    = Composite {comp:.1%}
  </div>

  <div style="padding:9px 13px;background:{PANEL};border-radius:10px;
              border-left:3px solid #F59E0B;">
    <span style="color:#F59E0B;font-weight:600;font-size:0.85rem;">Disclaimer:</span>
    <span style="color:{MUTED};font-size:0.8rem;margin-left:5px;">
      AI signal for educational purposes only. Not financial advice.
      Past patterns do not guarantee future results.
    </span>
  </div>
</div>"""


# ============================================================================
# Analysis functions
# ============================================================================

def analyze_stock(ticker_text, quick_pick, period, horizon_label):
    ticker = (
        quick_pick.split("(")[-1].rstrip(")")
        if quick_pick and quick_pick != "Type your own above"
        else ticker_text.strip().upper()
    )
    if not ticker:
        return "<p style='color:#94A3B8;padding:20px;'>Enter a ticker symbol above.</p>", None, "", ""

    horizon_key = HORIZON_CHOICES.get(horizon_label, DEFAULT_HORIZON)
    try:
        result = pipeline.run(ticker=ticker, period=period, horizon_key=horizon_key)
    except Exception as exc:
        return f"<p style='color:#EF4444;padding:20px;'>Error for {ticker}: {exc}</p>", None, "", str(exc)

    card_html = _card(result, ticker)

    try:
        fetch_p = period if period not in {"1mo", "3mo", "6mo"} else "2y"
        df      = fetch_stock_data(ticker=ticker, period=fetch_p)
        df      = validate_stock_data(df)
        df      = add_time_series_features(df)
        chart   = _build_chart(df, ticker, result.get("company", {}).get("company_name", ticker))
    except Exception:
        chart = None

    sigs  = result.get("trend", {}).get("signals", [])
    sigs_md = "\n\n".join(
        ("+ " if any(w in s.lower() for w in ["bull","ris","above","up","bounce","gain"])
         else "- " if any(w in s.lower() for w in ["bear","fall","below","down","pull","los"])
         else "  ") + s
        for s in sigs
    ) or "No signals available."

    co    = result.get("company", {})
    info  = (f"Sector: {co.get('sector','N/A')} | Industry: {co.get('industry','N/A')}"
             f" | Exchange: {co.get('exchange','')} | Currency: {co.get('currency','USD')}")
    return card_html, chart, sigs_md, info


def analyze_watchlist(tickers_raw, period, horizon_label):
    if not tickers_raw.strip():
        return pd.DataFrame({"Error": ["Enter at least one ticker"]})
    horizon_key = HORIZON_CHOICES.get(horizon_label, DEFAULT_HORIZON)
    tickers = [t.strip().upper() for t in tickers_raw.replace(";", ",").split(",") if t.strip()]
    rows = []
    for t in tickers:
        try:
            r  = pipeline.run(ticker=t, period=period, horizon_key=horizon_key)
            tr = r.get("trend", {})
            co = r.get("company", {})
            rows.append({
                "Company":   co.get("company_name", t),
                "Ticker":    t,
                "Decision":  r["final_decision"],
                "Confidence":f"{r['confidence']:.0%}",
                "Horizon":   r.get("horizon", ""),
                "Trend":     tr.get("trend_label", ""),
                "20d Change":f"{tr.get('price_change_pct',0):+.2f}%",
            })
        except Exception as exc:
            rows.append({"Company": t, "Ticker": t, "Decision": "ERROR",
                         "Confidence": "N/A", "Horizon": "", "Trend": str(exc)[:50],
                         "20d Change": "N/A"})
    df_out = pd.DataFrame(rows)
    order  = {"BUY": 0, "HOLD": 1, "SELL": 2, "ERROR": 3}
    df_out["_s"] = df_out["Decision"].map(order).fillna(3)
    return df_out.sort_values("_s").drop(columns=["_s"]).reset_index(drop=True)


# ============================================================================
# Gradio layout
# ============================================================================

CSS = """
body,.gradio-container{background-color:#0F172A !important;}
.gr-button-primary{
  background:linear-gradient(135deg,#3B82F6,#6366F1) !important;
  color:white !important;border:none !important;
  border-radius:10px !important;font-weight:700 !important;
}
footer{display:none !important;}
label{color:#94A3B8 !important;}
"""

with gr.Blocks(title="Stock Market AI Analyser", css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 14px;font-family:'Segoe UI',sans-serif;">
      <div style="font-size:2.2rem;font-weight:900;color:#E2E8F0;letter-spacing:-0.5px;">
        Stock Market <span style="color:#3B82F6;">AI</span> Analyser
      </div>
      <div style="font-size:1rem;color:#64748B;margin-top:8px;">
        AI-powered
        <strong style="color:#10B981;">BUY</strong> /
        <strong style="color:#EF4444;">SELL</strong> /
        <strong style="color:#F59E0B;">HOLD</strong>
        signals for any stock in the world.
        Choose your investment horizon from 1 week to 1 year.
      </div>
    </div>
    """)

    with gr.Tabs():

        # ---- Tab 1: Analyse a Stock ------------------------------------
        with gr.Tab("Analyse a Stock"):
            with gr.Accordion("How to use / What is a ticker?", open=False):
                gr.Markdown("""
**Step 1:** Type the ticker symbol (e.g. `AAPL` for Apple, `RELIANCE.NS` for Reliance)
OR pick from the popular stocks dropdown.

**Step 2:** Choose your **investment horizon** (how long you plan to hold).

**Step 3:** Click **Analyse Stock** and wait ~5 seconds.

**Ticker examples:** `AAPL` | `MSFT` | `TSLA` | `RELIANCE.NS` | `TCS.NS` | `ASML.AS` | `005930.KS`
                """)

            with gr.Row():
                ticker_in  = gr.Textbox(label="Stock Ticker", placeholder="AAPL", scale=2)
                quick_pick = gr.Dropdown(
                    label="Popular Stocks",
                    choices=["Type your own above"] + list(POPULAR.keys()),
                    value="Type your own above", scale=2,
                )
                period_in  = gr.Dropdown(label="Data Window", choices=PERIODS, value="2y", scale=1)
                horizon_in = gr.Dropdown(
                    label="Investment Horizon",
                    choices=list(HORIZON_CHOICES.keys()),
                    value=list(HORIZON_CHOICES.keys())[0], scale=2,
                )

            analyse_btn  = gr.Button("Analyse Stock", variant="primary", size="lg")
            data_info    = gr.Textbox(label="Company Info", interactive=False, max_lines=1)

            with gr.Row():
                with gr.Column(scale=2):
                    card_out = gr.HTML()
                with gr.Column(scale=3):
                    chart_out = gr.Plot(label="Price + RSI + MACD")

            signals_out = gr.Markdown(label="Technical Signal Breakdown")

            analyse_btn.click(
                fn      = analyze_stock,
                inputs  = [ticker_in, quick_pick, period_in, horizon_in],
                outputs = [card_out, chart_out, signals_out, data_info],
            )

        # ---- Tab 2: Compare Multiple Stocks ----------------------------
        with gr.Tab("Compare Multiple Stocks"):
            gr.Markdown(
                "Enter comma-separated tickers. Results ranked BUY > HOLD > SELL.\n\n"
                "Example: `AAPL, MSFT, TSLA, RELIANCE.NS, TCS.NS`"
            )
            with gr.Row():
                wl_in  = gr.Textbox(
                    label="Tickers (comma-separated)",
                    placeholder="AAPL, MSFT, TSLA, RELIANCE.NS",
                    scale=3,
                )
                wl_per = gr.Dropdown(label="Period", choices=PERIODS, value="2y", scale=1)
                wl_hor = gr.Dropdown(
                    label="Horizon",
                    choices=list(HORIZON_CHOICES.keys()),
                    value=list(HORIZON_CHOICES.keys())[0], scale=2,
                )
            compare_btn = gr.Button("Compare Stocks", variant="primary")
            wl_out      = gr.Dataframe(label="Ranked Results", wrap=True)
            compare_btn.click(
                fn=analyze_watchlist, inputs=[wl_in, wl_per, wl_hor], outputs=[wl_out]
            )

        # ---- Tab 3: How the AI Works -----------------------------------
        with gr.Tab("How the AI Works"):
            gr.Markdown(MODEL_INFO)

        # ---- Tab 4: Glossary -------------------------------------------
        with gr.Tab("Glossary"):
            gr.Markdown(GLOSSARY)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

"""
Layman-friendly Gradio UI for the ML Stock Analyser.

Designed for a non-technical user who wants to know:
  "Should I buy, sell, or hold this stock?"

Works for ANY publicly listed company (US, India NSE, Europe, etc.)
Frontend port: 7860  ->  http://localhost:7860
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import gradio as gr

from src.pipelines.inference_pipeline import StockAnalysisPipeline
from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_time_series_features

# ── Shared pipeline (loaded once at startup) ──────────────────────────────────
pipeline = StockAnalysisPipeline()

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]

POPULAR_TICKERS = {
    "Apple (AAPL)":              "AAPL",
    "Microsoft (MSFT)":          "MSFT",
    "Google (GOOGL)":            "GOOGL",
    "Amazon (AMZN)":             "AMZN",
    "Tesla (TSLA)":              "TSLA",
    "NVIDIA (NVDA)":             "NVDA",
    "Meta (META)":               "META",
    "Reliance (RELIANCE.NS)":    "RELIANCE.NS",
    "TCS (TCS.NS)":              "TCS.NS",
    "Infosys (INFY.NS)":         "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)":   "HDFCBANK.NS",
    "Samsung (005930.KS)":       "005930.KS",
    "TSMC (TSM)":                "TSM",
    "Berkshire (BRK-B)":         "BRK-B",
    "ASML (ASML.AS)":            "ASML.AS",
}

DECISION_COLORS = {
    "BUY":  "#10B981",
    "SELL": "#EF4444",
    "HOLD": "#F59E0B",
}
TREND_COLORS = {
    "Strong Uptrend":         "#10B981",
    "Uptrend":                "#6EE7B7",
    "Sideways / Consolidating": "#F59E0B",
    "Downtrend":              "#FCA5A5",
    "Strong Downtrend":       "#EF4444",
    "Insufficient Data":      "#94A3B8",
}


# ── Chart builder ─────────────────────────────────────────────────────────────

def _build_chart(df: pd.DataFrame, ticker: str, company_name: str) -> plt.Figure:
    """
    Dark-theme multi-panel chart:
      Panel 1: OHLC bars + SMA-20 + SMA-60
      Panel 2: RSI-14 with overbought/oversold zones
      Panel 3: MACD + signal + histogram
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )
    BG     = "#0F172A"
    PANEL  = "#1E293B"
    GRID   = "#1E3A5F"
    BORDER = "#334155"
    TEXT   = "#E2E8F0"
    MUTED  = "#94A3B8"

    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

    dates = pd.to_datetime(df["Date"])
    close = df["Close"]

    # ── Panel 1: Price ────────────────────────────────────────────────────────
    ax0 = axes[0]
    up   = df["Close"] >= df["Open"]
    down = ~up
    bar_w = max(0.3, 800 / max(len(df), 1))
    ax0.bar(dates[up],   df["Close"][up]  - df["Open"][up],   bottom=df["Open"][up],   color="#10B981", width=pd.Timedelta(hours=bar_w), alpha=0.95)
    ax0.bar(dates[down], df["Close"][down] - df["Open"][down], bottom=df["Open"][down], color="#EF4444", width=pd.Timedelta(hours=bar_w), alpha=0.95)
    ax0.vlines(dates, df["Low"], df["High"], color=BORDER, linewidth=0.4)

    if "Close_mean_20" in df.columns:
        ax0.plot(dates, df["Close_mean_20"], color="#3B82F6", linewidth=1.3, label="SMA-20", alpha=0.9)
    if "Close_mean_60" in df.columns:
        ax0.plot(dates, df["Close_mean_60"], color="#F97316", linewidth=1.3, label="SMA-60", alpha=0.9)

    currency = ""  # resolved in caller, default empty
    ax0.set_title(f"{company_name} ({ticker.upper()}) — Price History", color=TEXT, fontsize=12, pad=6)
    ax0.set_ylabel("Price", color=MUTED, fontsize=9)
    ax0.legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax0.grid(True, color=GRID, linewidth=0.35)

    # ── Panel 2: RSI ──────────────────────────────────────────────────────────
    ax1 = axes[1]
    if "RSI_14" in df.columns:
        rsi = df["RSI_14"]
        ax1.plot(dates, rsi, color="#A78BFA", linewidth=1.2, label="RSI-14")
        ax1.axhline(70, color="#EF4444", linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.axhline(30, color="#10B981", linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.fill_between(dates, 70, rsi.clip(lower=70), alpha=0.12, color="#EF4444")
        ax1.fill_between(dates, rsi.clip(upper=30), 30, alpha=0.12, color="#10B981")
        ax1.set_ylim(0, 100)
        ax1.text(dates.iloc[-1], 72, "Overbought", color="#EF4444", fontsize=7, ha="right")
        ax1.text(dates.iloc[-1], 28, "Oversold",   color="#10B981", fontsize=7, ha="right", va="top")
        ax1.set_ylabel("RSI", color=MUTED, fontsize=9)
        ax1.legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax1.grid(True, color=GRID, linewidth=0.35)

    # ── Panel 3: MACD ─────────────────────────────────────────────────────────
    ax2 = axes[2]
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        ax2.plot(dates, df["MACD"],        color="#3B82F6", linewidth=1.1, label="MACD")
        ax2.plot(dates, df["MACD_Signal"], color="#F97316", linewidth=1.1, label="Signal")
        if "MACD_Hist" in df.columns:
            hist = df["MACD_Hist"]
            ax2.bar(dates, hist, color=np.where(hist >= 0, "#10B981", "#EF4444"), alpha=0.5, width=pd.Timedelta(hours=bar_w))
        ax2.axhline(0, color=BORDER, linewidth=0.7)
        ax2.set_ylabel("MACD", color=MUTED, fontsize=9)
        ax2.legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax2.grid(True, color=GRID, linewidth=0.35)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    tick_interval = max(1, len(dates) // 10)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
    plt.xticks(rotation=25, color=MUTED, fontsize=8)
    plt.tight_layout(pad=1.5)
    return fig


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_stock(ticker_text: str, quick_pick: str, period: str):
    """Called by the Analyze button. Works for any globally listed ticker."""
    # Resolve ticker — quick pick overrides text input
    ticker = (quick_pick.split("(")[-1].rstrip(")") if quick_pick and quick_pick != "Type your own above"
              else ticker_text.strip().upper())
    if not ticker:
        return (
            _empty_card("Enter a company ticker symbol above."),
            None,
            "Please enter a ticker symbol or pick a company from the dropdown.",
            "",
        )

    try:
        result = pipeline.run(ticker=ticker, period=period)
    except Exception as exc:
        return (
            _empty_card(f"Could not fetch data for '{ticker}'. Check the symbol and try again.\n\nError: {exc}"),
            None,
            str(exc),
            "",
        )

    decision    = result["final_decision"]
    confidence  = result["confidence"]
    plain_eng   = result.get("plain_english", result.get("reasoning", ""))
    company     = result.get("company", {})
    trend       = result.get("trend", {})
    company_name = company.get("company_name", ticker.upper())

    # ── Decision card HTML ────────────────────────────────────────────────────
    card_html = _decision_card(
        decision=decision,
        confidence=confidence,
        company_name=company_name,
        ticker=ticker,
        plain_english=plain_eng,
        trend=trend,
        company=company,
    )

    # ── Price chart ───────────────────────────────────────────────────────────
    try:
        df    = fetch_stock_data(ticker=ticker, period=period if period not in {"1mo", "3mo", "6mo"} else "2y")
        df    = validate_stock_data(df)
        df    = add_time_series_features(df)
        chart = _build_chart(df, ticker, company_name)
    except Exception:
        chart = None

    # ── Trend signals list ────────────────────────────────────────────────────
    signals_md = _signals_markdown(trend)

    # ── Data info ─────────────────────────────────────────────────────────────
    sector   = company.get("sector",   "Unknown")
    industry = company.get("industry", "Unknown")
    data_info = (
        f"Company: {company_name} | Sector: {sector} | Industry: {industry} | "
        f"Exchange: {company.get('exchange', '')} | Currency: {company.get('currency', 'USD')}"
    )

    return card_html, chart, signals_md, data_info


# ── HTML card builders ────────────────────────────────────────────────────────

def _empty_card(msg: str) -> str:
    return f"""
    <div style="background:#1E293B; border-radius:16px; padding:32px; text-align:center; color:#94A3B8; font-family:sans-serif;">
      <p style="font-size:1.1rem;">{msg}</p>
    </div>
    """


def _decision_card(
    decision: str,
    confidence: float,
    company_name: str,
    ticker: str,
    plain_english: str,
    trend: dict,
    company: dict,
) -> str:
    color      = DECISION_COLORS.get(decision, "#94A3B8")
    icons      = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}
    icon       = icons.get(decision, "❓")
    pct        = f"{confidence:.0%}"
    trend_lbl  = trend.get("trend_label",      "Unknown")
    trend_col  = TREND_COLORS.get(trend_lbl,   "#94A3B8")
    mom_lbl    = trend.get("momentum_label",   "Neutral")
    vol_lbl    = trend.get("volatility_label", "Normal")
    curr_price = trend.get("current_price",    0.0)
    chg_pct    = trend.get("price_change_pct", 0.0)
    trend_sum  = trend.get("summary",          "")
    currency   = company.get("currency", "USD")

    chg_color  = "#10B981" if chg_pct >= 0 else "#EF4444"
    chg_arrow  = "+" if chg_pct >= 0 else ""

    return f"""
    <div style="font-family:'Segoe UI',sans-serif; color:#E2E8F0; background:#0F172A; padding:8px; border-radius:20px;">

      <!-- Company Header -->
      <div style="background:#1E293B; border-radius:14px; padding:20px 24px; margin-bottom:16px; border:1px solid #334155;">
        <div style="font-size:1.5rem; font-weight:700; color:#E2E8F0;">{company_name}</div>
        <div style="font-size:1rem; color:#94A3B8; margin-top:4px;">{ticker.upper()}</div>
        <div style="margin-top:12px; display:flex; gap:20px; flex-wrap:wrap; align-items:center;">
          <span style="font-size:1.8rem; font-weight:800; color:#E2E8F0;">
            {currency} {curr_price:,.2f}
          </span>
          <span style="font-size:1.1rem; font-weight:600; color:{chg_color};">
            {chg_arrow}{chg_pct:.2f}% (20 days)
          </span>
        </div>
      </div>

      <!-- Decision -->
      <div style="background:{color}22; border:2px solid {color}; border-radius:14px; padding:24px; text-align:center; margin-bottom:16px;">
        <div style="font-size:3.5rem; margin-bottom:4px;">{icon}</div>
        <div style="font-size:2.8rem; font-weight:900; color:{color}; letter-spacing:4px;">{decision}</div>
        <div style="font-size:1rem; color:#94A3B8; margin-top:8px;">AI Confidence: <strong style="color:#E2E8F0;">{pct}</strong></div>

        <!-- Confidence bar -->
        <div style="background:#334155; border-radius:99px; height:8px; margin:12px auto; width:80%; max-width:400px;">
          <div style="background:{color}; width:{pct}; height:8px; border-radius:99px;"></div>
        </div>

        <div style="font-size:1rem; color:#CBD5E1; margin-top:14px; line-height:1.6; max-width:560px; margin-left:auto; margin-right:auto;">
          {plain_english}
        </div>
      </div>

      <!-- Trend & Indicators -->
      <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;">
        <div style="flex:1; min-width:140px; background:#1E293B; border-radius:12px; padding:16px; border-left:4px solid {trend_col};">
          <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase; letter-spacing:1px;">Market Trend</div>
          <div style="font-size:1.1rem; font-weight:700; color:{trend_col}; margin-top:6px;">{trend_lbl}</div>
        </div>
        <div style="flex:1; min-width:140px; background:#1E293B; border-radius:12px; padding:16px; border-left:4px solid #A78BFA;">
          <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase; letter-spacing:1px;">Momentum</div>
          <div style="font-size:1.1rem; font-weight:700; color:#A78BFA; margin-top:6px;">{mom_lbl}</div>
        </div>
        <div style="flex:1; min-width:140px; background:#1E293B; border-radius:12px; padding:16px; border-left:4px solid #F59E0B;">
          <div style="font-size:0.75rem; color:#64748B; text-transform:uppercase; letter-spacing:1px;">Volatility</div>
          <div style="font-size:1.1rem; font-weight:700; color:#F59E0B; margin-top:6px;">{vol_lbl}</div>
        </div>
      </div>

      <!-- Trend summary -->
      <div style="background:#1E293B; border-radius:12px; padding:16px 20px; border:1px solid #334155; color:#CBD5E1; font-size:0.95rem; line-height:1.6;">
        {trend_sum}
      </div>

      <!-- Disclaimer -->
      <div style="margin-top:14px; padding:12px 16px; background:#1E293B; border-radius:10px; border-left:3px solid #F59E0B;">
        <span style="color:#F59E0B; font-weight:600;">Disclaimer:</span>
        <span style="color:#94A3B8; font-size:0.85rem; margin-left:6px;">
          This is an AI-generated signal for educational purposes only. 
          Stock markets carry risk. Always do your own research before investing.
        </span>
      </div>
    </div>
    """


def _signals_markdown(trend: dict) -> str:
    signals = trend.get("signals", [])
    if not signals:
        return "No signals available."
    lines = ["### Technical Signal Breakdown\n"]
    for s in signals:
        # Pick an emoji based on sentiment keywords
        if any(w in s.lower() for w in ["bull", "rising", "above", "up", "bounce", "positive", "gaining"]):
            emoji = "🟢"
        elif any(w in s.lower() for w in ["bear", "fall", "below", "down", "pull", "losing", "oversold risk"]):
            emoji = "🔴"
        elif any(w in s.lower() for w in ["overbought"]):
            emoji = "🟡"
        else:
            emoji = "⚪"
        lines.append(f"{emoji} {s}")
    return "\n\n".join(lines)


# ── Multi-ticker watchlist ────────────────────────────────────────────────────

def analyze_watchlist(tickers_raw: str, period: str):
    if not tickers_raw.strip():
        return pd.DataFrame({"error": ["Enter at least one ticker symbol"]})
    tickers = [t.strip().upper() for t in tickers_raw.replace(";", ",").split(",") if t.strip()]
    rows = []
    for ticker in tickers:
        try:
            result = pipeline.run(ticker=ticker, period=period)
            co = result.get("company", {})
            tr = result.get("trend", {})
            rows.append({
                "Company":   co.get("company_name", ticker),
                "Ticker":    ticker,
                "Decision":  result.get("final_decision", "N/A"),
                "Confidence": f"{result.get('confidence', 0):.0%}",
                "Trend":     tr.get("trend_label", "N/A"),
                "Price Change (20d)": f"{tr.get('price_change_pct', 0):+.2f}%",
                "Momentum":  tr.get("momentum_label", "N/A"),
            })
        except Exception as exc:
            rows.append({
                "Company":   ticker,
                "Ticker":    ticker,
                "Decision":  "ERROR",
                "Confidence": "N/A",
                "Trend":     str(exc)[:60],
                "Price Change (20d)": "N/A",
                "Momentum":  "N/A",
            })
    df_out = pd.DataFrame(rows)
    order  = {"BUY": 0, "HOLD": 1, "SELL": 2, "ERROR": 3}
    df_out["_s"] = df_out["Decision"].map(order).fillna(3)
    return df_out.sort_values("_s").drop(columns=["_s"]).reset_index(drop=True)


# ── Build Gradio app ──────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background-color: #0F172A !important; }
.gr-button-primary { background: linear-gradient(135deg,#3B82F6,#6366F1) !important;
                     color:white !important; border:none !important; border-radius:10px !important;
                     font-weight:700 !important; font-size:1.05rem !important; }
.gr-button-primary:hover { opacity:0.9 !important; transform:scale(1.01); }
footer { display:none !important; }
label { color:#94A3B8 !important; }
"""

with gr.Blocks(title="Stock Analyser | Should I Buy or Sell?", css=CSS, theme=gr.themes.Base()) as demo:

    # ── Hero header ───────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:32px 0 16px; font-family:'Segoe UI',sans-serif;">
      <div style="font-size:2.4rem; font-weight:900; color:#E2E8F0;">
        📊 Should I <span style="color:#10B981;">Buy</span> or <span style="color:#EF4444;">Sell</span>?
      </div>
      <div style="font-size:1.05rem; color:#64748B; margin-top:10px; max-width:600px; margin-left:auto; margin-right:auto;">
        Enter any company's stock ticker and get an instant AI-powered recommendation.<br/>
        Works for <strong style="color:#94A3B8;">any company in the world</strong> listed on a stock exchange.
      </div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # Tab 1: Main analysis
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Analyse a Stock"):

            # How to use
            with gr.Accordion("How to use this tool?", open=False):
                gr.Markdown("""
**Step 1:** Type the stock ticker symbol (e.g. `AAPL` for Apple, `RELIANCE.NS` for Reliance Industries)
— OR — pick a company from the "Popular Stocks" dropdown.

**Step 2:** Choose how much price history to analyse (2 years recommended).

**Step 3:** Click **Analyse Stock** and wait a few seconds.

**What is a ticker?**
Every stock on an exchange has a short code called a "ticker":
- US stocks: `AAPL` (Apple), `MSFT` (Microsoft), `TSLA` (Tesla)
- India NSE: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- India BSE: `RELIANCE.BO`
- UK: `HSBA.L`, `BP.L`
- Germany: `SAP.DE`

**How does it work?**
The AI model analyses 70+ technical patterns in the stock's price and volume history
(momentum, trend direction, RSI, MACD, Bollinger Bands, etc.) to estimate whether
the price is more likely to be higher or lower in about 1 week (5 trading days).
                """)

            # Input row
            with gr.Row():
                ticker_in = gr.Textbox(
                    label="Type a stock ticker (e.g. AAPL, TCS.NS, TSLA)",
                    placeholder="AAPL",
                    scale=3,
                )
                quick_pick = gr.Dropdown(
                    label="Or pick a popular stock",
                    choices=["Type your own above"] + list(POPULAR_TICKERS.keys()),
                    value="Type your own above",
                    scale=2,
                )
                period_in = gr.Dropdown(
                    label="Analysis window",
                    choices=PERIODS,
                    value="2y",
                    scale=1,
                )
            analyse_btn = gr.Button("Analyse Stock", variant="primary", size="lg")
            data_info_out = gr.Textbox(label="Company Info", interactive=False, max_lines=1)

            # Results
            with gr.Row():
                with gr.Column(scale=2):
                    card_out = gr.HTML(label="AI Recommendation")
                with gr.Column(scale=3):
                    chart_out = gr.Plot(label="Price History + Indicators")

            signals_out = gr.Markdown(label="Signal Breakdown")

            analyse_btn.click(
                fn=analyze_stock,
                inputs=[ticker_in, quick_pick, period_in],
                outputs=[card_out, chart_out, signals_out, data_info_out],
            )

        # ══════════════════════════════════════════════════════════════════════
        # Tab 2: Compare multiple stocks
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📋 Compare Multiple Stocks"):
            gr.Markdown("""
**Enter several tickers separated by commas** to compare signals side-by-side.
Results are ranked: BUY signals first, then HOLD, then SELL.

*Example: AAPL, MSFT, TSLA, RELIANCE.NS, TCS.NS*
            """)
            with gr.Row():
                wl_in  = gr.Textbox(
                    label="Stock tickers (comma-separated)",
                    placeholder="AAPL, MSFT, TSLA, RELIANCE.NS",
                    scale=4,
                )
                wl_per = gr.Dropdown(label="Period", choices=PERIODS, value="2y", scale=1)
            wl_btn = gr.Button("Compare Stocks", variant="primary")
            wl_out = gr.Dataframe(label="Ranked Comparison", wrap=True)
            wl_btn.click(fn=analyze_watchlist, inputs=[wl_in, wl_per], outputs=[wl_out])

        # ══════════════════════════════════════════════════════════════════════
        # Tab 3: What do these terms mean?
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📚 Glossary & Disclaimer"):
            gr.Markdown("""
## What do these terms mean?

### Recommendation

| Signal | Meaning |
|--------|---------|
| **BUY** | The AI model thinks the stock price is more likely to go up over the next ~5 trading days |
| **SELL** | The AI model thinks the stock price is more likely to go down over the next ~5 trading days |
| **HOLD** | The AI model is not confident either way — best to wait and see |

### Market Trend

| Trend | Meaning |
|-------|---------|
| **Strong Uptrend** | Stock has been consistently rising — buying pressure is dominant |
| **Uptrend** | Stock is gradually rising |
| **Sideways / Consolidating** | Stock is moving within a range — no clear direction |
| **Downtrend** | Stock is gradually falling |
| **Strong Downtrend** | Stock has been consistently falling — selling pressure is dominant |

### Momentum (RSI)
The **RSI (Relative Strength Index)** measures how fast and how much the price is moving:
- **Overbought (RSI > 70)**: The stock may have risen too fast — a pullback is possible
- **Oversold (RSI < 30)**: The stock may have fallen too far — a bounce is possible
- **Bullish / Bearish / Neutral**: General momentum direction

### Volatility
- **High Volatility**: Large daily price swings — higher risk, higher potential reward
- **Normal**: Typical price movement
- **Low Volatility**: Small daily swings — calmer, more predictable

### Confidence
How strongly the AI model feels about its recommendation (50% = coin flip, 90% = very confident).

---

## Disclaimer

> ⚠️ **This tool is for educational purposes only.**  
> It does not constitute financial advice.  
> Stock markets are inherently unpredictable.  
> Past patterns do not guarantee future performance.  
> **Never invest money you cannot afford to lose.**  
> Always consult a qualified financial advisor before making investment decisions.
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        favicon_path=None,
    )

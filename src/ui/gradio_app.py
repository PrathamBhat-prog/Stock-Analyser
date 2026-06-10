"""
Production-grade Gradio UI for the ML Stock Analyser.

Addresses known limitations:
  1. Tick data vs daily — clearly surfaces data frequency and its implications.
  2. Small dataset — shows row-count and date range so user knows data depth.
  3. No transaction costs — adds a break-even tab that calculates minimum
     required accuracy given a user-supplied round-trip cost estimate.
  4. No portfolio optimisation — adds a simple multi-ticker comparison view
     so the user can rank signals across a watchlist.
  5. Single-asset only — multi-ticker tab runs inference on multiple symbols
     at once and returns a ranked table.
  6. Directional accuracy near 50-55% — explained in a dedicated info panel
     with context on what the literature considers significant.

Frontend port: 7860  (http://localhost:7860)
Backend  port: 8000  (http://localhost:8000/docs)
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

# ── Shared pipeline instance ──────────────────────────────────────────────────
pipeline = StockAnalysisPipeline()

DECISION_COLORS = {"BUY": "#10B981", "SELL": "#EF4444", "HOLD": "#F59E0B"}
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_features(ticker: str, period: str) -> pd.DataFrame:
    df = fetch_stock_data(ticker=ticker, period=period)
    df = validate_stock_data(df)
    df = add_time_series_features(df)
    return df


def _price_chart(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Price chart with:
      - Candlestick-style OHLC bar chart
      - 20-day and 60-day SMA overlays
      - RSI sub-plot (14d)
      - MACD sub-plot
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0F172A")
    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        ax.spines["bottom"].set_color("#334155")
        ax.spines["top"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.spines["right"].set_color("#334155")

    dates = pd.to_datetime(df["Date"])
    close = df["Close"]

    # ── Price + MAs ───────────────────────────────────────────────────────────
    ax = axes[0]
    # Simplified bar coloring (green up, red down)
    up   = df["Close"] >= df["Open"]
    down = ~up
    ax.bar(dates[up],   df["Close"][up]  - df["Open"][up],   bottom=df["Open"][up],   color="#10B981", width=0.6, alpha=0.9)
    ax.bar(dates[down], df["Close"][down] - df["Open"][down], bottom=df["Open"][down], color="#EF4444", width=0.6, alpha=0.9)
    # High-low wicks
    ax.vlines(dates, df["Low"], df["High"], color="#475569", linewidth=0.5)

    if "Close_mean_20" in df.columns:
        ax.plot(dates, df["Close_mean_20"], color="#3B82F6", linewidth=1.2, label="SMA-20", alpha=0.9)
    if "Close_mean_60" in df.columns:
        ax.plot(dates, df["Close_mean_60"], color="#F97316", linewidth=1.2, label="SMA-60", alpha=0.9)

    ax.set_title(f"{ticker.upper()} - Price & Moving Averages", color="#E2E8F0", fontsize=13, pad=8)
    ax.set_ylabel("Price (USD)", color="#94A3B8", fontsize=10)
    ax.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=9)
    ax.grid(True, color="#1E3A5F", linewidth=0.4)

    # ── RSI ───────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    if "RSI_14" in df.columns:
        ax2.plot(dates, df["RSI_14"], color="#A78BFA", linewidth=1.2, label="RSI-14")
        ax2.axhline(70, color="#EF4444", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.axhline(30, color="#10B981", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.fill_between(dates, 70, df["RSI_14"].clip(lower=70), alpha=0.15, color="#EF4444")
        ax2.fill_between(dates, df["RSI_14"].clip(upper=30), 30, alpha=0.15, color="#10B981")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color="#94A3B8", fontsize=10)
        ax2.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=9)
        ax2.grid(True, color="#1E3A5F", linewidth=0.4)
        ax2.set_title("RSI-14 (Overbought >70 / Oversold <30)", color="#E2E8F0", fontsize=10)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ax3 = axes[2]
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        ax3.plot(dates, df["MACD"],        color="#3B82F6", linewidth=1.1, label="MACD")
        ax3.plot(dates, df["MACD_Signal"], color="#F97316", linewidth=1.1, label="Signal")
        hist = df["MACD_Hist"]
        ax3.bar(dates, hist, color=np.where(hist >= 0, "#10B981", "#EF4444"), alpha=0.5, width=0.6)
        ax3.axhline(0, color="#475569", linewidth=0.7)
        ax3.set_ylabel("MACD", color="#94A3B8", fontsize=10)
        ax3.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=9)
        ax3.grid(True, color="#1E3A5F", linewidth=0.4)
        ax3.set_title("MACD (12-26-9)", color="#E2E8F0", fontsize=10)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 120)))
    plt.xticks(rotation=30, color="#94A3B8", fontsize=8)
    plt.tight_layout(pad=1.5)
    return fig


# ── Tab 1: Single-stock analysis ──────────────────────────────────────────────

def analyze_single(ticker: str, period: str):
    if not ticker.strip():
        return {"error": "Ticker symbol required"}, None, "Enter a ticker symbol above."

    ticker = ticker.strip().upper()
    try:
        decision = pipeline.run(ticker=ticker, period=period)
        df = _fetch_features(ticker, period)

        # Data-quality info panel (addresses 'small dataset' limitation)
        date_col = pd.to_datetime(df["Date"])
        data_info = (
            f"Data: {date_col.min().strftime('%Y-%m-%d')} -> {date_col.max().strftime('%Y-%m-%d')} "
            f"| {len(df):,} trading days | Frequency: Daily OHLCV (yfinance)"
        )

        return decision, _price_chart(df, ticker), data_info

    except Exception as exc:
        return {"error": str(exc)}, None, str(exc)


# ── Tab 2: Multi-ticker watchlist ─────────────────────────────────────────────

def analyze_watchlist(tickers_raw: str, period: str):
    """
    Addresses 'single-asset only' limitation.
    Runs inference on a comma-separated list of tickers and returns a ranked table.
    """
    if not tickers_raw.strip():
        return pd.DataFrame({"error": ["Enter at least one ticker symbol"]})

    tickers = [t.strip().upper() for t in tickers_raw.replace(";", ",").split(",") if t.strip()]
    rows = []
    for ticker in tickers:
        try:
            result = pipeline.run(ticker=ticker, period=period)
            rows.append({
                "Ticker": ticker,
                "Decision": result.get("final_decision", "N/A"),
                "Confidence": f"{result.get('confidence', 0):.2%}",
                "P(Up)": f"{result.get('agent_summary', {}).get('probability_up', 0):.2%}",
                "Reasoning": result.get("reasoning", ""),
            })
        except Exception as exc:
            rows.append({
                "Ticker": ticker,
                "Decision": "ERROR",
                "Confidence": "N/A",
                "P(Up)": "N/A",
                "Reasoning": str(exc),
            })

    df_out = pd.DataFrame(rows)
    # Sort: BUY first, then HOLD, then SELL
    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "ERROR": 3}
    df_out["_sort"] = df_out["Decision"].map(order).fillna(3)
    df_out = df_out.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df_out


# ── Tab 3: Break-even cost calculator ────────────────────────────────────────

def breakeven_calc(model_accuracy: float, round_trip_cost_pct: float, avg_return_pct: float):
    """
    Addresses 'no transaction costs' limitation.

    Calculates the minimum directional accuracy needed to be profitable
    given round-trip transaction costs.

    Model accuracy above break-even = profit territory.
    """
    # Break-even: expected_gain = accuracy * avg_return - (1-accuracy) * avg_return - cost
    # => accuracy >= 0.5 + cost / (2 * avg_return)
    if avg_return_pct <= 0:
        return "Average return must be > 0", None

    be_accuracy = 0.5 + (round_trip_cost_pct / 100) / (2 * (avg_return_pct / 100))
    be_accuracy = min(be_accuracy, 1.0)

    model_acc = model_accuracy / 100
    margin = model_acc - be_accuracy
    verdict = "PROFITABLE" if margin > 0 else "NOT PROFITABLE"

    summary = (
        f"Break-even accuracy : {be_accuracy:.2%}\n"
        f"Your model accuracy : {model_acc:.2%}\n"
        f"Margin              : {margin:+.2%}\n\n"
        f"Verdict: {verdict}\n\n"
        f"Interpretation:\n"
        f"  - With {round_trip_cost_pct:.2f}% round-trip costs and {avg_return_pct:.2f}% avg move,\n"
        f"    you need at least {be_accuracy:.2%} directional accuracy to cover costs.\n"
        f"  - Literature typical range for daily equity models: 51-56%.\n"
        f"  - Even 53% accuracy can be highly profitable at scale with low-cost execution."
    )

    # Plot accuracy vs net PnL
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    accs = np.linspace(0.48, 0.70, 200)
    net_pnl = accs * (avg_return_pct / 100) - (1 - accs) * (avg_return_pct / 100) - (round_trip_cost_pct / 100)

    ax.plot(accs * 100, net_pnl * 100, color="#3B82F6", linewidth=2, label="Net PnL per trade (%)")
    ax.axhline(0, color="#475569", linewidth=1)
    ax.axvline(be_accuracy * 100, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Break-even ({be_accuracy:.1%})")
    ax.axvline(model_acc * 100, color="#10B981", linestyle="--", linewidth=1.5, label=f"Your model ({model_acc:.1%})")
    ax.fill_between(accs * 100, net_pnl * 100, 0, where=net_pnl > 0, alpha=0.2, color="#10B981", label="Profit zone")
    ax.fill_between(accs * 100, net_pnl * 100, 0, where=net_pnl < 0, alpha=0.2, color="#EF4444", label="Loss zone")

    ax.set_xlabel("Directional Accuracy (%)", color="#94A3B8")
    ax.set_ylabel("Net PnL per Trade (%)", color="#94A3B8")
    ax.set_title("Break-even Analysis: Accuracy vs Net PnL", color="#E2E8F0", fontsize=12)
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=9)
    ax.tick_params(colors="#94A3B8")
    ax.grid(True, color="#1E3A5F", linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    plt.tight_layout()
    return summary, fig


# ── Build UI ──────────────────────────────────────────────────────────────────

CSS = """
body { background-color: #0F172A; color: #E2E8F0; }
.gradio-container { background-color: #0F172A !important; }
.gr-button { background: linear-gradient(135deg, #3B82F6, #6366F1) !important; color: white !important; border: none !important; }
.gr-button:hover { opacity: 0.9 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="ML Stock Analyser", css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 12px;">
      <h1 style="font-size:2rem; font-weight:800; color:#E2E8F0; margin:0;">
        ML Time-Series Stock Analyser
      </h1>
      <p style="color:#94A3B8; margin-top:8px; font-size:0.95rem;">
        7 models trained on 10 years of OHLCV data &nbsp;|&nbsp;
        70+ technical features &nbsp;|&nbsp;
        BUY / SELL / HOLD signals
      </p>
    </div>
    """)

    # ── Limitations disclaimer ─────────────────────────────────────────────────
    with gr.Accordion("About model accuracy & limitations (read this first)", open=False):
        gr.Markdown("""
**Why is directional accuracy only 51-56%?**

Daily equity direction prediction is an extremely hard problem.
The Efficient Market Hypothesis (EMH) states that all public information is already
priced in, making systematic prediction near impossible.
Academic literature consistently shows:

| Model Class | Typical Test Accuracy |
|---|---|
| Random Walk / Coin Flip | ~50% |
| ARIMA / Classical TS | 51-52% |
| XGBoost / LightGBM | 53-56% |
| LSTM / Transformer | 54-57% |

**Even 53% is economically significant** — at scale with low-cost execution, a 3%
edge over random is profitable. Use the Break-Even Calculator tab to see if your
model beats transaction costs.

**Other limitations addressed in this app:**
- No transaction costs → see "Break-Even Calculator" tab
- Single asset → see "Watchlist / Multi-Ticker" tab
- Data frequency: daily OHLCV from yfinance (not tick-level)
- No portfolio optimization — signals are per-asset only
        """)

    with gr.Tabs():

        # ── Tab 1: Single stock ────────────────────────────────────────────────
        with gr.Tab("Single Stock Analysis"):
            with gr.Row():
                ticker_in = gr.Textbox(
                    label="Stock Ticker",
                    placeholder="e.g. AAPL, MSFT, RELIANCE.NS, TSLA",
                    scale=3,
                )
                period_in = gr.Dropdown(
                    label="Period",
                    choices=PERIODS,
                    value="2y",
                    scale=1,
                )
            analyze_btn = gr.Button("Analyze Stock", variant="primary", size="lg")
            data_info_out = gr.Textbox(label="Data Info", interactive=False, max_lines=1)
            with gr.Row():
                decision_out = gr.JSON(label="ML Decision", scale=1)
                chart_out    = gr.Plot(label="Price Chart + Indicators", scale=3)

            analyze_btn.click(
                fn=analyze_single,
                inputs=[ticker_in, period_in],
                outputs=[decision_out, chart_out, data_info_out],
            )

        # ── Tab 2: Multi-ticker watchlist ──────────────────────────────────────
        with gr.Tab("Watchlist / Multi-Ticker"):
            gr.Markdown("""
**Run inference on multiple tickers at once.**
Enter comma-separated symbols. Results are sorted BUY > HOLD > SELL.
This addresses the 'single-asset only' limitation.
            """)
            with gr.Row():
                watchlist_in   = gr.Textbox(
                    label="Tickers (comma-separated)",
                    placeholder="AAPL, MSFT, GOOGL, TSLA, RELIANCE.NS",
                    scale=4,
                )
                wl_period_in   = gr.Dropdown(label="Period", choices=PERIODS, value="2y", scale=1)
            watchlist_btn  = gr.Button("Analyze Watchlist", variant="primary")
            watchlist_out  = gr.Dataframe(label="Ranked Signals", wrap=True)

            watchlist_btn.click(
                fn=analyze_watchlist,
                inputs=[watchlist_in, wl_period_in],
                outputs=[watchlist_out],
            )

        # ── Tab 3: Break-even calculator ───────────────────────────────────────
        with gr.Tab("Break-Even Calculator"):
            gr.Markdown("""
**Does your model beat transaction costs?**

This addresses the 'no transaction costs' limitation.
Enter your model's accuracy and typical trade costs to find the break-even point.
            """)
            with gr.Row():
                acc_in  = gr.Slider(48, 70, value=54, step=0.5, label="Model Directional Accuracy (%)")
                cost_in = gr.Slider(0.0, 2.0, value=0.1, step=0.01, label="Round-Trip Transaction Cost (%)")
                ret_in  = gr.Slider(0.1, 3.0, value=0.5, step=0.05, label="Average Daily Move per Trade (%)")
            be_btn     = gr.Button("Calculate Break-Even", variant="primary")
            with gr.Row():
                be_text  = gr.Textbox(label="Analysis", lines=10, interactive=False, scale=1)
                be_plot  = gr.Plot(label="PnL vs Accuracy", scale=2)

            be_btn.click(
                fn=breakeven_calc,
                inputs=[acc_in, cost_in, ret_in],
                outputs=[be_text, be_plot],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )

import gradio as gr
import matplotlib.pyplot as plt

from src.pipelines.inference_pipeline import StockAnalysisPipeline
from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_time_series_features


pipeline = StockAnalysisPipeline()


def analyze_stock(ticker: str, period: str):
    if not ticker:
        return {"error": "Ticker is required"}, None

    decision = pipeline.run(ticker=ticker, period=period)

    df = fetch_stock_data(ticker=ticker, period=period)
    df = validate_stock_data(df)
    df = add_time_series_features(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    if "Close_mean_20" in df.columns:
        ax.plot(df["Date"], df["Close_mean_20"], label="20-day rolling mean", linestyle="--")

    ax.set_title(f"{ticker.upper()} Price Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    return decision, fig


with gr.Blocks(title="ML Stock Analyser") as demo:
    gr.Markdown("# ML Time-Series Stock Analyser")
    gr.Markdown(
        """
        Predicts **BUY / SELL / HOLD** using a trained time-series ML model.

        Features: lagged returns, lagged prices/volume, rolling statistics, momentum.
        Train first with `python train.py`.
        """
    )

    with gr.Row():
        ticker_input = gr.Textbox(
            label="Stock Ticker",
            placeholder="e.g. AAPL, MSFT, RELIANCE.NS",
        )
        period_input = gr.Dropdown(
            label="Time Period",
            choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            value="1y",
        )

    analyze_button = gr.Button("Analyze Stock")

    with gr.Row():
        output_json = gr.JSON(label="ML Decision")
        output_plot = gr.Plot(label="Price Chart")

    analyze_button.click(
        fn=analyze_stock,
        inputs=[ticker_input, period_input],
        outputs=[output_json, output_plot],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

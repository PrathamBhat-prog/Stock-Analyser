import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from src.pipelines.inference_pipeline import StockAnalysisPipeline
from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data
from src.data.features import add_technical_indicators


# Initialize pipeline once
pipeline = StockAnalysisPipeline()


def analyze_stock(ticker: str, period: str):
    """
    Run full analysis and return:
    - Decision JSON
    - Price chart
    """

    if not ticker:
        return {"error": "Ticker is required"}, None

    # Run decision pipeline
    decision = pipeline.run(ticker=ticker, period=period)

    # Fetch data again for plotting
    df = fetch_stock_data(ticker=ticker, period=period)
    df = validate_stock_data(df)
    df = add_technical_indicators(df)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    ax.plot(df["Date"], df["SMA_20"], label="SMA 20", linestyle="--")
    ax.plot(df["Date"], df["SMA_50"], label="SMA 50", linestyle="--")

    ax.set_title(f"{ticker.upper()} Price Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    return decision, fig


with gr.Blocks(title="Multi-Agent Stock Analyser") as demo:
    gr.Markdown("# 📈 Multi-Agent Stock Analyser")
    gr.Markdown(
        """
        Analyze stocks using a **multi-agent system**:
        - Technical Analysis Agent  
        - Risk Analysis Agent  
        - Decision Aggregation Agent  
        """
    )

    with gr.Row():
        ticker_input = gr.Textbox(
            label="Stock Ticker",
            placeholder="e.g. AAPL, MSFT, RELIANCE.NS"
        )

        period_input = gr.Dropdown(
            label="Time Period",
            choices=[
                "1mo", "3mo", "6mo",
                "1y", "2y", "5y", "10y"
            ],
            value="1y"
        )

    analyze_button = gr.Button("Analyze Stock")

    with gr.Row():
        output_json = gr.JSON(label="Analysis Result")
        output_plot = gr.Plot(label="Price & Moving Averages")

    analyze_button.click(
        fn=analyze_stock,
        inputs=[ticker_input, period_input],
        outputs=[output_json, output_plot]
    )


if __name__ == "__main__":
    demo.launch()

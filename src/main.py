from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipelines.inference_pipeline import StockAnalysisPipeline


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Stock Analyser API",
    description=(
        "AI-powered BUY / SELL / HOLD recommendations for any globally listed stock. "
        "Works for US, India NSE/BSE, European, and any yfinance-supported ticker."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = StockAnalysisPipeline()


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    ticker: str
    period: str = "2y"


class AnalyseResponse(BaseModel):
    final_decision: str
    confidence:     float
    reasoning:      str
    plain_english:  str
    company:        dict
    trend:          dict
    agent_summary:  dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.post("/analyze", response_model=AnalyseResponse)
def analyze(request: AnalyseRequest):
    """
    Analyse any stock and return a BUY / SELL / HOLD recommendation.

    Works for any ticker supported by yfinance:
      - US:    AAPL, MSFT, GOOGL, TSLA, NVDA ...
      - India: RELIANCE.NS, TCS.NS, INFY.NS ...
      - EU:    ASML.AS, SAP.DE, HSBA.L ...
      - Crypto/ETFs also supported.
    """
    try:
        result = pipeline.run(ticker=request.ticker.upper(), period=request.period)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

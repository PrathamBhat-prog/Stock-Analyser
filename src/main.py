from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.pipelines.inference_pipeline import StockAnalysisPipeline
from src.agents.decision_agent import HORIZONS, DEFAULT_HORIZON


app = FastAPI(
    title="ML Stock Analyser API",
    description=(
        "AI-powered BUY / SELL / HOLD recommendations for any globally listed stock. "
        "Supports multiple investment horizons: 1 week, 1 month, 3 months, 6 months, 1 year. "
        "Works for US, India NSE/BSE, European, and any yfinance-supported ticker."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = StockAnalysisPipeline()


class AnalyseRequest(BaseModel):
    ticker:      str
    period:      str = "2y"
    horizon_key: str = DEFAULT_HORIZON    # "5d" | "21d" | "63d" | "126d" | "252d"


class AnalyseResponse(BaseModel):
    final_decision:  str
    confidence:      float
    horizon:         str
    horizon_days:    int
    composite_score: float
    ml_probability:  float
    trend_score:     float
    ml_weight:       float
    trend_weight:    float
    reasoning:       str
    plain_english:   str
    company:         dict
    trend:           dict
    agent_summary:   dict


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0"}


@app.get("/horizons")
def list_horizons():
    """Return available investment horizons."""
    return {"horizons": HORIZONS}


@app.post("/analyze", response_model=AnalyseResponse)
def analyze(request: AnalyseRequest):
    """
    Analyse any stock for any investment horizon.

    horizon_key options:
      "5d"   - 1 week   (ML-primary)
      "21d"  - 1 month  (ML + Trend balanced)
      "63d"  - 3 months (Trend-primary)
      "126d" - 6 months (Trend-primary)
      "252d" - 1 year   (Trend-primary)
    """
    if request.horizon_key not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon_key. Choose from: {list(HORIZONS.keys())}",
        )
    try:
        result = pipeline.run(
            ticker      = request.ticker.upper(),
            period      = request.period,
            horizon_key = request.horizon_key,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

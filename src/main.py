from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.inference_pipeline import StockAnalysisPipeline


# ------------------------
# App Initialization
# ------------------------

app = FastAPI(
    title="Multi-Agent Stock Analysis API",
    description="End-to-end stock analysis using technical, risk, and decision agents",
    version="1.0.0"
)

pipeline = StockAnalysisPipeline()


# ------------------------
# Request / Response Models
# ------------------------

class StockAnalysisRequest(BaseModel):
    ticker: str
    period: str = "1y"


class StockAnalysisResponse(BaseModel):
    final_decision: str
    confidence: float
    reasoning: str
    agent_summary: dict


# ------------------------
# Health Check
# ------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ------------------------
# Inference Endpoint
# ------------------------

@app.post("/analyze", response_model=StockAnalysisResponse)
def analyze_stock(request: StockAnalysisRequest):
    try:
        result = pipeline.run(
            ticker=request.ticker,
            period=request.period
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

<#
.SYNOPSIS
    Local testing script for the ML Stock Analyser pipeline.

.DESCRIPTION
    Runs end-to-end verification:
      1. Install / update dependencies
      2. Smoke-test pipeline imports
      3. Quick training run (3 tickers, 5-year history)
      4. Open the LSTM epoch proof plot
      5. Print local testing instructions for full 10-year run

.USAGE
    .\test_local.ps1
#>

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  ML Stock Analyser - Local Test Runner" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "[1/5] Activating virtual environment ..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "[1/5] No venv found - using system Python." -ForegroundColor DarkYellow
}

# 2. Install dependencies
Write-Host "[2/5] Installing / updating dependencies ..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

# 3. Smoke test
Write-Host "[3/5] Running pipeline smoke test ..." -ForegroundColor Yellow
python verify_pipeline.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test FAILED. Aborting." -ForegroundColor Red
    exit 1
}

# 4. Quick training run
Write-Host ""
Write-Host "[4/5] Running quick training (AAPL MSFT GOOGL - 5 year period) ..." -ForegroundColor Yellow
Write-Host "      This trains all 7 models and generates docs/training_results/epoch_plot.png"
Write-Host ""
python train.py --tickers AAPL MSFT GOOGL --period 5y

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training FAILED." -ForegroundColor Red
    exit 1
}

# 5. Open epoch plot
$plotPath = "docs\training_results\epoch_plot.png"
if (Test-Path $plotPath) {
    Write-Host ""
    Write-Host "[5/5] Opening epoch proof plot ..." -ForegroundColor Yellow
    Invoke-Item $plotPath
    Write-Host "      Plot saved at: $(Resolve-Path $plotPath)" -ForegroundColor Green
} else {
    Write-Host "[5/5] Epoch plot not found at $plotPath - check training logs." -ForegroundColor Red
}

# Summary
Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "  Quick test complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Full 10-year production training:" -ForegroundColor White
Write-Host "    python train.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  View MLflow experiment results:" -ForegroundColor White
Write-Host "    mlflow ui --backend-store-uri mlruns" -ForegroundColor Cyan
Write-Host "    then open http://localhost:5000" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Launch Gradio UI (Frontend - port 7860):" -ForegroundColor White
Write-Host "    python -m src.ui.gradio_app" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Launch FastAPI Backend (port 8000):" -ForegroundColor White
Write-Host "    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Cyan
Write-Host "    then open http://localhost:8000/docs" -ForegroundColor DarkGray
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""

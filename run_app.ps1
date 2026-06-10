$ErrorActionPreference = "Stop"

# Paths
$Root = $PSScriptRoot
$VenvActivate = Join-Path $Root "venv\Scripts\Activate.ps1"
$InnerDir = Join-Path $Root "Stock-analyser"

# Activate Venv
Write-Host "Activating venv from $VenvActivate..."
& $VenvActivate

# Change Directory
Write-Host "Changing directory to $Root..."
Set-Location -Path $Root

# Run Uvicorn
Write-Host "Starting Uvicorn..."
uvicorn src.main:app --reload

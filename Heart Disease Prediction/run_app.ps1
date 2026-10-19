# ==============================================================================
# CardioVision: Heart Disease Prediction using Machine Learning
# Windows PowerShell Execution Script
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Red
Write-Host "       CardioVision: Heart Disease Prediction using Machine Learning          " -ForegroundColor White
Write-Host "==============================================================================" -ForegroundColor Red
Write-Host ""

$CurrentDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $CurrentDir

Write-Host "[1/3] Verifying Python installation..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python was not found in PATH! Please ensure Python 3.9+ is installed." -ForegroundColor Red
    pause
    exit
}

Write-Host ""
Write-Host "[2/3] Installing / Verifying requirements..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "[3/3] Launching Single-Page Web Application in browser..." -ForegroundColor Green
Write-Host "👉 http://localhost:8501" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run app.py

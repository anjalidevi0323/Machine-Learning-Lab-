@echo off
title CardioVision - Heart Disease Prediction Web Application
color 0A

echo ==============================================================================
echo        CardioVision: Heart Disease Prediction using Machine Learning
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Verifying Python installation...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.9-3.12.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Checking dependencies from requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install. Attempting to start application anyway...
)

echo.
echo [3/3] Launching Single-Page Web Application...
echo The app will open in your default browser at http://localhost:8501
echo.
python -m streamlit run app.py

pause

@echo off
echo.
echo ========================================
echo INX Future Inc - Employee Performance 
echo Prediction App Launcher
echo ========================================
echo.

cd /d "c:\Users\tonyn\Downloads\IABAC exam elizabeth"

echo Checking if virtual environment exists...
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo Virtual environment activated successfully!
) else (
    echo Virtual environment not found. Using system Python...
)

echo.
echo Installing/Updating required packages...
pip install -r src\requirements_streamlit.txt

echo.
echo Checking package installations...
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import lightgbm; print(f'LightGBM version: {lightgbm.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"

echo.
echo Navigating to src directory...
cd src

echo.
echo Starting Streamlit application...
echo.
echo The app will open in your default web browser.
echo If it doesn't open automatically, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

streamlit run streamlit_app.py

pause

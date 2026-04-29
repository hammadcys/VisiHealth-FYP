@echo off
REM VisiHealth AI - Flask Backend Launcher
REM This script starts the Flask development server

echo ============================================================
echo VisiHealth AI - Starting Backend Server
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing/Updating dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting Flask server...
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

REM Set environment variables
set FLASK_ENV=development
set FLASK_DEBUG=True

REM Start Flask app
python app.py

pause

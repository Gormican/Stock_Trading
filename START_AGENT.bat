@echo off
title Taylor's Trading Agent
color 0A
cls
echo.
echo  ============================================
echo    Taylor's Trading Agent  -  Starting Up
echo  ============================================
echo.

:: ── Step 1: Check Python ─────────────────────────────────────────────────────
echo  [1/3] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo.
    echo  *** ERROR: Python is not installed or not in PATH ***
    echo.
    echo  Please do this:
    echo    1. Go to https://www.python.org/downloads/
    echo    2. Click the big yellow Download button
    echo    3. Run the installer
    echo    4. IMPORTANT: Check the box "Add Python to PATH" before clicking Install
    echo    5. Then double-click this file again
    echo.
    pause
    exit /b 1
)
echo  Python OK!
echo.

:: ── Step 2: Install Streamlit if missing ─────────────────────────────────────
echo  [2/3] Checking Streamlit...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo  Installing packages for the first time - this takes 1-2 minutes...
    echo.
    pip install streamlit pandas yfinance feedparser vaderSentiment alpaca-py
    if %errorlevel% neq 0 (
        echo.
        echo  *** ERROR: Could not install packages ***
        echo  Try running this window as Administrator (right-click the .bat file)
        echo.
        pause
        exit /b 1
    )
    echo.
    echo  Packages installed successfully!
)
echo  Streamlit OK!
echo.

:: ── Step 3: Clear port 8501 if already in use ────────────────────────────────
echo  [3/4] Clearing port 8501 if already running...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: ── Step 4: Launch ───────────────────────────────────────────────────────────
echo  [4/4] Starting the app...
echo.
echo  ============================================
echo   App is running!
echo.
echo   Open Edge and go to:
echo.
echo        http://localhost:8501
echo.
echo   Keep this window open while using the app.
echo   Close this window to stop the app.
echo  ============================================
echo.

:: Try to open Edge automatically
set EDGE="%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if exist %EDGE% (
    echo  Opening Edge in 4 seconds...
    timeout /t 4 /nobreak >nul
    start "" %EDGE% http://localhost:8501
) else (
    echo  Opening browser in 4 seconds...
    timeout /t 4 /nobreak >nul
    start "" http://localhost:8501
)

:: Run Streamlit (this keeps the window open)
cd /d C:\claudeworkspace\Stock
python -m streamlit run "C:\claudeworkspace\Stock\app.py" --server.headless true --browser.gatherUsageStats false --server.port 8501

echo.
echo  The app has stopped. Press any key to close.
pause

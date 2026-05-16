@echo off
title Stock Agent

echo ==========================================
echo   Stock Agent - Starting up...
echo ==========================================
echo.
echo   App will be available at:
echo.
echo     On this computer:  http://localhost:8501
echo     From your phone:   http://100.123.20.109:8501
echo                        (via Tailscale)
echo.
echo   Keep this window open while using the app.
echo   Close this window to stop the app.
echo.
echo ==========================================
echo.

:: Try to open Edge automatically on this computer
set EDGE="%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if exist %EDGE% (
    echo   Opening Edge in 4 seconds...
    timeout /t 4 /nobreak >nul
    start "" %EDGE% http://localhost:8501
) else (
    echo   Opening browser in 4 seconds...
    timeout /t 4 /nobreak >nul
    start "" http://localhost:8501
)

:: Run Streamlit (this keeps the window open)
cd /d C:\claudeworkspace\Stock
python -m streamlit run "C:\claudeworkspace\Stock\app.py" --server.address 0.0.0.0 --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false

echo.
echo The app has stopped. Press any key to close.
pause

@echo off
title Push Code to GitHub
color 0A
cls
echo.
echo  ============================================
echo    Saving Your Trading Agent to GitHub
echo  ============================================
echo.

cd /d C:\Users\%USERNAME%\Stock

echo  Adding files...
git add .
git reset HEAD config.json

set /p MSG="Enter a commit message (what did you change?): "
if "%MSG%"=="" set MSG=Update trading agent

git commit -m "%MSG%"

echo.
echo  Pushing to GitHub...
git push origin main

echo.
echo  ============================================
echo   Done! Your code is safely on GitHub.
echo   config.json was NOT uploaded (your keys
echo   are safe on your computer only).
echo  ============================================
echo.
pause

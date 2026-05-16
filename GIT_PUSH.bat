@echo off
title Push Trading Agent to GitHub
color 0A
cls
echo.
echo  ============================================
echo    Push Trading Agent to GitHub
echo  ============================================
echo.

cd /d "C:\ClaudeWorkspace\Stock"

:: Check this is actually a git repo
if not exist ".git" (
    echo  ERROR: C:\ClaudeWorkspace\Stock is not a git repository.
    echo  Run "git init" and connect it to GitHub first.
    echo.
    pause
    exit /b 1
)

:: Suppress LF/CRLF warnings (Windows uses CRLF, Linux/Mac use LF — Git normalizes automatically)
git config core.autocrlf true

:: Remove .claude/ from tracking if it was accidentally added before
git rm -r --cached .claude/ >nul 2>&1

:: Show what changed
echo  Changed files:
echo  --------------------------------------------
git status --short
echo.

:: Ask for a commit message (or use a timestamp default)
set /p MSG="  Commit message (press Enter for timestamp): "
if "%MSG%"=="" (
    for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set D=%%c-%%a-%%b
    for /f "tokens=1-2 delims=: " %%a in ("%time%") do set T=%%a:%%b
    set MSG=Update %D% %T%
)

echo.
echo  Adding all files...
git add .

echo  Committing: %MSG%
git commit -m "%MSG%"

echo  Pushing to GitHub...
git push

echo.
if %ERRORLEVEL%==0 (
    color 0A
    echo  ============================================
    echo   Done! Changes pushed to GitHub.
    echo  ============================================
) else (
    color 0C
    echo  ============================================
    echo   Push failed. See error above.
    echo   Common fixes:
    echo     - Check internet connection
    echo     - Run: git remote -v   (verify remote URL)
    echo     - Run: git pull        (if remote has newer commits)
    echo  ============================================
)
echo.
pause

@echo off
cd /d "%~dp0"
title Local AI Clip Finder

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  The virtual environment is missing. Run install.bat first.
    echo.
    pause & exit /b 1
)

REM Make sure the local LLM server is up (harmless if it already is).
where ollama >nul 2>&1
if not errorlevel 1 (
    curl -s -m 2 http://127.0.0.1:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo  Starting Ollama in the background ...
        start "" /min ollama serve
        timeout /t 3 /nobreak >nul
    )
)

".venv\Scripts\python.exe" main.py serve %*
pause

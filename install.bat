@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Local AI Clip Finder - installer

echo.
echo  ============================================================
echo    LOCAL AI CLIP FINDER - installer
echo    Everything installed here runs offline. No API keys.
echo  ============================================================
echo.

REM ---------------------------------------------------------------- Python
set "PYEXE="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info < (3,13) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
    python --version >nul 2>&1
    if not errorlevel 1 (
        echo  [!] No Python 3.10-3.12 found. Falling back to the default python.
        echo      If installation fails, install Python 3.11 from python.org.
        set "PYEXE=python"
    )
)
if not defined PYEXE (
    echo  [X] Python was not found.
    echo      Install Python 3.11 from https://www.python.org/downloads/
    echo      and tick "Add python.exe to PATH".
    pause & exit /b 1
)

for /f "delims=" %%v in ('%PYEXE% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%v"
echo  [1/5] Python !PYVER!  (%PYEXE%)

REM ------------------------------------------------------- virtual env
if exist ".venv\Scripts\python.exe" (
    echo  [2/5] Reusing existing virtual environment .venv
) else (
    echo  [2/5] Creating virtual environment .venv ...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo  [X] Could not create the virtual environment.
        pause & exit /b 1
    )
)
set "VPY=%CD%\.venv\Scripts\python.exe"

REM ---------------------------------------------------------- dependencies
echo  [3/5] Installing Python packages (a few minutes the first time) ...
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%VPY%" -m pip install -r requirements-core.txt --disable-pip-version-check --timeout 30 --retries 10
if errorlevel 1 (
    echo.
    echo  [X] Dependency installation failed. Scroll up for the reason.
    pause & exit /b 1
)

REM The CUDA runtime wheels are ~1 GB and are the most likely thing to stall on
REM a slow connection. They are optional: without them Whisper falls back to the
REM CPU, so a failure here must not abort the install.
echo        Installing CUDA runtime libraries for GPU transcription (~1 GB) ...
"%VPY%" -m pip install "nvidia-cublas-cu12>=12.3.4.1" "nvidia-cudnn-cu12>=9.1.0.70" ^
        --disable-pip-version-check --timeout 30 --retries 20
if errorlevel 1 (
    echo.
    echo        [!] The CUDA libraries did not install. This is NOT fatal - the app
    echo            will transcribe on the CPU. To retry GPU support later, run:
    echo              .venv\Scripts\pip install -U nvidia-cublas-cu12 nvidia-cudnn-cu12
    echo.
)

REM ---------------------------------------------------------------- FFmpeg
echo  [4/5] Checking FFmpeg ...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo        FFmpeg not found - installing with winget ...
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements --silent
    where ffmpeg >nul 2>&1
    if errorlevel 1 (
        echo        [!] FFmpeg still not on PATH. Close and reopen this window,
        echo            or install it manually: https://www.gyan.dev/ffmpeg/builds/
    ) else ( echo        FFmpeg installed. )
) else ( echo        FFmpeg OK )

REM ---------------------------------------------------------------- Ollama
echo  [5/5] Checking the local LLM (Ollama) ...
where ollama >nul 2>&1
if errorlevel 1 (
    echo        Ollama not found - installing with winget ...
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements --silent
)
where ollama >nul 2>&1
if errorlevel 1 (
    echo        [!] Install Ollama manually from https://ollama.com/download
    echo            then run:  ollama pull qwen2.5:7b-instruct
) else (
    echo        Pulling qwen2.5:7b-instruct (~4.7 GB, fits an 8 GB GPU) ...
    start "" /wait ollama pull qwen2.5:7b-instruct
)

echo.
echo  ============================================================
echo    Done. Run  start.bat  to open the app.
echo.
echo    Whisper models download automatically on first use
echo    into  cache\models\ .
echo  ============================================================
echo.
"%VPY%" main.py check
pause

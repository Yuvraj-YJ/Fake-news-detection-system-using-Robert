@echo off
title Fake News Detector CLI Launcher
echo ======================================================================
echo           Fake News Detection System (CLI Launcher)
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not added to PATH.
    echo Please install Python and check 'Add to PATH' option.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment detected.
)
echo.

:: 3. Activate Virtual Environment
echo Activating environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo.

:: 4. Upgrade pip
echo Updating pip package...
python -m pip install --upgrade pip
echo.

:: 5. Install PyTorch CPU if needed
echo Checking PyTorch library...
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo Downloading PyTorch CPU package...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
) else (
    echo PyTorch is already installed.
)
echo.

:: 6. Install dependencies
echo Checking project dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Dependency installation failed.
    pause
    exit /b 1
)
echo All dependencies checked and satisfied.
echo.

:: 7. Run CLI application
echo Starting command line interface...
echo.
python cli_app.py

echo.
echo Application finished.
pause

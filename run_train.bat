@echo off
title Model Training Launcher
echo ======================================================================
echo           RoBERTa Model Trainer (Fine-tuning Launcher)
echo ======================================================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo Error: Virtual environment 'venv' not found!
    echo Please run run_cli.bat first to set up the environment.
    pause
    exit /b 1
)

echo Activating environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Error: Failed to activate environment.
    pause
    exit /b 1
)
echo.

echo Starting train script...
python train.py
if %errorlevel% neq 0 (
    echo Error: Training failed or was stopped.
) else (
    echo Training process finished successfully.
)

echo.
pause

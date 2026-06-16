@echo off
title Model Accuracy Evaluator
echo ======================================================================
echo           RoBERTa Model Accuracy Evaluator (Validation Set)
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

echo Starting evaluation script on dataset...
python evaluate.py
if %errorlevel% neq 0 (
    echo Error: Evaluation failed.
)

echo.
pause

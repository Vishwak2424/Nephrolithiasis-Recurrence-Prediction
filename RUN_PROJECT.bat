@echo off
title Nephrolithiasis Final Year Project - Enhanced
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 was not found.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" generate_demo_dataset.py
".venv\Scripts\python.exe" train.py
".venv\Scripts\python.exe" app.py
pause

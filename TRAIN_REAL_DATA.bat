@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run RUN_PROJECT.bat once first.
  pause
  exit /b 1
)
echo Put your real dataset at:
echo data\nephrolithiasis.csv
echo.
".venv\Scripts\python.exe" train.py
pause

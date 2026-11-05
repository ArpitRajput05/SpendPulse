@echo off
echo ===================================================
echo   Starting SpendPulse (Backend + Frontend UI)...
echo ===================================================
cd /d "%~dp0"

IF EXIST ".venv\Scripts\python.exe" (
    SET "PY_CMD=.venv\Scripts\python.exe"
) ELSE (
    SET "PY_CMD=python"
)

%PY_CMD% run_app.py
pause

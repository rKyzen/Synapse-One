@echo off
REM Synapse One - portable launcher (CMD).
REM One command to start the backend from a clean machine:
REM   - creates the virtual environment if missing
REM   - installs the package if missing
REM   - starts uvicorn on 127.0.0.1:8000
REM Usage:  start.bat   (or:  start.bat --reload  for development)

setlocal
cd /d "%~dp0"

call bootstrap.bat
if errorlevel 1 exit /b 1

echo.
echo Starting Synapse One backend ...
echo   Swagger UI : http://127.0.0.1:8000/docs
echo   Status     : http://127.0.0.1:8000/status
echo   Press CTRL+C to stop.
echo.

if "%1"=="--reload" (
    ".venv\Scripts\python.exe" -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000 --reload
) else (
    ".venv\Scripts\python.exe" -m uvicorn synapse.api:app --host 127.0.0.1 --port 8000
)
endlocal
